import numpy
import tensorflow
from tensorflow import keras

from utils.utils_RIFair import add_perturbation_to_similar, generate_similar_items
from utils.utils_evaluate import check_item_IF
from utils.utils_generate import far_similar_C, compute_grad_loss, normal_prob, random_pick, \
    compute_dataset_vote_label_C


def ADF_Global(model, test_item, similar_items, protected_attr, search_times, extent, K=0):
    """
    global generation phase of ADF
    分类任务
    :return:
    """
    x, y = numpy.split(test_item, [-1, ], axis=1)
    x_i = x.copy().reshape(1, -1)
    for _ in range(search_times):
        grad1 = compute_grad_loss(x_i, model)
        sign1 = numpy.sign(grad1)
        pre_i = model.predict(x_i)
        similar_pre_i = model.predict(similar_items)
        max_x_i = similar_items[far_similar_C(similar_pre_i, pre_i)].reshape(1, -1)
        grad2 = compute_grad_loss(max_x_i, model)
        sign2 = numpy.sign(grad2)
        # 确定扰动方向
        direction = numpy.zeros_like(x_i)
        for j in range(x.shape[1]):
            if j not in protected_attr and sign1[j] == sign2[j]:
                direction[0, j] = sign1[j]
        perturbation = extent * direction
        perturbation = tensorflow.sign(perturbation) * tensorflow.minimum(tensorflow.math.abs(perturbation), 0.05)
        perturbation = numpy.array(perturbation)
        generate_x_i = x_i + perturbation
        x_i = generate_x_i
        # 检查扰动后样本预测结果是否公平
        similar_x_i = add_perturbation_to_similar(similar_items, perturbation)
        IF = check_item_IF(numpy.argmax(model.predict(x_i), axis=1),
                           numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, 0, K)
        if not IF:
            return numpy.concatenate((x_i, y.reshape(1, -1)), axis=1)
    return numpy.concatenate((x_i, y.reshape(1, -1)), axis=1)


def ADF_evaluation(model, test_item, dataset, protected_attr, global_search_times, P_eps):
    """
    对样本进行TB搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
    :return:
    """
    similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
    return ADF_Global(model, test_item, similar_items, protected_attr, global_search_times, P_eps)


def run_ADF_experiment(model_file, test_file, dataset, protected_attr, G_time, P_eps):
    """
    ADF fairness test generation
    :return:
    """
    model = keras.models.load_model(model_file)
    test_data = numpy.load(test_file)

    result = []
    for i in range(test_data.shape[0]):
        test_item = test_data[i].copy().reshape(1, -1)
        R = ADF_evaluation(model, test_item, dataset, protected_attr, G_time, P_eps)
        result.append(R)

    return numpy.squeeze(numpy.array(result))


def fair_ADF_experiment(M_file, T_file, D_tag, P_attr, G_times, P_eps, ADF_file):
    """
    ADF 测试样本生成
    :return:
    """
    Bias_D = run_ADF_experiment(M_file, T_file, D_tag, P_attr, G_times, P_eps)
    numpy.save(ADF_file, Bias_D)
