import numpy
import tensorflow
from tensorflow import keras

from utils.utils_RIFair import add_perturbation_to_similar, generate_similar_items
from utils.utils_evaluate import check_item_IF
from utils.utils_generate import far_similar_C, compute_grad_EIDIG, random_pick, normal_prob, \
    compute_dataset_vote_label_C


def EIDIG_Global(model, test_item, similar_items, protected_attr, search_times, extent, D=0.5, K=0):
    """
    global generation phase of EIDIG
    :return:
    """
    generate_data = []
    generate_similar = []
    x, y = numpy.split(test_item, [-1, ], axis=1)
    x_i = x.copy().reshape(1, -1)
    # y_i = y[s_id].copy()
    bias_tag = False
    # 初始化动量
    grad1 = numpy.zeros_like(x_i).astype(float)
    grad2 = numpy.zeros_like(x_i).astype(float)
    for _ in range(search_times):
        grad1 = D * grad1 + compute_grad_EIDIG(x_i, model)
        sign1 = numpy.sign(grad1)
        # 计算预测距离最远的样本， 计算导数进行扰动
        pre_i = model.predict(x_i)
        similar_pre_i = model.predict(similar_items)
        max_x_i = similar_items[far_similar_C(similar_pre_i, pre_i)].reshape(1, -1)
        grad2 = D * grad2 + compute_grad_EIDIG(max_x_i, model)
        sign2 = numpy.sign(grad2)

        direction = numpy.zeros_like(x_i)
        for j in range(x.shape[1]):
            if j not in protected_attr and sign1[0, j] == sign2[0, j]:
                direction[0, j] = (-1) * sign1[0, j]

        perturbation = extent * direction
        perturbation = tensorflow.sign(perturbation) * tensorflow.minimum(tensorflow.math.abs(perturbation), 0.05)
        perturbation = numpy.array(perturbation)

        generate_x_i = x_i + perturbation
        x_i = generate_x_i
        generate_data.append(numpy.concatenate((x_i, y.reshape(1, -1)), axis=1))
        # 检查扰动后样本预测结果是否公平
        similar_x_i = add_perturbation_to_similar(similar_items, perturbation)
        generate_similar.append(similar_x_i)
        IF = check_item_IF(numpy.argmax(model.predict(x_i), axis=1),
                           numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, 0, K)
        if not IF:
            bias_tag = True

    return numpy.squeeze(numpy.array(generate_data)), generate_similar, bias_tag


# def EIDIG_Local(model, test_items, similar_items, protected_attr, search_times, extent, bias_tag, K=0, U=1000):
#     """
#     local generation phase of EIDIG
#     :return:
#     """
#     direction = [-1, 1]
#     generate_x = []
#     generate_similar = []
#     for search_id in range(len(test_items)):
#         x_i = test_items[search_id].copy().reshape(1, -1)
#         similar_x_i = similar_items[search_id]
#         generate_x.append(x_i)
#         generate_similar.append(similar_x_i)
#         # 计算预测距离最远的样本,计算梯度，进行局部扰动
#         pre_i = model.predict(x_i)
#         similar_pre_i = model.predict(similar_x_i)
#         max_x_i = similar_x_i[far_similar_C(similar_pre_i, pre_i)].reshape(1, -1)
#         grad1 = compute_grad_EIDIG(x_i, model)
#         grad2 = compute_grad_EIDIG(max_x_i, model)
#         p = normal_prob(grad1, grad2, protected_attr, epsilon=1e-6)
#         p0 = p.copy()
#         suc_iter = 0
#         for _ in range(search_times):
#             if suc_iter >= U:
#                 # 计算预测距离最远的样本
#                 pre_i = model.predict(x_i)[0]
#                 similar_pre_i = model.predict(similar_x_i)
#                 max_x_i = similar_x_i[far_similar_C(similar_pre_i, pre_i)]
#                 # 计算梯度，进行局部扰动
#                 grad1 = compute_grad_EIDIG(x_i, model)
#                 grad2 = compute_grad_EIDIG(max_x_i, model)
#                 p = normal_prob(grad1, grad2, protected_attr, epsilon=1e-6)
#                 suc_iter = 0
#             suc_iter += 1
#             a = random_pick(p)
#             s = random_pick([0.5, 0.5])
#             pert_x = x_i.copy()
#             pert_x[0, a] = pert_x[0, a] + direction[s] * extent
#             x_i = pert_x
#             generate_x.append(x_i)
#             # 检查扰动后样本预测结果是否公平
#             perturbation = numpy.zeros_like(x_i)
#             perturbation[0, a] = perturbation[0, a] + direction[s] * extent
#             similar_x_i = add_perturbation_to_similar_items(similar_x_i, perturbation)
#             generate_similar.append(similar_x_i)
#             IF = check_item_IF(numpy.argmax(model.predict(x_i), axis=1),
#                                numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, 0, K)
#             if not IF:
#                 bias_tag = True
#             else:
#                 # reset
#                 x_i = test_items[search_id].copy().reshape(1, -1)
#                 p = p0.copy()
#                 suc_iter = 0
#
#     return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(generate_similar)), bias_tag


def EIDIG_evaluation(model, test_item, dataset, protected_attr, global_search_times, P_eps):
    """
    对样本进行TB搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
    :return:
    """
    similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
    return EIDIG_Global(model, test_item, similar_items, protected_attr, global_search_times, P_eps)
    # return EIDIG_Local(model, G_data, G_similar, protected_attr, local_search_times, P_eps, G_Tag)


def run_EIDIG_experiment(model_file, test_file, dataset, protected_attr, G_time, P_eps):
    """
    进行EIDIG测试
    :return:
    """
    model = keras.models.load_model(model_file)
    test_data = numpy.load(test_file)

    bias_data = []
    bias_similar_data = []
    bias_tag = []
    for i in range(test_data.shape[0]):
        test_item = test_data[i].copy().reshape(1, -1)
        Bias_D, Bias_S, Bias_T = EIDIG_evaluation(model, test_item, dataset, protected_attr, G_time, P_eps)

        unique_bias_data, unique_index = numpy.unique(Bias_D, return_index=True, axis=0)

        bias_data.append(unique_bias_data)
        index = unique_index.tolist()
        for j in index:
            bias_similar_data.append(Bias_S[j])
        bias_tag.append(Bias_T)

    return bias_data, bias_similar_data, bias_tag


def fair_EIDIG_experiment(M_files, T_file, D_tag, P_attr, G_times, P_eps, EIDG_f_0, EIDG_f_1, EIDG_f_2):
    """
    EIDIG 测试样本生成
    :return:
    """
    for i in range(len(M_files)):
        Bias_D, Bias_S, Bias_C = run_EIDIG_experiment(M_files[i], T_file, D_tag, P_attr, G_times, P_eps)
        numpy.save(EIDG_f_0[i], Bias_D)
        numpy.save(EIDG_f_1[i], Bias_S)
        numpy.save(EIDG_f_2[i], Bias_C)
