import numpy
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.utils_generate import data_augmentation_adult, compute_loss_grad, approximate_by_total_derivative, \
    get_search_seeds, \
    data_augmentation_compas, data_augmentation_credit, data_augmentation_bank


def FSGM_evaluation(model, global_seeds, attack_eps):
    """
    快速梯度攻击 adversarial attack,泰勒公式对生成样本标签的近似
    :return:
    """
    adversarial_x = []
    adversarial_y = []
    adversarial_position = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()
        adversarial_cond = False
        # loss function gradient
        y_cate = to_categorical(y_i, num_classes=2)
        grad_i = compute_loss_grad(x_i, y_cate, model)
        perturbation = attack_eps * numpy.sign(grad_i)
        perturbation_x_i = x_i + perturbation
        adversarial_x.append(perturbation_x_i)
        adversarial_y.append(y_i)

        # 检查扰动后样本预测结果是否准确
        pre = numpy.argmax(model.predict(perturbation_x_i), axis=1).reshape(-1, 1)[0]
        if pre != y_i:
            adversarial_cond = True

        adversarial_position.append(adversarial_cond)

    adversaril_data = numpy.concatenate((numpy.squeeze(numpy.array(adversarial_x)), numpy.array(adversarial_y)), axis=1)
    return numpy.unique(adversaril_data, axis=0),adversarial_position



def retrain_FGSM(model_file, test_file, attack_eps):
    """
    FSGM test generation
    :return:
    """
    model = keras.models.load_model(model_file)
    test_data = numpy.load(test_file)
    adversarial_data, adversarial_position = FSGM_evaluation(model, test_data, attack_eps)
    return adversarial_data, adversarial_position


def robust_FGSM_experiment(M_files, T_file, A_eps, adversarial_data_file, adversarial_position_file):
    """
    fast 测试样本生成
    :return:
    """
    for i in range(len(M_files)):
        adversarial_data, adversarial_position = retrain_FGSM(M_files[i], T_file, A_eps)
        numpy.save(adversarial_data_file[i], adversarial_data)
        numpy.save(adversarial_position_file[i], adversarial_position)
