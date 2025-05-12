import time
import numpy
import tensorflow
import xlsxwriter
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.utils_evaluate import robustness_result_evaluation
from utils.utils_input_output import write_worksheet_header, write_worksheet_2d_data
from utils.utils_generate import approximate_by_total_derivative, data_augmentation_adult, compute_loss_grad, \
    get_search_seeds, \
    data_augmentation_compas, data_augmentation_credit, data_augmentation_bank


def project(values_tmp, epsilon):
    """
    对扰动进行投影
    Projected Gradient Descent (PGD)
    :return:
    """
    values_tmp = tensorflow.sign(values_tmp) * tensorflow.minimum(tensorflow.math.abs(values_tmp), epsilon)
    return values_tmp


def PGD_generation(model, global_seeds, attack_eps, attack_step, proj_eps):
    """
    Projected Gradient Descent 梯度攻击 adversarial attack,
    泰勒公式对生成样本标签的近似
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
        adv_x_i = x[i].copy().reshape(1, -1)

        for _ in range(attack_step):
            # 损失函数梯度，扰动变量
            y_cate = to_categorical(y_i, num_classes=2)
            grad_i = compute_loss_grad(adv_x_i, y_cate, model)
            perturbation = attack_eps * numpy.sign(grad_i)
            adv_x_i = adv_x_i + perturbation
            perturbation = project(adv_x_i - x_i, proj_eps)
            adv_x_i = numpy.array(x_i + perturbation)
            adversarial_x.append(adv_x_i)
            adversarial_y.append(y_i)
            # 检查扰动后样本预测结果是否准确
            pre = numpy.argmax(model.predict(adv_x_i), axis=1).reshape(-1, 1)[0]
            if pre != y_i:
                adversarial_cond = True

        adversarial_position.append(adversarial_cond)
    adversarial_data = numpy.concatenate((numpy.squeeze(numpy.array(adversarial_x)), numpy.array(adversarial_y)), axis=1)
    return numpy.unique(adversarial_data, axis=0),adversarial_position




def retrain_PGD(model_file, test_file, attack_eps, attack_step, proj_eps):
    """
    PGD test generation
    :return:
    """
    model = keras.models.load_model(model_file)
    test_data = numpy.load(test_file)
    adversarial_data, adversarial_position = PGD_generation(model, test_data, attack_eps, attack_step, proj_eps)
    return adversarial_data, adversarial_position


def robust_PGD_experiment(M_files, T_file, A_eps, A_step, P_eps, adversarial_data_file, adversarial_position_file):
    """
    pgd 测试样本生成
    :return:
    """
    for i in range(len(M_files)):
        # 泰勒公式估计标签
        adversarial_data, adversarial_position = retrain_PGD(M_files[i], T_file, A_eps, A_step, P_eps)
        numpy.save(adversarial_data_file[i], adversarial_data)
        numpy.save(adversarial_position_file[i], adversarial_position)
