import math
import numpy
import tensorflow
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.losses import mean_squared_error


def compute_loss_grad(x, label, model, loss_func=mean_squared_error):
    """计算模型损失函数相对于输入features的导数 dloss(f(x),y)/d(x)"""
    x = tensorflow.constant(x, dtype=tensorflow.float64)
    label = tensorflow.constant([label], dtype=tensorflow.float64)
    with tensorflow.GradientTape() as tape:
        tape.watch(x)
        loss = loss_func(label, model(x))
    gradient = tape.gradient(loss, x)
    return gradient[0].numpy()


def project(values_tmp, epsilon):
    """
    Auto Projected Gradient Descent (Auto-PGD)
    :return:
    """
    values_tmp = tensorflow.sign(values_tmp) * tensorflow.minimum(tensorflow.math.abs(values_tmp), epsilon)
    return values_tmp


def get_beta(y_0, s_0, grad_1):
    beta2 = (y_0 - 2 * s_0 * (y_0 * y_0) / (s_0 * y_0)) * grad_1 / (s_0 * y_0)
    return numpy.nan_to_num(beta2)


def ACG_evaluation(model, global_seeds, attack_eps, attack_step, proj_eps):
    """
    Projected Gradient Descent 梯度攻击 adversarial attack,
    :return:
    """
    result = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    p_0 = 0
    p_1 = 0.22
    var_w = [p_0, p_1]
    while True:
        p_j = var_w[-1] + max(var_w[-1] - var_w[-2] - 0.03, 0.06)
        if p_j > 1:
            break
        var_w.append(p_j)
    var_w = [math.ceil(p * attack_step) for p in var_w]

    for i in range(x.shape[0]):
        adversarial_cond = False
        list_f_max = []
        list_eps = []
        list_f = []

        eps = 2 * attack_eps
        x_0 = x[i].copy().reshape(1, -1)
        y_0 = y[i].copy()
        y_cate_0 = to_categorical(y_0, num_classes=2)
        grad_0 = compute_loss_grad(x_0, y_cate_0, model)

        x_pre_pre = x_0
        x_pre = x_0
        s_pre = grad_0
        predict_index = int(y_0[0])
        # list_f.append(model.predict(x_0)[0][predict_index])

        for k in range(attack_step):
            if k == 0:
                s_0 = grad_0
                perturbation = eps * numpy.sign(s_0)
                x_1 = x_0 + perturbation
                perturbation = project(x_1 - x_0, proj_eps)
                x_1 = x_0 + perturbation

                list_f.append(model.predict(x_1)[0][predict_index])
                # 检查扰动后样本预测结果是否准确
                pre = numpy.argmax(model.predict(x_1), axis=1).reshape(-1, 1)[0]
                if pre != y_0:
                    break
            else:
                grad_gap = compute_loss_grad(x_0, y_cate_0, model) - compute_loss_grad(x_1, y_cate_0, model)
                beta_1 = get_beta(grad_gap, s_0, compute_loss_grad(x_1, y_cate_0, model))
                s_1 = compute_loss_grad(x_1, y_cate_0, model) + beta_1 * s_0
                perturbation = eps * numpy.sign(s_1)
                x_2 = x_1 + perturbation
                perturbation = project(x_2 - x_0, proj_eps)
                x_2 = x_0 + perturbation

                list_f.append(model.predict(x_2)[0][predict_index])
                # 检查扰动后样本预测结果是否准确
                pre = numpy.argmax(model.predict(x_2), axis=1).reshape(-1, 1)[0]
                if pre != y_0:
                    break
                if model.predict(x_2)[0][predict_index] > model.predict(x_pre_pre)[0][predict_index]:
                    x_pre_pre = x_1
                    x_pre = x_2
                    s_pre = s_1

            if k in var_w:
                list_f_max.append(model.predict(x_pre)[0][predict_index])
                list_eps.append(eps)
                if k == 0:
                    continue
                if ACG_condition1(list_f, var_w[var_w.index(k) - 1], k) or ACG_condition2(list_eps, list_f_max, -2, -1):
                    eps = 0.5 * eps
                    s_1 = s_pre
                    x_1 = x_pre_pre
                    x_2 = x_pre

            s_0 = s_1
            x_0 = x_1
            x_1 = x_2
        result.append(numpy.concatenate((x_1, y_0.reshape(1, -1)), axis=1))
    # 保存结果
    return numpy.squeeze(numpy.array(result))


def ACG_condition1(list_f, min_index, max_index):
    """
    ACG_condition1
    :return:
    """
    p = 0
    for i in range(max_index - min_index):
        if list_f[i + min_index] < list_f[i + min_index + 1]:
            p += 1
    return p < 0.75 * (max_index - min_index)


def ACG_condition2(list_eps, list_f_max, min_index, max_index):
    """
    ACG_condition2
    :return:
    """
    if list_eps[min_index] == list_eps[max_index] and list_f_max[min_index] == list_f_max[max_index]:
        return True
    else:
        return False


def run_ACG_experiment(model_file, test_file, attack_eps, attack_step, proj_eps):
    """
    PGD test generation
    :return:
    """
    model = keras.models.load_model(model_file)
    test_data = numpy.load(test_file)
    adversarial_data = ACG_evaluation(model, test_data, attack_eps, attack_step, proj_eps)
    return adversarial_data


def robust_ACG_experiment(M_file, T_file, A_eps, A_step, P_eps, adversarial_data_file):
    """
    pgd 测试样本生成
    :return:
    """
    adversarial_data = run_ACG_experiment(M_file, T_file, A_eps, A_step, P_eps)
    numpy.save(adversarial_data_file, adversarial_data)
