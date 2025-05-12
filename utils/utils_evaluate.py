import pickle

import numpy
from scipy.spatial.distance import mahalanobis
from tensorflow.python.keras.losses import mean_squared_error
from utils.utils_input_output import get_search_instances, get_search_times, get_search_labels


def model_performance(datas, dist=0, K=0):
    """
    根据模型获取结果，分类模型
    :return:
    """
    ACC_result = check_ACC(datas[2], datas[1])
    IF_result = check_IF(datas[2], datas[4], datas[0], datas[3], dist, K)
    AF_result = check_AF(datas[1], datas[2], datas[4], datas[0], datas[3], dist, K)
    return ACC_result + IF_result + AF_result


def adversarial_analysis(datas, dist=0, K=0):
    """
    分析模型对抗扰动生成结果
    :return:
    """
    Acc_cond = check_dist(mean_squared_error(datas[2], datas[1]), dist)
    # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
    fair_cond = numpy.ones(Acc_cond.shape)
    for i in range(len(datas[4])):
        D_distance = mean_squared_error(datas[2].astype(float), datas[4][i].astype(float))
        Kd_distance = K * mean_squared_error(numpy.concatenate((datas[0][0], datas[0][1]), axis=1),
                                             numpy.concatenate((datas[3][i][0], datas[3][i][1]), axis=1))
        fair_cond = numpy.logical_and(fair_cond, check_dist(D_distance - Kd_distance, dist))

    # 计算 T&F 与 T|B
    TF_cond = numpy.logical_and(Acc_cond, fair_cond)
    TB_cond = numpy.logical_and(Acc_cond, ~fair_cond)
    FF_cond = numpy.logical_and(~Acc_cond, fair_cond)
    FB_cond = numpy.logical_and(~Acc_cond, ~fair_cond)

    TF = numpy.sum(TF_cond)
    TB = numpy.sum(TB_cond)
    FF = numpy.sum(FF_cond)
    FB = numpy.sum(FB_cond)
    D_sum = TF + TB + FF + FB

    return TF, TB, FF, FB, D_sum


def check_ACC(pre, label):
    """
    计算模型的预测结果与真实标记相等的比例及数量
    :return:
    """
    acc_n = numpy.sum(pre == label)
    acc_r = acc_n / label.shape[0]
    false_n = label.shape[0] - acc_n
    false_r = 1 - acc_r
    return [acc_r]


def check_dist(data, dist):
    """
    检查data中小于dist的结果
    :return:
    """
    result = []
    for i in range(data.shape[0]):
        if data[i] <= dist:
            result.append(True)
        else:
            result.append(False)
    return numpy.array(result).astype(bool)


def check_IF(pre, aug_pre, x, aug_x, dist, K):
    """
    检查 individual fair
    模型预测结果是否满足 D(f(x1),f(x2))<=Kd(x1,x2)
    :return:
    """
    IF_cond = numpy.ones(pre.shape[0])
    for i in range(len(aug_pre)):
        # D(f(x1),f(x2))<=Kd(x1,x2)+dist
        D_distance = mean_squared_error(pre.astype(float), aug_pre[i].astype(float))
        Kd_distance = K * mean_squared_error(numpy.concatenate((x[0], x[1]), axis=1),
                                             numpy.concatenate((aug_x[i][0], aug_x[i][1]), axis=1))
        IF_cond = numpy.logical_and(IF_cond, check_dist(D_distance - Kd_distance, dist))

    IF_num = numpy.sum(IF_cond)
    IF_rate = IF_num / IF_cond.shape[0]
    IB_num = numpy.sum(~IF_cond)
    IB_rate = IB_num / IF_cond.shape[0]
    return [IF_rate]


def check_AF(label, pre, aug_pre, x, aug_x, dist, K):
    """
    检查待测样本的准确公平性
    :param label: 待测样本标签
    :param pre: 待测样本预测结果
    :param aug_pre: 待测样本的相似样本预测结果集合
    :param x: 待测样本
    :param aug_x: 待测样本相似样本集合
    :param dist: 准确公平性超参数1
    :param K: 准确公平性超参数2
    :return: 待测样本准确且公平率，错误或歧视率
    """
    # D(y,f(x))<=Kd(x,x)+epsilon
    Acc_cond = check_dist(mean_squared_error(pre, label), dist)
    # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
    fair_cond = numpy.ones(Acc_cond.shape)
    for i in range(len(aug_pre)):
        D_distance = mean_squared_error(pre.astype(float), aug_pre[i].astype(float))
        Kd_distance = K * mean_squared_error(numpy.concatenate((x[0], x[1]), axis=1),
                                             numpy.concatenate((aug_x[i][0], aug_x[i][1]), axis=1))
        fair_cond = numpy.logical_and(fair_cond, check_dist(D_distance - Kd_distance, dist))

    # 计算 T&F 与 T|B
    TF_cond = numpy.logical_and(Acc_cond, fair_cond)
    TB_cond = numpy.logical_and(Acc_cond, ~fair_cond)
    FF_cond = numpy.logical_and(~Acc_cond, fair_cond)
    FB_cond = numpy.logical_and(~Acc_cond, ~fair_cond)

    TF = numpy.sum(TF_cond)
    TB = numpy.sum(TB_cond)
    FF = numpy.sum(FF_cond)
    FB = numpy.sum(FB_cond)
    D_sum = TF + TB + FF + FB

    TFR = numpy.sum(TF_cond) / D_sum
    TBR = numpy.sum(TB_cond) / D_sum
    FFR = numpy.sum(FF_cond) / D_sum
    FBR = numpy.sum(FB_cond) / D_sum

    if (TFR + FFR) == 0:
        F_recall = 0
    else:
        F_recall = TFR / (TFR + FFR)

    if (TFR + TBR) == 0:
        F_precision = 0
    else:
        F_precision = TFR / (TFR + TBR)

    if (F_recall + F_precision) == 0:
        F_F1 = 0
    else:
        F_F1 = (2 * F_recall * F_precision) / (F_recall + F_precision)

    return [TFR, TBR, FFR, FBR, F_recall, F_precision, F_F1, D_sum]


def get_AF_condition(label, pre, aug_pre, x, aug_x, dist=0, K=0):
    """
    检查待测样本的准确公平性
    :param label: 待测样本标签
    :param pre: 待测样本预测结果
    :param aug_pre: 待测样本的相似样本预测结果集合
    :param x: 待测样本
    :param aug_x: 待测样本相似样本集合
    :param dist: 准确公平性超参数1
    :param K: 准确公平性超参数2
    :return: 待测样本准确且公平率，错误或歧视率
    """
    # D(y,f(x))<=Kd(x,x)+epsilon
    Acc_cond = check_dist(mean_squared_error(pre, label), dist)
    # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
    fair_cond = numpy.ones(Acc_cond.shape)
    for i in range(len(aug_pre)):
        D_distance = mean_squared_error(pre.astype(float), aug_pre[i].astype(float))
        Kd_distance = K * mean_squared_error(numpy.concatenate((x[0], x[1]), axis=1),
                                             numpy.concatenate((aug_x[i][0], aug_x[i][1]), axis=1))
        fair_cond = numpy.logical_and(fair_cond, check_dist(D_distance - Kd_distance, dist))

    # 计算 T&F 与 T|B
    TF_cond = numpy.logical_and(Acc_cond, fair_cond)
    TB_cond = numpy.logical_and(Acc_cond, ~fair_cond)
    FF_cond = numpy.logical_and(~Acc_cond, fair_cond)
    FB_cond = numpy.logical_and(~Acc_cond, ~fair_cond)

    return [TF_cond, TB_cond, FF_cond, FB_cond]


def check_group_fairness(protected_index, privilege, predication, label):
    """
    检查不同群体中的TP,FP,TN,FN
    :return:
    """
    # 保护属性分组
    group_0 = protected_index == privilege
    group_1 = protected_index != privilege
    # 标签分组
    predication_0 = predication == 0
    predication_1 = predication == 1
    # 预测分组
    label_0 = label == 0
    label_1 = label == 0
    # statistical parity
    predication_1_group_0 = numpy.logical_and(predication_1, group_0)
    predication_1_group_1 = numpy.logical_and(predication_1, group_1)
    SPD = numpy.sum(predication_1_group_0) / numpy.sum(group_0) - numpy.sum(predication_1_group_1) / numpy.sum(group_1)
    # equal odds
    label_0_group_0 = numpy.logical_and(label_0, group_0)
    label_1_group_0 = numpy.logical_and(label_1, group_0)
    pre_1_label_0_group_0 = numpy.logical_and(predication_1, label_0_group_0)
    pre_1_label_1_group_0 = numpy.logical_and(predication_1, label_1_group_0)

    label_0_group_1 = numpy.logical_and(label_0, group_1)
    label_1_group_1 = numpy.logical_and(label_1, group_1)
    pre_1_label_0_group_1 = numpy.logical_and(predication_1, label_0_group_1)
    pre_1_label_1_group_1 = numpy.logical_and(predication_1, label_1_group_1)

    # EOD0 = numpy.sum(pre_1_label_0_group_0) / numpy.sum(label_0_group_0) - numpy.sum(pre_1_label_0_group_1) / numpy.sum(
    #     label_0_group_1)
    EOD = numpy.sum(pre_1_label_1_group_0) / numpy.sum(label_1_group_0) - numpy.sum(pre_1_label_1_group_1) / numpy.sum(
        label_1_group_1)
    # EOD = (abs(EOD0) + abs(EOD1))/2
    # 群体间精度差异
    acc = predication == label
    acc_group_0 = numpy.logical_and(acc, group_0)
    acc_group_1 = numpy.logical_and(acc, group_1)
    ACCD = numpy.sum(acc_group_0) / numpy.sum(group_0) - numpy.sum(acc_group_1) / numpy.sum(group_1)
    return [SPD, EOD, ACCD], group_0


def check_group_fairness1(protected_index0, privilege0, protected_index1, privilege1, predication, label):
    """
    多个保护属性
    检查不同群体中的TP,FP,TN,FN
    :return:
    """
    # 保护属性分组
    group_index0 = protected_index0 == privilege0
    group_index1 = protected_index1 == privilege1
    group_0 = numpy.logical_and(group_index0, group_index1)
    group_1 = ~group_0
    # 标签分组
    predication_0 = predication == 0
    predication_1 = predication == 1
    # 预测分组
    label_0 = label == 0
    label_1 = label == 0
    # statistical parity
    predication_1_group_0 = numpy.logical_and(predication_1, group_0)
    predication_1_group_1 = numpy.logical_and(predication_1, group_1)
    SPD = numpy.sum(predication_1_group_0) / numpy.sum(group_0) - numpy.sum(predication_1_group_1) / numpy.sum(group_1)
    # equal odds
    label_0_group_0 = numpy.logical_and(label_0, group_0)
    label_1_group_0 = numpy.logical_and(label_1, group_0)
    pre_1_label_0_group_0 = numpy.logical_and(predication_1, label_0_group_0)
    pre_1_label_1_group_0 = numpy.logical_and(predication_1, label_1_group_0)

    label_0_group_1 = numpy.logical_and(label_0, group_1)
    label_1_group_1 = numpy.logical_and(label_1, group_1)
    pre_1_label_0_group_1 = numpy.logical_and(predication_1, label_0_group_1)
    pre_1_label_1_group_1 = numpy.logical_and(predication_1, label_1_group_1)

    # EOD0 = numpy.sum(pre_1_label_0_group_0) / numpy.sum(label_0_group_0) - numpy.sum(pre_1_label_0_group_1) / numpy.sum(
    #     label_0_group_1)
    EOD = numpy.sum(pre_1_label_1_group_0) / numpy.sum(label_1_group_0) - numpy.sum(pre_1_label_1_group_1) / numpy.sum(
        label_1_group_1)
    # EOD = (abs(EOD0) + abs(EOD1))/2
    # 群体间精度差异
    acc = predication == label
    acc_group_0 = numpy.logical_and(acc, group_0)
    acc_group_1 = numpy.logical_and(acc, group_1)
    ACCD = numpy.sum(acc_group_0) / numpy.sum(group_0) - numpy.sum(acc_group_1) / numpy.sum(group_1)
    return [SPD, EOD, ACCD], group_0


def get_privilege_encode(dataset, feature_index, feature_name):
    """
    获取特权属性编码
    :return:
    """
    with open("../dataset/{}/data/vocab_dic.pkl".format(dataset), 'rb') as f:
        vocab_dic = pickle.load(f)

    feature_code = vocab_dic[feature_index][feature_name][0]
    fea_dim = numpy.load("../dataset/{}/data/fea_dim.npy".format(dataset)).tolist()
    if feature_index == 0:
        return feature_code
    else:
        return fea_dim[feature_index - 1] + feature_code


def check_items_AF(x1, x2, pre1, pre2, label, dist=0, K=0):
    """
    检查待测样本的准确公平性
    :param x1: 待测样本
    :param x2: 待测样本相似样本集合
    :param pre1: 待测样本预测结果
    :param pre2: 待测样本的相似样本预测结果集合
    :param label: 待测样本标签
    :param dist: 准确公平性超参数1
    :param K: 准确公平性超参数2
    :return: 待测样本准确且公平率，错误或歧视率
    """
    # 检查样本预测结果的准确性
    pre1 = numpy.argmax(pre1, axis=1).reshape(-1, 1)
    pre2 = numpy.argmax(pre2, axis=1).reshape(-1, 1)
    Acc = check_dist(mean_squared_error(pre1, label), dist)
    # 检查样本预测的公平性
    D_distance = mean_squared_error(pre1.astype(float), pre2.astype(float))
    Kd_distance = K * mean_squared_error(numpy.concatenate((x1[0], x1[1]), axis=1),
                                         numpy.concatenate((x2[0], x2[1]), axis=1))
    fair = check_dist(D_distance - Kd_distance, dist)

    TF_cond = numpy.logical_and(Acc, fair)
    TB_cond = numpy.logical_and(Acc, ~fair)
    FF_cond = numpy.logical_and(~Acc, fair)
    FB_cond = numpy.logical_and(~Acc, ~fair)
    return TF_cond, TB_cond, FF_cond, FB_cond


def get_AF_conditions(label, pre, aug_pre, x, aug_x, dist=0, K=0):
    """
    检查待测样本的准确公平性
    :param label: 待测样本标签
    :param pre: 待测样本预测结果
    :param aug_pre: 待测样本的相似样本预测结果集合
    :param x: 待测样本
    :param aug_x: 待测样本相似样本集合
    :param dist: 准确公平性超参数1
    :param K: 准确公平性超参数2
    :return: 待测样本准确且公平率，错误或歧视率
    """
    # D(y,f(x))<=Kd(x,x)+epsilon
    Acc_cond = check_dist(mean_squared_error(pre, label), dist)
    # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
    fair_cond = numpy.ones(Acc_cond.shape)
    for i in range(len(aug_pre)):
        D_distance = mean_squared_error(pre, aug_pre[i])
        Kd_distance = K * mean_squared_error(numpy.concatenate((x[0], x[1]), axis=1),
                                             numpy.concatenate((aug_x[i][0], aug_x[i][1]), axis=1))
        fair_cond = numpy.logical_and(fair_cond, check_dist(D_distance - Kd_distance, dist))

    # 计算 T&F 与 T|B
    TF_cond = numpy.logical_and(Acc_cond, fair_cond)
    TB_cond = numpy.logical_and(Acc_cond, ~fair_cond)
    FF_cond = numpy.logical_and(~Acc_cond, fair_cond)
    FB_cond = numpy.logical_and(~Acc_cond, ~fair_cond)

    return [TF_cond, TB_cond, FF_cond, FB_cond]


def mahalanobis_distance(adv_x, orig_x, cov_matrix):
    """
        计算index，value向量马氏距离
        :return:
        """
    result = []
    for i in range(len(adv_x)):
        result.append(mahalanobis(adv_x[i][0], orig_x[0], cov_matrix[0]) +
                      mahalanobis(adv_x[i][1], orig_x[1], cov_matrix[1]))
    return result


def calculate_average_mahalanobis_distance(dataset, protect_name, attack_name):
    """
    计算马氏距离均值
    :return:
    """
    data_file = ["../dataset/{}/data/{}_i.npy".format(dataset, "test"),
                 "../dataset/{}/data/{}_v.npy".format(dataset, "test"),
                 "../dataset/{}/data/{}_y.npy".format(dataset, "test")]
    index1 = numpy.load(data_file[0])
    value1 = numpy.load(data_file[1])
    cov_matrix = [numpy.cov(index1, rowvar=False), numpy.cov(value1, rowvar=False)]

    ma_dist = []
    for name1 in ["TB", "FF", "FB"]:
        generate_files = ["../dataset/{}/retrain/{}_{}_{}_i.txt".format(dataset, name1, protect_name, attack_name),
                          "../dataset/{}/retrain/{}_{}_{}_s.txt".format(dataset, name1, protect_name, attack_name),
                          "../dataset/{}/retrain/{}_{}_{}_y.txt".format(dataset, name1, protect_name, attack_name),
                          "../dataset/{}/retrain/{}_{}_{}_t.txt".format(dataset, name1, protect_name, attack_name)]
        adv_index, adv_value = get_search_instances(generate_files[0])
        search_times = get_search_times(generate_files[3]).reshape(-1, 1)
        for i in range(1, len(search_times)):
            search_times[i] += search_times[i - 1]
        for i in range(search_times.shape[0]):
            if i == 0:
                start_index = 0
                end_index = search_times[i][0]
            else:
                start_index = search_times[i - 1][0]
                end_index = search_times[i][0]
            # 计算本次搜索的马氏距离
            for ii in range(start_index, end_index):
                ma_dist.append(mahalanobis(adv_index[ii], adv_index[start_index], cov_matrix[0]) +
                               mahalanobis(adv_value[ii], adv_value[start_index], cov_matrix[1]))
    return numpy.average(ma_dist)


def calculate_average_perturbation_effect(datas):
    """
    计算各属性扰动对输出的影响
    embed_output0,embed_output1, model_pre0, model_pre1, label, times
    :return:
    """
    perturbs = []
    effects0 = []
    effects1 = []

    for i in range(datas[5].shape[0]):
        if i == 0:
            start_index = 0
            end_index = datas[5][i][0]
        else:
            start_index = datas[5][i - 1][0]
            end_index = datas[5][i][0]
        # 计算本次搜索的扰动及其影响
        for ii in range(start_index + 1, end_index):
            embed_perturb = datas[0][ii] - datas[0][ii - 1]
            pre_effect0 = numpy.sum(numpy.linalg.norm(datas[2][ii] - datas[2][ii - 1]))
            pre_effect1 = numpy.sum(numpy.linalg.norm(datas[3][ii] - datas[3][ii - 1]))
            # 计算 embedding 空间扰动模长
            norm_perturb = numpy.linalg.norm(embed_perturb, axis=1)
            perturbs.append(norm_perturb)
            # 确定对应属性扰动所导致的样本输出扰动
            position_perturb = numpy.nonzero(norm_perturb)
            perturb_effect0 = numpy.zeros_like(norm_perturb)
            perturb_effect0[position_perturb] = pre_effect0
            effects0.append(perturb_effect0)
            # 确定对应属性扰动所导致的相似样本输出扰动
            perturb_effect1 = numpy.zeros_like(norm_perturb)
            perturb_effect1[position_perturb] = pre_effect1
            effects1.append(perturb_effect1)

    avg_perturb = numpy.average(perturbs, axis=0)
    avg_effect0 = numpy.average(effects0, axis=0)
    avg_effect1 = numpy.average(effects1, axis=0)

    perturbation_effect0 = []
    perturbation_effect1 = []
    for i in range(len(avg_effect0)):
        if avg_effect0[i] == 0:
            perturbation_effect0.append(0)
        else:
            perturbation_effect0.append(avg_effect0[i] / avg_perturb[i])

        if avg_effect1[i] == 0:
            perturbation_effect1.append(0)
        else:
            perturbation_effect1.append(avg_effect1[i] / avg_perturb[i])

    return perturbation_effect0 + perturbation_effect1


def get_items_perturbation_effect(E_model, W_model, adv0):
    """
    计算各属性扰动对输出的影响
    :param E_model: embedding model
    :param W_model: whole model
    :param adv0: adv data
    :return:
    """
    # embedding层扰动
    embed_out = E_model.predict([adv0[0], adv0[1]])
    embed_perturb = []
    norm_embed_perturb = []
    for i in range(embed_out.shape[0]):
        if i == 0:
            embed_perturb.append(embed_out[i] - embed_out[i])
            norm_embed_perturb.append(numpy.sum(numpy.linalg.norm(embed_out[i] - embed_out[i], axis=1)))
        else:
            embed_perturb.append(embed_out[i] - embed_out[i - 1])
            norm_embed_perturb.append(numpy.sum(numpy.linalg.norm(embed_out[i] - embed_out[i - 1], axis=1)))
    # 样本预测结果扰动
    pre_out = W_model.predict([adv0[0], adv0[1]])
    pre_perturb = []
    norm_pre_perturb = []
    for i in range(pre_out.shape[0]):
        if i == 0:
            pre_perturb.append(pre_out[i] - pre_out[i])
            norm_pre_perturb.append(numpy.linalg.norm(pre_out[i] - pre_out[i]))
        else:
            pre_perturb.append(pre_out[i] - pre_out[i - 1])
            norm_pre_perturb.append(numpy.linalg.norm(pre_out[i] - pre_out[i - 1]))

    effects = []
    effect_dir = []
    embed_dir = []
    for i in range(len(norm_embed_perturb)):
        if norm_embed_perturb[i] == 0 or norm_pre_perturb[i] == 0:
            effects.append(0.0)
            effect_dir.append(pre_perturb[i])
            embed_dir.append(embed_perturb[i])
        else:
            effects.append(norm_pre_perturb[i] / norm_embed_perturb[i])
            effect_dir.append(pre_perturb[i] / norm_pre_perturb[i])
            embed_dir.append(embed_perturb[i] / norm_embed_perturb[i])

    return effects, norm_embed_perturb, norm_pre_perturb, effect_dir, embed_dir


def NLP_model_performance(datas, dist=0, K=0):
    """
    根据模型获取结果，分类模型
    :return:
    """
    ACC_result = check_ACC(datas[2], datas[1])
    IF_result = NLP_check_IF(datas[2], datas[4], datas[0], datas[3], dist, K)
    AF_result = NLP_check_AF(datas[1], datas[2], datas[4], datas[0], datas[3], dist, K)
    return ACC_result + IF_result + AF_result


def NLP_check_IF(pre, aug_pre, x, aug_x, dist, K):
    """
    检查 individual fair
    模型预测结果是否满足 D(f(x1),f(x2))<=Kd(x1,x2)
    :return:
    """
    IF_cond = numpy.ones(pre.shape[0])
    for i in range(len(aug_pre)):
        # D(f(x1),f(x2))<=Kd(x1,x2)+dist
        D_distance = mean_squared_error(pre.astype(float), aug_pre[i].astype(float))
        Kd_distance = K * mean_squared_error(x, aug_x[i])
        IF_cond = numpy.logical_and(IF_cond, check_dist(D_distance - Kd_distance, dist))

    IF_num = numpy.sum(IF_cond)
    IF_rate = IF_num / IF_cond.shape[0]
    IB_num = numpy.sum(~IF_cond)
    IB_rate = IB_num / IF_cond.shape[0]
    return [IF_rate]


def NLP_check_AF(label, pre, aug_pre, x, aug_x, dist, K):
    """
    检查待测样本的准确公平性
    :param label: 待测样本标签
    :param pre: 待测样本预测结果
    :param aug_pre: 待测样本的相似样本预测结果集合
    :param x: 待测样本
    :param aug_x: 待测样本相似样本集合
    :param dist: 准确公平性超参数1
    :param K: 准确公平性超参数2
    :return: 待测样本准确且公平率，错误或歧视率
    """
    # D(y,f(x))<=Kd(x,x)+epsilon
    Acc_cond = check_dist(mean_squared_error(pre, label), dist)
    # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
    fair_cond = numpy.ones(Acc_cond.shape)
    for i in range(len(aug_pre)):
        D_distance = mean_squared_error(pre.astype(float), aug_pre[i].astype(float))
        Kd_distance = K * mean_squared_error(x, aug_x[i])
        fair_cond = numpy.logical_and(fair_cond, check_dist(D_distance - Kd_distance, dist))

    # 计算 T&F 与 T|B
    TF_cond = numpy.logical_and(Acc_cond, fair_cond)
    TB_cond = numpy.logical_and(Acc_cond, ~fair_cond)
    FF_cond = numpy.logical_and(~Acc_cond, fair_cond)
    FB_cond = numpy.logical_and(~Acc_cond, ~fair_cond)

    TF = numpy.sum(TF_cond)
    TB = numpy.sum(TB_cond)
    FF = numpy.sum(FF_cond)
    FB = numpy.sum(FB_cond)
    D_sum = TF + TB + FF + FB

    TFR = numpy.sum(TF_cond) / D_sum
    TBR = numpy.sum(TB_cond) / D_sum
    FFR = numpy.sum(FF_cond) / D_sum
    FBR = numpy.sum(FB_cond) / D_sum

    if (TFR + FFR) == 0:
        F_recall = 0
    else:
        F_recall = TFR / (TFR + FFR)

    if (TFR + TBR) == 0:
        F_precision = 0
    else:
        F_precision = TFR / (TFR + TBR)

    if (F_recall + F_precision) == 0:
        F_F1 = 0
    else:
        F_F1 = (2 * F_recall * F_precision) / (F_recall + F_precision)

    return [TFR, TBR, FFR, FBR, F_recall, F_precision, F_F1, D_sum]


def NLP_adversarial_analysis(datas, dist=0, K=0):
    """
    分析模型对抗扰动生成结果
    :return:
    """
    Acc_cond = check_dist(mean_squared_error(datas[2], datas[1]), dist)
    # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
    fair_cond = numpy.ones(Acc_cond.shape)
    for i in range(len(datas[4])):
        D_distance = mean_squared_error(datas[2].astype(float), datas[4][i].astype(float))
        Kd_distance = K * mean_squared_error(datas[0], datas[3][i])
        fair_cond = numpy.logical_and(fair_cond, check_dist(D_distance - Kd_distance, dist))

    # 计算 T&F 与 T|B
    TF_cond = numpy.logical_and(Acc_cond, fair_cond)
    TB_cond = numpy.logical_and(Acc_cond, ~fair_cond)
    FF_cond = numpy.logical_and(~Acc_cond, fair_cond)
    FB_cond = numpy.logical_and(~Acc_cond, ~fair_cond)

    TF = numpy.sum(TF_cond)
    TB = numpy.sum(TB_cond)
    FF = numpy.sum(FF_cond)
    FB = numpy.sum(FB_cond)
    D_sum = TF + TB + FF + FB

    return TF, TB, FF, FB, D_sum


def check_NLP_items_AF(x1, x2, pre1, pre2, label, dist=0, K=0):
    """
    检查待测样本的准确公平性
    :param x1: 待测样本
    :param x2: 待测样本相似样本集合
    :param pre1: 待测样本预测结果
    :param pre2: 待测样本的相似样本预测结果集合
    :param label: 待测样本标签
    :param dist: 准确公平性超参数1
    :param K: 准确公平性超参数2
    :return: 待测样本准确且公平率，错误或歧视率
    """
    # 检查样本预测结果的准确性
    pre1 = numpy.argmax(pre1, axis=1).reshape(-1, 1)
    pre2 = numpy.argmax(pre2, axis=1).reshape(-1, 1)
    Acc = check_dist(mean_squared_error(pre1, label), dist)
    # 检查样本预测的公平性
    D_distance = mean_squared_error(pre1.astype(float), pre2.astype(float))
    Kd_distance = K * mean_squared_error(x1, x2)
    fair = check_dist(D_distance - Kd_distance, dist)

    TF_cond = numpy.logical_and(Acc, fair)
    TB_cond = numpy.logical_and(Acc, ~fair)
    FF_cond = numpy.logical_and(~Acc, fair)
    FB_cond = numpy.logical_and(~Acc, ~fair)
    return TF_cond, TB_cond, FF_cond, FB_cond


def get_first_true_item(items):
    """

    :return:
    """
    for i in range(len(items)):
        if items[i]:
            return i
    return None


def get_NLP_items_perturbation_effect(E_model, W_model, adv0):
    """
    计算各属性扰动对输出的影响
    :param E_model: embedding model
    :param W_model: whole model
    :param adv0: adv data
    :return:
    """
    # embedding层扰动
    embed_out = E_model.predict(adv0)
    embed_perturb = []
    norm_embed_perturb = []
    for i in range(embed_out.shape[0]):
        if i == 0:
            embed_perturb.append(embed_out[i] - embed_out[i])
            norm_embed_perturb.append(numpy.sum(numpy.linalg.norm(embed_out[i] - embed_out[i], axis=1)))
        else:
            embed_perturb.append(embed_out[i] - embed_out[i - 1])
            norm_embed_perturb.append(numpy.sum(numpy.linalg.norm(embed_out[i] - embed_out[i - 1], axis=1)))
    # 样本预测结果扰动
    pre_out = W_model.predict(adv0)
    pre_perturb = []
    norm_pre_perturb = []
    for i in range(pre_out.shape[0]):
        if i == 0:
            pre_perturb.append(pre_out[i] - pre_out[i])
            norm_pre_perturb.append(numpy.linalg.norm(pre_out[i] - pre_out[i]))
        else:
            pre_perturb.append(pre_out[i] - pre_out[i - 1])
            norm_pre_perturb.append(numpy.linalg.norm(pre_out[i] - pre_out[i - 1]))

    effects = []
    effect_dir = []
    embed_dir = []
    for i in range(len(norm_embed_perturb)):
        if norm_embed_perturb[i] == 0 or norm_pre_perturb[i] == 0:
            effects.append(0.0)
            effect_dir.append(pre_perturb[i])
            embed_dir.append(embed_perturb[i])
        else:
            effects.append(norm_pre_perturb[i] / norm_embed_perturb[i])
            effect_dir.append(pre_perturb[i] / norm_pre_perturb[i])
            embed_dir.append(embed_perturb[i] / norm_embed_perturb[i])

    return effects, norm_embed_perturb, norm_pre_perturb, effect_dir, embed_dir

# def get_generate_dataset_prob(data_name, model_name, generate_model_name):
#     """
#     获取重训练数据
#     :return:
#     """
#     model_file = "../dataset/{}/model/{}.".format(data_name, model_name)
#     custom_layers = {'NumericalAndCategoryEmbedding': NumericalAndCategoryEmbedding,
#                      'AutoIntTransformerBlock': AutoIntTransformerBlock}
#     model = load_model(model_file, custom_objects=custom_layers)
#
#     # 获取TB生成结果
#     R_raw_file = ["../dataset/{}/test/P_TB_{}_i.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_TB_{}_v.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_TB_{}_y.npy".format(data_name, generate_model_name)]
#     R_aug_file = ["../dataset/{}/test/P_TB_aug_{}_i.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_TB_aug_{}_v.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_TB_search_time_{}_y.npy".format(data_name, generate_model_name)]
#
#     TB_index = numpy.squeeze(numpy.load(R_raw_file[0]))
#     TB_value = numpy.squeeze(numpy.load(R_raw_file[1]))
#     TB_label = numpy.squeeze(numpy.load(R_raw_file[2])).reshape(-1, 1)
#
#     aug_TB_index = numpy.squeeze(numpy.load(R_aug_file[0]))
#     aug_TB_value = numpy.squeeze(numpy.load(R_aug_file[1]))
#
#     # 获取FB生成结果
#     R_raw_file = ["../dataset/{}/test/P_FB_{}_i.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_FB_{}_v.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_FB_{}_y.npy".format(data_name, generate_model_name)]
#     R_aug_file = ["../dataset/{}/test/P_FB_aug_{}_i.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_FB_aug_{}_v.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_FB_search_time_{}_y.npy".format(data_name, generate_model_name)]
#
#     FB_index = numpy.squeeze(numpy.load(R_raw_file[0]))
#     FB_value = numpy.squeeze(numpy.load(R_raw_file[1]))
#     FB_label = numpy.squeeze(numpy.load(R_raw_file[2])).reshape(-1, 1)
#
#     aug_FB_index = numpy.squeeze(numpy.load(R_aug_file[0]))
#     aug_FB_value = numpy.squeeze(numpy.load(R_aug_file[1]))
#
#     # 合并TB，FB生成数据
#     D_index = numpy.concatenate((TB_index, FB_index), axis=0)
#     D_value = numpy.concatenate((TB_value, FB_value), axis=0)
#     D_label = numpy.concatenate((TB_label, FB_label), axis=0)
#
#     aug_D_index = numpy.concatenate((aug_TB_index, aug_FB_index), axis=0)
#     aug_D_value = numpy.concatenate((aug_TB_value, aug_FB_value), axis=0)
#
#     # 获取FF生成结果
#     R_raw_file = ["../dataset/{}/test/P_FF_{}_i.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_FF_{}_v.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_FF_{}_y.npy".format(data_name, generate_model_name)]
#     R_aug_file = ["../dataset/{}/test/P_FF_aug_{}_i.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_FF_aug_{}_v.npy".format(data_name, generate_model_name),
#                   "../dataset/{}/test/P_FF_search_time_{}_y.npy".format(data_name, generate_model_name)]
#
#     FF_index = numpy.squeeze(numpy.load(R_raw_file[0]))
#     FF_value = numpy.squeeze(numpy.load(R_raw_file[1]))
#     FF_label = numpy.squeeze(numpy.load(R_raw_file[2])).reshape(-1, 1)
#
#     aug_FF_index = numpy.squeeze(numpy.load(R_aug_file[0]))
#     aug_FF_value = numpy.squeeze(numpy.load(R_aug_file[1]))
#
#     # 合并TB，FB,FF生成数据
#     D_index = numpy.concatenate((D_index, FF_index), axis=0)
#     D_value = numpy.concatenate((D_value, FF_value), axis=0)
#     D_label = numpy.concatenate((D_label, FF_label), axis=0)
#
#     raw_input = [D_index, D_value]
#     pre = numpy.argmax(model.predict(raw_input), axis=1).reshape(-1, 1)
#
#     aug_D_index = numpy.concatenate((aug_D_index, aug_FF_index), axis=0)
#     aug_D_value = numpy.concatenate((aug_D_value, aug_FF_value), axis=0)
#
#     aug_input = []
#     aug_pre = []
#     for i in range(aug_D_index.shape[1]):
#         aug_input.append([aug_D_index[:, i, :], aug_D_value[:, i, :]])
#         aug_pre.append(numpy.argmax(model.predict([aug_D_index[:, i, :], aug_D_value[:, i, :]]), axis=1).reshape(-1, 1))
#
#     return raw_input, D_label, pre, aug_input, aug_pre

# def data_augmentation_adult(data, protected_index):
#     """
#     对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
#     :return:
#     """
#     aug = []
#     for i in range(data.shape[0]):
#         data_list = []
#         for a_0 in [uniform(17, 30), uniform(30, 60), uniform(60, 90)]:
#             for a_1 in [uniform(0, 2), uniform(2, 4)]:
#                 for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
#                     aug_data = data[i].tolist()
#                     aug_data[protected_index[0]] = round(a_0) / (90 - 17)  # 归一化
#                     aug_data[protected_index[1]] = round(a_1) / (4)
#                     aug_data[protected_index[2]] = round(a_2) / (1)
#                     # data_list.append(aug_data)
#                     # 生成测试数据集相似样本时，去除真实标记
#                     data_list.append(aug_data[:-1])
#         aug.append(data_list)
#     return numpy.squeeze(numpy.array(aug))
#
#
# def data_augmentation_credit(data, protected_index):
#     """
#     对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
#     :return:
#     """
#     aug = []
#     for i in range(data.shape[0]):
#         data_list = []
#         for a_0 in [0, 1]:
#             for a_1 in [uniform(19, 35), uniform(35, 55), uniform(55, 75)]:
#                 aug_data = data[i].tolist()
#                 aug_data[protected_index[0]] = a_0
#                 aug_data[protected_index[1]] = round(a_1) / (75 - 19)  # 归一化
#                 # data_list.append(aug_data)
#                 # 生成测试数据集相似样本时，去除真实标记
#                 data_list.append(aug_data[:-1])
#         aug.append(data_list)
#     return numpy.squeeze(numpy.array(aug))
#
#
# def data_augmentation_compas(data, protected_index):
#     """
#     对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
#     :return:
#     """
#     aug = []
#     for i in range(data.shape[0]):
#         data_list = []
#         for a_0 in [uniform(18, 30), uniform(30, 60), uniform(60, 96)]:
#             for a_1 in [uniform(0, 2.5), uniform(2.5, 5)]:
#                 for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
#                     aug_data = data[i].tolist()
#                     aug_data[protected_index[0]] = round(a_0) / (96 - 18)  # 归一化
#                     aug_data[protected_index[1]] = round(a_1) / (5)
#                     aug_data[protected_index[2]] = round(a_2) / (1)
#                     # data_list.append(aug_data)
#                     # 生成测试数据集相似样本时，去除真实标记
#                     data_list.append(aug_data[:-1])
#         aug.append(data_list)
#     return numpy.squeeze(numpy.array(aug))
#
#
# def data_augmentation_bank(data, protected_index):
#     """
#     对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
#     :return:
#     """
#     aug = []
#     for i in range(data.shape[0]):
#         data_list = []
#         for a_1 in [uniform(18, 35), uniform(35, 55), uniform(55, 75), uniform(75, 95)]:
#             aug_data = data[i].tolist()
#             aug_data[protected_index[0]] = round(a_1) / (95 - 18)  # 归一化
#             # data_list.append(aug_data)
#             # 生成测试数据集相似样本时，去除真实标记
#             data_list.append(aug_data[:-1])
#         aug.append(data_list)
#     return numpy.squeeze(numpy.array(aug))
#
#
# def generate_similar_dataset(D_tag, dataset, protected_attr):
#     """
#     生成样本的相似样本
#     :return:
#     """
#     # 生成相似样本
#     if D_tag == "adult":
#         similar_dataset = data_augmentation_adult(dataset, protected_attr)
#     elif D_tag == "compas":
#         similar_dataset = data_augmentation_compas(dataset, protected_attr)
#     elif D_tag == "credit":
#         similar_dataset = data_augmentation_credit(dataset, protected_attr)
#     else:
#         similar_dataset = data_augmentation_bank(dataset, protected_attr)
#     return similar_dataset


# def calculate_mse(data1, data2):
#     """
#     计算均方误差
#     :return:
#     """
#     # MSE_distance = []
#     # for i in range(data1.shape[0]):
#     #     num = 0
#     #     for j in range(data1.shape[1]):
#     #         num += ((data1[i, j] - data2[i, j]) * (data1[i, j] - data2[i, j]))
#     #     distance = num / data1.shape[1]
#     #     MSE_distance.append(distance)
#     # return numpy.array(MSE_distance).astype(float)
#     return mean_squared_error(data1, data2)


# def calculate_crossentropy(data1, data2):
#     """
#     计算categorical_crossentropy误差
#     :return:
#     """
#
#     return categorical_crossentropy(data1, data2)


# def compare_data(data1, data2):
#     """
#     检查data1中小于data2的结果
#     :return:
#     """
#     result = []
#     for i in range(data1.shape[0]):
#         if data1[i] <= data2[i]:
#             result.append(True)
#         else:
#             result.append(False)
#     return numpy.array(result).astype(bool)
#
#
# def check_item_IF(pre1, pre2, x1, x2, dist, K):
#     """
#     检查样本与相似样本的预测结果是否 individual fair
#     D(f(x1),f(x2))<=Kd(x1,x2)
#     :return:
#     """
#     for i in range(len(pre2)):
#         D_distance = calculate_mse(pre1.reshape(1, 1), pre2[i].reshape(1, 1))
#         Kd_distance = K * calculate_mse(x1, x2[i].reshape(1, -1))
#         IF_result = check_dist(D_distance - Kd_distance, dist)
#         if IF_result[0]:
#             pass
#         else:
#             return False
#     return True
#
#
# def check_item_AF(label, pre1, pre2, x1, x2, dist, K):
#     """
#     检查数据集的 accurate fairness
#     """
#     MSE = calculate_mse(pre1.reshape(1, 1), label.reshape(1, 1))
#     AF_result = check_dist(MSE, dist)
#     for i in range(len(pre2)):
#         # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
#         D_distance = calculate_mse(label.reshape(1, 1), pre2[i].reshape(1, 1))
#         Kd_distance = K * calculate_mse(x1, x2[i].reshape(1, -1))
#         AF_result = numpy.logical_and(AF_result, check_dist(D_distance - Kd_distance, dist))
#         if AF_result[0]:
#             pass
#         else:
#             return False
#     return True
#
#
# def check_item_confusion(label, pre1, pre2, x1, x2, dist, K):
#     """
#     计算 fairness confusion
#     """
#     MSE = calculate_mse(pre1.reshape(1, 1), label.reshape(1, 1))
#     True_cond = check_dist(MSE, dist)
#     AF_cond = numpy.ones(True_cond.shape)
#     for j in range(len(pre2)):
#         # D(y,f(similar_x))<=Kd(x,similar_x)+dist
#         D_distance = calculate_mse(label.reshape(1, 1), pre2[j].reshape(1, 1))
#         Kd_distance = K * calculate_mse(x1, x2[j].reshape(1, -1))
#         AF_cond = numpy.logical_and(AF_cond, check_dist(D_distance - Kd_distance, dist))
#
#     IF_cond = numpy.ones(True_cond.shape)
#     for i in range(len(pre2)):
#         # D(f(x),f(similar_x))<=Kd(x,similar_x)+dist
#         D_distance = calculate_mse(pre1.reshape(1, 1), pre2[i].reshape(1, 1))
#         Kd_distance = K * calculate_mse(x1, x2[i].reshape(1, -1))
#         IF_cond = numpy.logical_and(IF_cond, check_dist(D_distance - Kd_distance, dist))
#
#     # 计算 TF TB FF FB
#     TF_cond = numpy.logical_and(True_cond, IF_cond)
#     TB_cond = numpy.logical_and(True_cond, ~IF_cond)
#     FF_cond = numpy.logical_and(~True_cond, IF_cond)
#     FB_cond = numpy.logical_and(~True_cond, ~IF_cond)
#
#     return TF_cond[0], [TB_cond[0], FF_cond[0], FB_cond[0]]
#
#
# # def check_dataset_confusion(label, pre, aug_pre, x, aug_x, dist, K):
# #     """
# #     计算 fairness confusion
# #     """
# #     MSE = calculate_mse(pre, label)
# #     True_cond = check_dist(MSE, dist)
# #     AF_cond = numpy.ones(True_cond.shape)
# #     for j in range(len(aug_pre)):
# #         # D(y,f(similar_x))<=Kd(x,similar_x)+dist
# #         D_distance = calculate_mse(label, aug_pre[j])
# #         Kd_distance = K * calculate_mse(numpy.concatenate((x[0], x[1]), axis=1),
# #                                         numpy.concatenate((aug_x[j][0], aug_x[j][1]), axis=1))
# #         AF_cond = numpy.logical_and(AF_cond, check_dist(D_distance - Kd_distance, dist))
# #
# #     IF_cond = numpy.ones(True_cond.shape)
# #     for i in range(len(aug_pre)):
# #         # D(f(x),f(similar_x))<=Kd(x,similar_x)+dist
# #         D_distance = calculate_mse(pre, aug_pre[i])
# #         Kd_distance = K * calculate_mse(numpy.concatenate((x[0], x[1]), axis=1),
# #                                         numpy.concatenate((aug_x[i][0], aug_x[i][1]), axis=1))
# #         IF_cond = numpy.logical_and(IF_cond, check_dist(D_distance - Kd_distance, dist))
# #
# #     # 计算 TF TB FF FB
# #     TF_cond = numpy.logical_and(True_cond, IF_cond)
# #     TB_cond = numpy.logical_and(True_cond, ~IF_cond)
# #     FF_cond = numpy.logical_and(~True_cond, IF_cond)
# #     FB_cond = numpy.logical_and(~True_cond, ~IF_cond)
# #
# #     TF = numpy.sum(TF_cond)
# #     TB = numpy.sum(TB_cond)
# #     FF = numpy.sum(FF_cond)
# #     FB = numpy.sum(FB_cond)
# #
# #     TFR = numpy.sum(TF_cond) / TF_cond.shape[0]
# #     TBR = numpy.sum(TB_cond) / TF_cond.shape[0]
# #     FFR = numpy.sum(FF_cond) / TF_cond.shape[0]
# #     FBR = numpy.sum(FB_cond) / TF_cond.shape[0]
# #
# #     if (TFR + FFR) == 0:
# #         F_recall = 0
# #     else:
# #         F_recall = TFR / (TFR + FFR)
# #
# #     if (TFR + TBR) == 0:
# #         F_precision = 0
# #     else:
# #         F_precision = TFR / (TFR + TBR)
# #
# #     if (F_recall + F_precision) == 0:
# #         F_F1 = 0
# #     else:
# #         F_F1 = (2 * F_recall * F_precision) / (F_recall + F_precision)
# #
# #     return [TFR, TBR, FFR, FBR, F_recall, F_precision, F_F1, TF, TB, FF, FB, TF + TB + FF + FB]
#
#
# def get_AF_condition(label, pre1, pre2, x1, x2, dist, K):
#     """
#     计算 fairness confusion
#     """
#     MSE = calculate_mse(pre1, label)
#     True_cond = check_dist(MSE, dist)
#     AF_cond = numpy.ones(True_cond.shape)
#     for j in range(len(pre2)):
#         # D(y,f(similar_x))<=Kd(x,similar_x)+dist
#         D_distance = calculate_mse(label, pre2[j])
#         Kd_distance = K * calculate_mse(x1, x2[j])
#         AF_cond = numpy.logical_and(AF_cond, check_dist(D_distance - Kd_distance, dist))
#
#     IF_cond = numpy.ones(True_cond.shape)
#     for i in range(len(pre2)):
#         # D(f(x),f(similar_x))<=Kd(x,similar_x)+dist
#         D_distance = calculate_mse(pre1, pre2[i])
#         Kd_distance = K * calculate_mse(x1, x2[i])
#         IF_cond = numpy.logical_and(IF_cond, check_dist(D_distance - Kd_distance, dist))
#
#     # 计算 TF TB FF FB
#     TF_cond = numpy.logical_and(True_cond, IF_cond)
#     TB_cond = numpy.logical_and(True_cond, ~IF_cond)
#     FF_cond = numpy.logical_and(~True_cond, IF_cond)
#     FB_cond = numpy.logical_and(~True_cond, ~IF_cond)
#
#     return [TF_cond, TB_cond, FF_cond, FB_cond]
#
#
# # 对模型进行评估
# def regression_evaluation(file1, file2, file3, dist=0.001, K=0):
#     """
#     根据模型获取结果，回归模型
#     :return:
#     """
#     model = load_model(file1)
#     data1 = numpy.load(file2)
#     x1, y1 = numpy.split(data1, [-1, ], axis=1)
#     pre1 = model.predict(x1)
#     data2 = numpy.squeeze(numpy.load(file3))
#     x2 = []
#     pre2 = []
#     for j in range(data2.shape[1]):
#         x2.append(data2[:, j, :])
#         pre2.append(model.predict(data2[:, j, :]))
#     ACC_result = check_ACC(pre1, y1, dist)
#     IF_result = check_IF(pre1, pre2, x1, x2, dist, K)
#     AF_result = check_AF(y1, pre1, pre2, x1, x2, dist, K)
#     Fair_confusion = check_dataset_confusion(y1, pre1, pre2, x1, x2, dist, K)
#     return ACC_result + IF_result + AF_result + Fair_confusion
#
#
# def check_regression(model_files, file1, file2, eval_file, dist):
#     """
#     评估分类模型
#     :return:
#     """
#     header_name = ["avg", "std", "acc R", "false R", "acc N", "false N", "SUM",
#                    "IFR", "IBR", "IFN", "IBN", "SUM",
#                    "A&F R", "F|B R", "A&F", "F|B", "SUM",
#                    "TFR", "TBR", "FFR", "FBR", 'F_recall', 'F_precision', 'F_F1', 'TF', 'TB', 'FF', "FB", "SUM"]
#     evaluation_data = []
#     for f in model_files:
#         result_evaluation = regression_evaluation(f, file1, file2, dist)
#         evaluation_data.append(result_evaluation)
#         print(result_evaluation)
#
#     workbook_name = xlsxwriter.Workbook(eval_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(header_name, worksheet)
#     write_worksheet_2d_data(evaluation_data, worksheet)
#     workbook_name.close()
#
#
# def robustness_result_evaluation(file1, test_data, D_tag, P_attr, dist=0, K=0):
#     """
#     根据模型获取结果，分类模型
#     :return:
#     """
#     model = load_model(file1)
#     generated_similar = generate_similar_dataset(D_tag, test_data, P_attr)
#     x1, y1 = numpy.split(test_data, [-1, ], axis=1)
#     pre = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)
#     x2 = []
#     pre2 = []
#     for j in range(generated_similar.shape[1]):
#         x2.append(generated_similar[:, j, :])
#         pre2.append(numpy.argmax(model.predict(generated_similar[:, j, :]), axis=1).reshape(-1, 1))
#     TFR, TBR, FFR, FBR, _, _, _, _, _, _, _, SUM = check_dataset_confusion(y1, pre, pre2, x1, x2, dist, K)
#
#     return [TFR, TBR, FFR, FBR, SUM]
#
#
# def fairness_result_evaluation(file1, test_data, D_tag, P_attr, dist=0, K=0):
#     """
#     根据模型获取结果，分类模型
#     :return:
#     """
#     model = load_model(file1)
#     generated_data = test_data[0]
#     for i in range(test_data.shape[0] - 1):
#         generated_data = numpy.concatenate((generated_data, test_data[i + 1]), axis=0)
#     generated_similar = generate_similar_dataset(D_tag, generated_data, P_attr)
#     x1, y1 = numpy.split(generated_data, [-1, ], axis=1)
#     pre = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)
#     x2 = []
#     pre2 = []
#     for j in range(generated_similar.shape[1]):
#         x2.append(generated_similar[:, j, :])
#         pre2.append(numpy.argmax(model.predict(generated_similar[:, j, :]), axis=1).reshape(-1, 1))
#     TFR, TBR, FFR, FBR, _, _, _, _, _, _, _, SUM = check_dataset_confusion(y1, pre, pre2, x1, x2, dist, K)
#     return [TFR, TBR, FFR, FBR, SUM]
#
#
# def robust_accurate_fairness_result_evaluation(model_file, test_data, D_tag, P_attr, dist=0, K=0):
#     """
#     根据模型获取结果，分类模型
#     :return:
#     """
#     model = load_model(model_file)
#     generated_data = test_data[0]
#     for i in range(test_data.shape[0] - 1):
#         generated_data = numpy.concatenate((generated_data, test_data[i + 1]), axis=0)
#     generated_similar = generate_similar_dataset(D_tag, generated_data, P_attr)
#     x1, y1 = numpy.split(generated_data, [-1, ], axis=1)
#     pre = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)
#     x2 = []
#     pre2 = []
#     for j in range(generated_similar.shape[1]):
#         x2.append(generated_similar[:, j, :])
#         pre2.append(numpy.argmax(model.predict(generated_similar[:, j, :]), axis=1).reshape(-1, 1))
#
#     TFR, TBR, FFR, FBR, _, _, _, _, _, _, _, SUM = check_dataset_confusion(y1, pre, pre2, x1, x2, dist, K)
#     return [TFR, TBR, FFR, FBR, SUM]
#
#
# def calculate_cos(cluster_data, generate_data, cos_threshold):
#     """
#     计算生成样本与聚类中心的余弦相似度，并选择大于cos_threshold的位置
#     :return:
#     """
#     condition = []
#     cos_result = []
#     for i in range(generate_data.shape[0]):
#         cos_max = 0
#         for j in range(cluster_data.shape[0]):
#             if cos(cluster_data[j], generate_data[i]) > cos_max:
#                 cos_max = cos(cluster_data[j], generate_data[i])
#         cos_result.append(cos_max)
#         if cos_max > cos_threshold:
#             condition.append(True)
#         else:
#             condition.append(False)
#
#     return numpy.array(cos_result), condition
#
#
# def cos(a, b):  #
#     dot = 0
#     mod_a = 0
#     mod_b = 0
#     for i in range(len(a)):
#         dot += a[i] * b[i]
#         mod_a += a[i] * a[i]
#         mod_b += b[i] * b[i]
#
#     return dot / (math.sqrt(mod_a) * math.sqrt(mod_b))

# def check_classification1(model_files, file1, file2, S_file, eval_file):
#     """
#     评估分类模型
#     :return:
#     """
#     header_name = ["Acc R", "acc N", "False R", "False N",
#                    "IF R", "IF N", "IB R", "IB N",
#                    "A&F R", "A&F", "F|B R", "F|B",
#                    "TFR", "TBR", "FFR", "FBR", 'F_recall', 'F_precision', 'F_F1', 'TF', 'TB', 'FF', "FB", "SUM",
#                    "cos"]
#     evaluation_data = []
#     for f in model_files:
#         result_evaluation = adversarial_evaluation(f, file1, file2, S_file)
#         evaluation_data.append(result_evaluation)
#         print(result_evaluation)
#
#     workbook_name = xlsxwriter.Workbook(eval_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(header_name, worksheet)
#     write_worksheet_2d_data(evaluation_data, worksheet)
#     workbook_name.close()


# def check_under_fit_model(model_file, test_x, data2, test_y, threshold1, threshold2, dist=0):
#     """
#     检查模型是否欠拟合
#     :return:
#     """
#     model = load_model(model_file)
#     test_pre = numpy.argmax(model.predict(test_x), axis=1).reshape(-1, 1)
#     ACC_result = check_performance(test_pre, test_y, dist)
#
#     x2 = []
#     pre2 = []
#     for j in range(data2.shape[1]):
#         x2.append(data2[:, j, :])
#         pre2.append(numpy.argmax(model.predict(data2[:, j, :]), axis=1).reshape(-1, 1))
#     IF_result = check_IF(test_pre, pre2, test_x, x2, dist, K=0)
#
#     if ACC_result[0] > threshold1 and IF_result[0] > threshold2:
#         print("acc:{:.4f}".format(ACC_result[0]) + "-----IF:{:.4f}".format(IF_result[0]))
#         return False
#     else:
#         print("acc:{:.4f}".format(ACC_result[0]) + "-----IF:{:.4f}".format(IF_result[0]))
#         return True

# def get_IF_condition(pre_x, pre_similar_x, x, similar_x, K):
#     """
#     获取fta condition
#     :return:
#     """
#     FTA_cond = []
#     for i in range(len(pre_similar_x)):
#         # D(f(x),f(similar_x))<=Kd(x,similar_x)
#         D_distance = calculate_MSE(pre_x, pre_similar_x[i])
#         Kd_distance = K * calculate_MSE(x, similar_x[i])
#         FTA_cond.append(check_epsilon(D_distance - Kd_distance, 0))
#     fair_cond = FTA_cond[0]
#     for j in range(len(FTA_cond)):
#         fair_cond = numpy.logical_and(fair_cond, FTA_cond[j])
#     return fair_cond, ~fair_cond

# def get_AF_condition(label, pre_x, pre_similar_x, x, similar_x, epsilon, K):
#     """
#     获取准确公平性条件
#     :return:
#     """
#     # 样本performance
#     anchor_distance = calculate_MSE(pre_x, label)
#     anchor_cond = check_epsilon(anchor_distance, epsilon)
#     # 计算相似样本的condition
#     similar_conditions = []
#     for i in range(len(pre_similar_x)):
#         # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
#         D_distance = calculate_MSE(label, pre_similar_x[i])
#         Kd_distance = K * calculate_MSE(x, similar_x[i])
#         similar_conditions.append(check_epsilon(D_distance - Kd_distance, epsilon))
#     similar_cond = similar_conditions[0]
#     for j in range(len(similar_conditions)):
#         similar_cond = numpy.logical_and(similar_cond, similar_conditions[j])
#
#     cond1 = numpy.logical_and(anchor_cond, similar_cond)
#     cond2 = numpy.logical_and(anchor_cond, ~similar_cond)
#     cond3 = numpy.logical_and(~anchor_cond, similar_cond)
#     cond4 = numpy.logical_and(~anchor_cond, ~similar_cond)
#     return cond1, cond2, cond3, cond4


# # privilege score
# def get_privileged_result(label, pre_x, pre_similar_x, x, similar_x, epsilon, K):
#     """
#     get the privileged protected attributes
#     :return:
#     """
#     # 歧视状态
#     performance = calculate_MSE(pre_x, label)
#     performance_cond = check_epsilon(performance, epsilon)
#     performance_cond_similar = []
#     for i in range(len(pre_similar_x)):
#         # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
#         # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
#         D_distance = calculate_MSE(label, pre_similar_x[i])
#         Kd_distance = K * calculate_MSE(x, similar_x[i])
#         performance_cond_similar.append(check_epsilon(D_distance - Kd_distance, epsilon))
#     similar_fair_cond = performance_cond_similar[0]
#     for j in range(len(performance_cond_similar)):
#         similar_fair_cond = numpy.logical_and(similar_fair_cond, performance_cond_similar[j])
#
#     for j in range(len(performance_cond_similar)):
#         TB_cond = numpy.logical_and(performance_cond, ~similar_fair_cond)
#         FB_cond = numpy.logical_and(~performance_cond, ~similar_fair_cond)
#
#     bias_cond = numpy.logical_or(TB_cond, FB_cond)
#     # 歧视输出中的特权位置
#     return get_privilege_items(bias_cond, pre_x, pre_similar_x, x, similar_x)
#
#
# def get_privilege_items(bias_cond, pre_x, pre_similar_x, x, similar_x):
#     "根据歧视状态，输出预测结果输出中特权样本"
#     privilege_items = []
#     predication = numpy.array([pre_x] + pre_similar_x).astype(float)
#     features = numpy.array([x] + similar_x).astype(float)
#     predication_mean = numpy.mean(predication, axis=0)
#     # 大于平均值，认为受到优待
#     privilege_place = numpy.where(predication > predication_mean, True, False)
#     for bias_index in range(bias_cond.shape[0]):
#         if bias_cond[bias_index]:
#             for similar_i in range(privilege_place.shape[0]):
#                 if privilege_place[similar_i, bias_index, 0]:
#                     privilege_items.append(features[similar_i, bias_index, :])
#     return numpy.array(privilege_items)
#
#
# def numpy_range(input_n, up_bound, lower_bound):
#     condition = []
#     for i in range(input_n.shape[0]):
#         if lower_bound < input_n[i] <= up_bound:
#             condition.append(True)
#         else:
#             condition.append(False)
#     return numpy.array(condition).astype(bool)
#
#
# def numpy_check(input_n, target, delta=0.001):
#     condition = []
#     for i in range(input_n.shape[0]):
#         if target - delta <= input_n[i] <= target + delta:
#             condition.append(True)
#         else:
#             condition.append(False)
#     return numpy.array(condition).astype(bool)
#
#
# def analysis_ctrip_privilege(privilege_items):
#     """
#     分析哪些保护属性群体更容易被歧视，歧视发生在哪些非保护属性上
#     :return:
#     """
#     subgroup_names = []
#     for n1 in ["Long", 'Short']:  # Confirmation duration
#         for n2 in ["More", 'Less']:  # advance days
#             for n3 in ["Superior", 'Affordable']:  # hotel  level ordered
#                 for n4 in ["Category 1", 'Category 2']:  # hotel category1
#                     for n5 in ["High", 'Low']:  # hotel level recommended
#                         for n6 in ["Long", 'Short']:  # Length of stay
#                             subgroup_names.append([n1, n2, n3, n4, n5, n6])
#
#     condition = []
#     for i in [0, 1, 2, 3, 4, 5]:
#         # 将各保护属性按取值分为两类
#         up_c = numpy_range(privilege_items[:, i], 2, 0.5)
#         low_c = numpy_range(privilege_items[:, i], 0.5, -1)
#         condition.append([up_c, low_c])
#
#     privilege_scores = []
#     for c1 in condition[0]:
#         for c2 in condition[1]:
#             for c3 in condition[2]:
#                 for c4 in condition[3]:
#                     for c5 in condition[4]:
#                         for c6 in condition[5]:
#                             s1 = numpy.logical_and(c1, c2)
#                             s2 = numpy.logical_and(c3, c4)
#                             s3 = numpy.logical_and(c5, c6)
#                             score = numpy.sum(numpy.logical_and(s1, numpy.logical_and(s2, s3)))
#                             privilege_scores.append(score)
#     user_privilege_score = []
#     for h in range(len(subgroup_names)):
#         user_privilege_score.append(subgroup_names[h] + [privilege_scores[h]])
#
#     # user_P_R = numpy.array(user_privilege_score)
#     # numpy.sort(user_P_R)
#     #
#     # star_5_cond = numpy_check(privilege_items[:, 17], 0)
#     # star_7_cond = numpy_check(privilege_items[:, 17], 2 / 6)
#     # star_9_cond = numpy_check(privilege_items[:, 17], 4 / 6)
#     # star_11_cond = numpy_check(privilege_items[:, 17], 1)
#     # star_cond = [star_5_cond, star_7_cond, star_9_cond, star_11_cond]
#     #
#     # star_P_S = []
#     # for s_c in star_cond:
#     #     star_P_S.append(numpy.sum(s_c))
#
#     return user_privilege_score
