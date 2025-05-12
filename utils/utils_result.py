import pickle
import numpy


def restore_adult_nlp(index_data, value_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    fea_dim = numpy.load("../dataset/adult/data/fea_dim.npy").tolist()
    with open("../dataset/adult/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    NLP_data = []
    for i in range(13):
        if i in [0, 3, 9, 10, 11]:
            NLP_data.append(str(value_data[i]))
        elif i in [1, 2, 4, 5, 6, 7, 8, 12]:
            feature_index, raw_code = return_feature_and_index(index_data[i], fea_dim)
            for k in vocab_dic[feature_index].keys():
                if vocab_dic[feature_index][k][0] == raw_code:
                    NLP_data.append(k)
                    break
    if label_data == 0:
        NLP_data.append('<=50K')
    else:
        NLP_data.append('>50K')

    if pre_data == 0:
        NLP_data.append('<=50K')
    else:
        NLP_data.append('>50K')
    return NLP_data


def restore_NLP_adult_nlp(input_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    with open("../dataset/NLP/adult/data/vocab_dic.pkl", 'rb') as f:
        word_index = pickle.load(f)
    reverse_word_index = {index: word for word, index in word_index.items()}
    NLP_data = ' '.join([reverse_word_index.get(i, '?') for i in input_data if i != 0])

    if label_data == 0:
        NLP_data = NLP_data + ',<=50K'
    else:
        NLP_data = NLP_data + ',>50K'

    if pre_data == 0:
        NLP_data = NLP_data + ',<=50K.'
    else:
        NLP_data = NLP_data + ',>50K.'
    return [NLP_data]


def restore_bank_nlp(index_data, value_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    fea_dim = numpy.load("../dataset/bank/data/fea_dim.npy").tolist()
    with open("../dataset/bank/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    NLP_data = []
    for i in range(16):
        if i in [5, 9, 11, 12, 13, 14]:
            NLP_data.append(str(value_data[i]))
        elif i in [0, 1, 2, 3, 4, 6, 7, 8, 10, 15]:
            feature_index, raw_code = return_feature_and_index(index_data[i], fea_dim)
            for k in vocab_dic[feature_index].keys():
                if vocab_dic[feature_index][k][0] == raw_code:
                    NLP_data.append(k)
                    break
    if label_data == 0:
        NLP_data.append('no')
    else:
        NLP_data.append('yes')

    if pre_data == 0:
        NLP_data.append('no')
    else:
        NLP_data.append('yes')
    return NLP_data


def restore_NLP_bank_nlp(input_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    with open("../dataset/NLP/bank/data/vocab_dic.pkl", 'rb') as f:
        word_index = pickle.load(f)
    reverse_word_index = {index: word for word, index in word_index.items()}
    NLP_data = ' '.join([reverse_word_index.get(i, '?') for i in input_data if i != 0])

    if label_data == 0:
        NLP_data = NLP_data + ',negative'
    else:
        NLP_data = NLP_data + ',positive'

    if pre_data == 0:
        NLP_data = NLP_data + ',negative.'
    else:
        NLP_data = NLP_data + ',positive.'
    return [NLP_data]


def restore_compas_nlp(index_data, value_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    fea_dim = numpy.load("../dataset/compas/data/fea_dim.npy").tolist()
    with open("../dataset/compas/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    NLP_data = []
    for i in range(18):
        if i in [3, 4, 7, 10, 13, 15, 17]:
            NLP_data.append(str(value_data[i]))
        elif i in [0, 1, 2, 5, 6, 8, 9, 11, 12, 14, 16]:
            feature_index, raw_code = return_feature_and_index(index_data[i], fea_dim)
            for k in vocab_dic[feature_index].keys():
                if vocab_dic[feature_index][k][0] == raw_code:
                    NLP_data.append(k)
                    break
    if label_data == 0:
        NLP_data.append('Negative')
    else:
        NLP_data.append('Positive')

    if pre_data == 0:
        NLP_data.append('Negative')
    else:
        NLP_data.append('Positive')
    return NLP_data


def restore_NLP_compas_nlp(input_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    with open("../dataset/NLP/compas/data/vocab_dic.pkl", 'rb') as f:
        word_index = pickle.load(f)
    reverse_word_index = {index: word for word, index in word_index.items()}
    NLP_data = ' '.join([reverse_word_index.get(i, '?') for i in input_data if i != 0])

    if label_data == 0:
        NLP_data = NLP_data + ',negative'
    else:
        NLP_data = NLP_data + ',positive'

    if pre_data == 0:
        NLP_data = NLP_data + ',negative.'
    else:
        NLP_data = NLP_data + ',positive.'
    return [NLP_data]


def restore_credit_nlp(index_data, value_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    fea_dim = numpy.load("../dataset/credit/data/fea_dim.npy").tolist()
    with open("../dataset/credit/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    NLP_data = []
    for i in range(21):
        if i in [1, 4, 7, 11, 13, 16, 18]:
            NLP_data.append(str(value_data[i]))
        elif i in [0, 2, 3, 5, 6, 8, 9, 10, 12, 14, 15, 17, 19, 20]:
            # if i == 0:
            #     raw_code = index_data[i]
            # else:
            #     raw_code = index_data[i] - fea_dim[i - 1]
            feature_index, raw_code = return_feature_and_index(index_data[i], fea_dim)
            for k in vocab_dic[feature_index].keys():
                if vocab_dic[feature_index][k][0] == raw_code:
                    NLP_data.append(k)
                    break
    if label_data == 0:
        NLP_data.append('Negative')
    else:
        NLP_data.append('Positive')

    if pre_data == 0:
        NLP_data.append('Negative')
    else:
        NLP_data.append('Positive')
    return NLP_data


def restore_employment_nlp(index_data, value_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    fea_dim = numpy.load("../dataset/ACS/employment/data/fea_dim.npy").tolist()
    with open("../dataset/ACS/employment/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    NLP_data = []
    for i in range(16):
        if i in [0]:
            NLP_data.append(str(value_data[i]))
        elif i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            feature_index, raw_code = return_feature_and_index(index_data[i], fea_dim)
            for k in vocab_dic[feature_index].keys():
                if vocab_dic[feature_index][k][0] == raw_code:
                    NLP_data.append(k)
                    break
    if label_data == 0:
        NLP_data.append('negative')
    else:
        NLP_data.append('positive')

    if pre_data == 0:
        NLP_data.append('negative')
    else:
        NLP_data.append('positive')
    return NLP_data


def restore_NLP_employment_nlp(input_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    with open("../dataset/ACS/NLP/employment/data/vocab_dic.pkl", 'rb') as f:
        word_index = pickle.load(f)
    reverse_word_index = {index: word for word, index in word_index.items()}
    NLP_data = ' '.join([reverse_word_index.get(i, '?') for i in input_data if i != 0])

    if label_data == 0:
        NLP_data = NLP_data + ',negative'
    else:
        NLP_data = NLP_data + ',positive'

    if pre_data == 0:
        NLP_data = NLP_data + ',negative.'
    else:
        NLP_data = NLP_data + ',positive.'
    return [NLP_data]


def restore_income_nlp(index_data, value_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    fea_dim = numpy.load("../dataset/ACS/income/data/fea_dim.npy").tolist()
    with open("../dataset/ACS/income/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    NLP_data = []
    for i in range(10):
        if i in [0, 7]:
            NLP_data.append(str(value_data[i]))
        elif i in [1, 2, 3, 4, 5, 6, 8, 9]:
            feature_index, raw_code = return_feature_and_index(index_data[i], fea_dim)
            for k in vocab_dic[feature_index].keys():
                if vocab_dic[feature_index][k][0] == raw_code:
                    NLP_data.append(k)
                    break
    if label_data == 0:
        NLP_data.append('negative')
    else:
        NLP_data.append('positive')

    if pre_data == 0:
        NLP_data.append('negative')
    else:
        NLP_data.append('positive')
    return NLP_data


def restore_NLP_income_nlp(input_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    with open("../dataset/ACS/NLP/income/data/vocab_dic.pkl", 'rb') as f:
        word_index = pickle.load(f)
    reverse_word_index = {index: word for word, index in word_index.items()}
    NLP_data = ' '.join([reverse_word_index.get(i, '?') for i in input_data if i != 0])

    if label_data == 0:
        NLP_data = NLP_data + ',negative'
    else:
        NLP_data = NLP_data + ',positive'

    if pre_data == 0:
        NLP_data = NLP_data + ',negative.'
    else:
        NLP_data = NLP_data + ',positive.'
    return [NLP_data]


def restore_coverage_nlp(index_data, value_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    fea_dim = numpy.load("../dataset/ACS/coverage/data/fea_dim.npy").tolist()
    with open("../dataset/ACS/coverage/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    NLP_data = []
    for i in range(19):
        if i in [0, 14]:
            NLP_data.append(str(value_data[i]))
        elif i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]:
            feature_index, raw_code = return_feature_and_index(index_data[i], fea_dim)
            for k in vocab_dic[feature_index].keys():
                if vocab_dic[feature_index][k][0] == raw_code:
                    NLP_data.append(k)
                    break
    if label_data == 0:
        NLP_data.append('negative')
    else:
        NLP_data.append('positive')

    if pre_data == 0:
        NLP_data.append('negative')
    else:
        NLP_data.append('positive')
    return NLP_data


def restore_NLP_coverage_nlp(input_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    with open("../dataset/ACS/NLP/coverage/data/vocab_dic.pkl", 'rb') as f:
        word_index = pickle.load(f)
    reverse_word_index = {index: word for word, index in word_index.items()}
    NLP_data = ' '.join([reverse_word_index.get(i, '?') for i in input_data if i != 0])

    if label_data == 0:
        NLP_data = NLP_data + ',negative'
    else:
        NLP_data = NLP_data + ',positive'

    if pre_data == 0:
        NLP_data = NLP_data + ',negative.'
    else:
        NLP_data = NLP_data + ',positive.'
    return [NLP_data]


def restore_travel_nlp(index_data, value_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    fea_dim = numpy.load("../dataset/ACS/travel/data/fea_dim.npy").tolist()
    with open("../dataset/ACS/travel/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    NLP_data = []
    for i in range(16):
        if i in [0, 15]:
            NLP_data.append(str(value_data[i]))
        elif i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            feature_index, raw_code = return_feature_and_index(index_data[i], fea_dim)
            for k in vocab_dic[feature_index].keys():
                if vocab_dic[feature_index][k][0] == raw_code:
                    NLP_data.append(k)
                    break
    if label_data == 0:
        NLP_data.append('negative')
    else:
        NLP_data.append('positive')

    if pre_data == 0:
        NLP_data.append('negative')
    else:
        NLP_data.append('positive')
    return NLP_data


def restore_NLP_travel_nlp(input_data, label_data, pre_data):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    with open("../dataset/ACS/NLP/travel/data/vocab_dic.pkl", 'rb') as f:
        word_index = pickle.load(f)
    reverse_word_index = {index: word for word, index in word_index.items()}
    NLP_data = ' '.join([reverse_word_index.get(i, '?') for i in input_data if i != 0])

    if label_data == 0:
        NLP_data = NLP_data + ',negative'
    else:
        NLP_data = NLP_data + ',positive'

    if pre_data == 0:
        NLP_data = NLP_data + ',negative.'
    else:
        NLP_data = NLP_data + ',positive.'
    return [NLP_data]


def return_feature_and_index(index, feature_dim):
    """
    返回编码所属于的属性坐标，其及取值
    :return:
    """
    for i in range(len(feature_dim)):
        if index <= feature_dim[i]:
            if i == 0:
                return i, index
            else:
                return i, index - feature_dim[i - 1]

# import random
# from random import uniform

# import pandas
# import xlsxwriter
# from tensorflow.python.keras.models import load_model
#
# from utils.utils_evaluate import robustness_result_evaluation, fairness_result_evaluation, \
#     robust_accurate_fairness_result_evaluation, calculate_cos
# from utils.utils_generate import data_augmentation_adult, data_augmentation_compas, data_augmentation_credit, \
#     data_augmentation_bank
# from utils.utils_input_output import write_worksheet_header, write_worksheet_2d_data


# def robustness_result(M_file, Adv_data, seed_file):
#     """
#     检查生成的对抗样本的准确性，个体公平性，准确公平性,与测试种子的余弦相似度
#     :return:
#     """
#     adversarial_result = robustness_result_evaluation(M_file, Adv_data, seed_file)
#     return adversarial_result
#
#
# def fairness_result(M_file, Adv_data, D_tag, P_attr, seed_file):
#     """
#     检查生成的对抗样本的准确性，个体公平性，准确公平性,与测试种子的余弦相似度
#     :return:
#     """
#     adversarial_result = fairness_result_evaluation(M_file, Adv_data, D_tag, P_attr, seed_file)
#     return adversarial_result
#
#
# def fairness_result_DICE(M_file, Adv_data, Similar_data, seed_file):
#     """
#     检查生成的对抗样本的准确性，个体公平性，准确公平性,与测试种子的余弦相似度
#     :return:
#     """
#     model = load_model(M_file)
#
#     x1, y1 = numpy.split(Adv_data, [-1, ], axis=1)
#     pre = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)
#     x2 = []
#     pre2 = []
#     for j in range(Similar_data.shape[1]):
#         x2.append(Similar_data[:, j, :])
#         pre2.append(numpy.argmax(model.predict(Similar_data[:, j, :]), axis=1).reshape(-1, 1))
#     IF_result = check_IF(pre, pre2, x1, x2, dist=0, K=0)
#
#     cluster_data = numpy.load(seed_file)
#     cos_result, select_cond = calculate_cos(cluster_data, Adv_data, 1.0)
#     return [IF_result[-1], Adv_data.shape[0], numpy.mean(cos_result)]


# def robust_fair_result(M_file, Adv_data, Similar_data, seed_file):
#     """
#     检查生成的对抗样本的准确性，个体公平性，准确公平性,与测试种子的余弦相似度
#     :return:
#     """
#     adversarial_result = robust_accurate_fairness_result_evaluation(M_file, Adv_data, Similar_data, seed_file)
#     return adversarial_result


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
#
#
# def get_model_attack_evaluation(D_tag, Result_file, P_attr):
#     """
#     获取模型的鲁棒性、公平性、准确公平性的测试结果
#     :return:
#     """
#     Robust_APGD = []
#     Robust_ACG = []
#     Fair_ADF = []
#     Fair_DICE = []
#     Robust_AccFair = []
#
#     M_file = "../dataset/{}/model/BL.h5".format(D_tag)
#
#     ACG_D_file = "../dataset/{}/test/Test_ACG_BL_data.npy".format(D_tag)
#     ACG_P_file = "../dataset/{}/test/Test_ACG_BL_position.npy".format(D_tag)
#     ACG_D = numpy.load(ACG_D_file)
#     ACG_P = numpy.load(ACG_P_file)
#
#     APGD_D_file = "../dataset/{}/test/Test_APGD_BL_data.npy".format(D_tag)
#     APGD_P_file = "../dataset/{}/test/Test_APGD_BL_position.npy".format(D_tag)
#     APGD_D = numpy.load(APGD_D_file)
#     APGD_P = numpy.load(APGD_P_file)
#
#     DICE_D_file = "../dataset/{}/test/Test_DICE_BL_data.npy".format(D_tag)
#     DICE_P_file = "../dataset/{}/test/Test_DICE_BL_position.npy".format(D_tag)
#     DICE_D = numpy.load(DICE_D_file, allow_pickle=True)
#     DICE_P = numpy.load(DICE_P_file)
#
#     ADF_D_file = "../dataset/{}/test/Test_ADF_BL_data.npy".format(D_tag)
#     ADF_P_file = "../dataset/{}/test/Test_ADF_BL_position.npy".format(D_tag)
#     ADF_D = numpy.load(ADF_D_file, allow_pickle=True)
#     ADF_P = numpy.load(ADF_P_file)
#
#     RobustFair_D_file = "../dataset/{}/test/Test_RobustFair_BL_data.npy".format(D_tag)
#     RobustFair_P_file = "../dataset/{}/test/Test_RobustFair_BL_FB_P.npy".format(D_tag)
#     RobustFair_C_file = "../dataset/{}/test/Test_RobustFair_BL_FB_D.npy".format(D_tag)
#
#     RF_D = numpy.load(RobustFair_D_file, allow_pickle=True)
#     AF_P = numpy.load(RobustFair_P_file)
#     AF_C = numpy.load(RobustFair_C_file)
#
#     Num_head = ["TF_R", "TB_R", "FF_R", "FB_R", "SUM"]
#     Fairness_confusion_ACG = robustness_result_evaluation(M_file, ACG_D, D_tag, P_attr)
#     Fairness_confusion_APGD = robustness_result_evaluation(M_file, APGD_D, D_tag, P_attr)
#     Fairness_confusion_ADF = fairness_result_evaluation(M_file, ADF_D, D_tag, P_attr)
#     Fairness_confusion_DICE = fairness_result_evaluation(M_file, DICE_D, D_tag, P_attr)
#     Fairness_confusion_RCF = robust_accurate_fairness_result_evaluation(M_file, RF_D, D_tag, P_attr)
#
#     Rate_head = ["false rate", "bias rate", "false or bias rate"]
#
#     Attack_rate_ACG = [numpy.sum(ACG_P) / ACG_P.shape[0], 0, 0]
#     Attack_rate_APGD = [numpy.sum(APGD_P) / APGD_P.shape[0], 0, 0]
#     Attack_rate_ADF = [0, numpy.sum(ADF_P) / ADF_P.shape[0], 0]
#     Attack_rate_DICE = [0, numpy.sum(DICE_P) / DICE_P.shape[0], 0]
#
#     Attack_rate_ACF = [numpy.sum(numpy.logical_or(AF_C[:, 1], AF_C[:, 2])) / AF_P.shape[0],
#                        numpy.sum(numpy.logical_or(AF_C[:, 0], AF_C[:, 2])) / AF_P.shape[0],
#                        numpy.sum(AF_P) / AF_P.shape[0]]
#
#     Robust_ACG.append(Attack_rate_ACG + Fairness_confusion_ACG)
#     Robust_APGD.append(Attack_rate_APGD + Fairness_confusion_APGD)
#     Fair_ADF.append(Attack_rate_ADF + Fairness_confusion_ADF)
#     Fair_DICE.append(Attack_rate_DICE + Fairness_confusion_DICE)
#     Robust_AccFair.append(Attack_rate_ACF + Fairness_confusion_RCF)
#
#     attack_result = [["ACG"] + numpy.mean(Robust_ACG, axis=0).tolist(),
#                      ["APGD"] + numpy.mean(Robust_APGD, axis=0).tolist(),
#                      ["ADF"] + numpy.mean(Fair_ADF, axis=0).tolist(),
#                      ["DICE"] + numpy.mean(Fair_DICE, axis=0).tolist(),
#                      ["RAF"] + numpy.mean(Robust_AccFair, axis=0).tolist()]
#     workbook_name = xlsxwriter.Workbook(Result_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(["Method"] + Rate_head + Num_head, worksheet)
#     write_worksheet_2d_data(attack_result, worksheet)
#     workbook_name.close()
#
#
# def get_retrain_model_RAF_attack(D_tag, P_attr, Result_file):
#     """
#     获取重训练模型RobustFair的评估结果结果
#     :return:
#     """
#     retrain_model_result = []
#     for method in ["APGD", "ACG", "ADF", "DICE", "RobustFair"]:
#         name = "{}_{}_{}".format("Random", "BL", method)
#         M_file = "../dataset/{}/check/{}.h5".format(D_tag, name)
#
#         RF_evl_file = "../dataset/{}/test/Retrain_RobustFair_{}_data.npy".format(D_tag, name)
#         RF_P_file = "../dataset/{}/test/Retrain_RobustFair_{}_FB_P.npy".format(D_tag, name)
#         RF_AFC_file = "../dataset/{}/test/Retrain_RobustFair_{}_FB_D.npy".format(D_tag, name)
#
#         RF_data = numpy.load(RF_evl_file, allow_pickle=True)
#         AF_P = numpy.load(RF_P_file)
#         AF_D = numpy.load(RF_AFC_file)
#
#         Num_head = ["TFR", "TBR", "FFR", "FBR", "SUM"]
#         Robust_fair_result = robust_accurate_fairness_result_evaluation(M_file, RF_data, D_tag, P_attr)
#         Rate_head = ["false rate", "bias rate", "false or bias rate"]
#         Robust_fair_rate = [numpy.sum(numpy.logical_or(AF_D[:, 1], AF_D[:, 2])) / AF_P.shape[0],
#                             numpy.sum(numpy.logical_or(AF_D[:, 0], AF_D[:, 2])) / AF_P.shape[0],
#                             numpy.sum(AF_P) / AF_P.shape[0]]
#         retrain_model_result.append([name] + Robust_fair_rate + Robust_fair_result)
#
#     workbook_name = xlsxwriter.Workbook(Result_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(["Re-Model"] + Rate_head + Num_head, worksheet)
#     write_worksheet_2d_data(retrain_model_result, worksheet)
#     workbook_name.close()
#
#
# def get_retrain_model_ADF_attack(D_tag, P_attr, Result_file):
#     """
#     获取重训练模型RobustFair的评估结果结果
#     :return:
#     """
#     retrain_model_result = []
#     for method in ["APGD", "ACG", "ADF", "DICE", "RobustFair"]:
#         name = "{}_{}_{}".format("Random", "BL", method)
#         M_file = "../dataset/{}/check/{}.h5".format(D_tag, name)
#
#         E_file = "../dataset/{}/test/Retrain_ADF_{}_data.npy".format(D_tag, name)
#         P_file = "../dataset/{}/test/Retrain_ADF_{}_position.npy".format(D_tag, name)
#
#         eval_D = numpy.load(E_file, allow_pickle=True)
#         eval_P = numpy.load(P_file)
#
#         Num_head = ["TFR", "TBR", "FFR", "FBR", "SUM"]
#         fair_result = fairness_result_evaluation(M_file, eval_D, D_tag, P_attr)
#         Rate_head = ["false rate", "bias rate", "false or bias rate"]
#         Robust_fair_rate = [0, numpy.sum(eval_P) / eval_P.shape[0], 0]
#         retrain_model_result.append([name] + Robust_fair_rate + fair_result)
#
#     workbook_name = xlsxwriter.Workbook(Result_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(["Re-Model"] + Rate_head + Num_head, worksheet)
#     write_worksheet_2d_data(retrain_model_result, worksheet)
#     workbook_name.close()
#
#
# def get_retrain_model_EIDIG_attack(D_tag, P_attr, Result_file):
#     """
#     获取重训练模型RobustFair的评估结果结果
#     :return:
#     """
#     retrain_model_result = []
#     for method in ["APGD", "ACG", "ADF", "DICE", "RobustFair"]:
#         name = "{}_{}_{}".format("Random", "BL", method)
#         M_file = "../dataset/{}/check/{}.h5".format(D_tag, name)
#
#         E_file = "../dataset/{}/test/Retrain_EIDIG_{}_data.npy".format(D_tag, name)
#         P_file = "../dataset/{}/test/Retrain_EIDIG_{}_position.npy".format(D_tag, name)
#
#         eval_D = numpy.load(E_file, allow_pickle=True)
#         eval_P = numpy.load(P_file)
#
#         Num_head = ["TFR", "TBR", "FFR", "FBR", "SUM"]
#         fair_result = fairness_result_evaluation(M_file, eval_D, D_tag, P_attr)
#         Rate_head = ["false rate", "bias rate", "false or bias rate"]
#         Robust_fair_rate = [0, numpy.sum(eval_P) / eval_P.shape[0], 0]
#         retrain_model_result.append([name] + Robust_fair_rate + fair_result)
#
#     workbook_name = xlsxwriter.Workbook(Result_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(["Re-Model"] + Rate_head + Num_head, worksheet)
#     write_worksheet_2d_data(retrain_model_result, worksheet)
#     workbook_name.close()
#
#
# def get_retrain_model_ACG_attack(D_tag, P_attr, Result_file):
#     """
#     获取重训练模型RobustFair的评估结果结果
#     :return:
#     """
#     retrain_model_result = []
#     for method in ["APGD", "ACG", "ADF", "DICE", "RobustFair"]:
#         name = "{}_{}_{}".format("Random", "BL", method)
#         M_file = "../dataset/{}/check/{}.h5".format(D_tag, name)
#
#         E_file = "../dataset/{}/test/Retrain_ACG_{}_data.npy".format(D_tag, name)
#         P_file = "../dataset/{}/test/Retrain_ACG_{}_position.npy".format(D_tag, name)
#
#         eval_D = numpy.load(E_file)
#         eval_P = numpy.load(P_file)
#
#         Num_head = ["TFR", "TBR", "FFR", "FBR", "SUM"]
#         Robust_fair_result = robustness_result_evaluation(M_file, eval_D, D_tag, P_attr)
#         Rate_head = ["false rate", "bias rate", "false or bias rate"]
#         Robust_fair_rate = [numpy.sum(eval_P) / eval_P.shape[0], 0, 0]
#         retrain_model_result.append([name] + Robust_fair_rate + Robust_fair_result)
#
#     workbook_name = xlsxwriter.Workbook(Result_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(["Re-Model"] + Rate_head + Num_head, worksheet)
#     write_worksheet_2d_data(retrain_model_result, worksheet)
#     workbook_name.close()
#
#
# def get_retrain_model_APGD_attack(D_tag, P_attr, Result_file):
#     """
#     获取重训练模型RobustFair的评估结果结果
#     :return:
#     """
#     retrain_model_result = []
#     for method in ["APGD", "ACG", "ADF", "DICE", "RobustFair"]:
#         name = "{}_{}_{}".format("Random", "BL", method)
#         M_file = "../dataset/{}/check/{}.h5".format(D_tag, name)
#
#         E_file = "../dataset/{}/test/Retrain_APGD_{}_data.npy".format(D_tag, name)
#         P_file = "../dataset/{}/test/Retrain_APGD_{}_position.npy".format(D_tag, name)
#
#         eval_D = numpy.load(E_file)
#         eval_P = numpy.load(P_file)
#
#         Num_head = ["TFR", "TBR", "FFR", "FBR", "SUM"]
#         Robust_fair_result = robustness_result_evaluation(M_file, eval_D, D_tag, P_attr)
#         Rate_head = ["false rate", "bias rate", "false or bias rate"]
#         Robust_fair_rate = [numpy.sum(eval_P) / eval_P.shape[0], 0, 0]
#         retrain_model_result.append([name] + Robust_fair_rate + Robust_fair_result)
#
#     workbook_name = xlsxwriter.Workbook(Result_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(["Re-Model"] + Rate_head + Num_head, worksheet)
#     write_worksheet_2d_data(retrain_model_result, worksheet)
#     workbook_name.close()
#
#
# def combine_retrain_attacks(D_tag, Result_file):
#     """
#
#     :return:
#     """
#     RAF_file = "../dataset/{}/result/Result_RAF_attack_retrain.xlsx".format(D_tag)
#     ADF_file = "../dataset/{}/result/Result_ADF_attack_retrain.xlsx".format(D_tag)
#     EIDIG_file = "../dataset/{}/result/Result_EIDIG_attack_retrain.xlsx".format(D_tag)
#     ACG_file = "../dataset/{}/result/Result_ACG_attack_retrain.xlsx".format(D_tag)
#     APGD_file = "../dataset/{}/result/Result_APGD_attack_retrain.xlsx".format(D_tag)
#
#     RAF_data = pandas.read_excel(RAF_file)
#     RAF_FB_R = numpy.split(RAF_data, [3, 4], axis=1)[1]
#
#     ADF_data = pandas.read_excel(ADF_file)
#     ADF_F_R = numpy.split(ADF_data, [2, 3], axis=1)[1]
#
#     EIDIG_data = pandas.read_excel(EIDIG_file)
#     EIDIG_F_R = numpy.split(EIDIG_data, [2, 3], axis=1)[1]
#
#     ACG_data = pandas.read_excel(ACG_file)
#     ACG_B_R = numpy.split(ACG_data, [1, 2], axis=1)[1]
#
#     APGD_data = pandas.read_excel(APGD_file)
#     APGD_B_R = numpy.split(APGD_data, [1, 2], axis=1)[1]
#     result = [numpy.array(ACG_B_R).reshape(1, -1), numpy.array(APGD_B_R).reshape(1, -1),
#               numpy.array(ADF_F_R).reshape(1, -1), numpy.array(EIDIG_F_R).reshape(1, -1),
#               numpy.array(RAF_FB_R).reshape(1, -1)]
#     result = numpy.transpose(numpy.squeeze(numpy.array(result)))
#
#     workbook_name = xlsxwriter.Workbook(Result_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(["False", "False", "Biased", "Biased", "False or Biased"], worksheet)
#     write_worksheet_2d_data(result, worksheet)
#     workbook_name.close()
#     print()
#
#
# def get_similar_data(cluster_data, generate_data, low_bound, up_bound):
#     """
#     计算生成样本与原始训练集的的相似性
#     :return:
#     """
#     similarity, _ = calculate_cos(cluster_data, generate_data, 1.0)
#     low_position = low_bound < similarity
#     up_position = similarity <= up_bound
#     data_position = numpy.logical_and(low_position, up_position)
#     return data_position
#
#
# def get_generate_fair_data(data):
#     """
#
#     :return:
#     """
#     generated_data = data[0]
#     for i in range(data.shape[0] - 1):
#         generated_data = numpy.concatenate((generated_data, data[i + 1]), axis=0)
#     # generated_similar = similar_data[0]
#     # for i in range(similar_data.shape[0] - 1):
#     #     generated_similar = numpy.concatenate((generated_similar, similar_data[i + 1]), axis=0)
#     # # 为相似样本生成标记
#     # x1, y1 = numpy.split(generated_data, [-1, ], axis=1)
#     # for j in range(generated_similar.shape[1]):
#     #     similar_data_label = numpy.concatenate((generated_similar[:, j, :], y1), axis=1)
#     #     generated_data = numpy.concatenate((generated_data, similar_data_label), axis=0)
#     return generated_data
#
#
# def check_similarity(cluster_data, generate_data, Result_file):
#     """
#     检查生成样本的相似性
#     :return:
#     """
#     result = []
#     Bounds = [0, 0.70, 0.8, 0.9, 1.0]
#     for r in range(len(Bounds) - 1):
#         similarity = get_similar_data(cluster_data, generate_data, Bounds[r], Bounds[r + 1])
#         numpy.save(Result_file.format(Bounds[r], Bounds[r + 1]), similarity)
#         result.append(numpy.sum(similarity))
#     return result
#
#
# def test_datat_similarity_evaluation(D_tag, Result_file):
#     """
#
#     :return:
#     """
#     Seed_data = numpy.load("../dataset/{}/test/test_avg_clusters.npy".format(D_tag))
#
#     ACG_data = numpy.load("../dataset/{}/test/Test_ACG_BL_data.npy".format(D_tag))
#     ACG_R_file = "../dataset/" + D_tag + "/test/Test_ACG_BL_data_{}_{}.npy"
#     ACG_result = check_similarity(Seed_data, ACG_data, ACG_R_file)
#
#     APGD_data = numpy.load("../dataset/{}/test/Test_APGD_BL_data.npy".format(D_tag))
#     APGD_R_file = "../dataset/" + D_tag + "/test/Test_APGD_BL_data_{}_{}.npy"
#     APGD_result = check_similarity(Seed_data, APGD_data, APGD_R_file)
#
#     ADF_data = numpy.load("../dataset/{}/test/Test_ADF_BL_data.npy".format(D_tag), allow_pickle=True)
#     ADF_R_file = "../dataset/" + D_tag + "/test/Test_ADF_BL_data_{}_{}.npy"
#     ADF_result = check_similarity(Seed_data, get_generate_fair_data(ADF_data), ADF_R_file)
#
#     EIDIG_data = numpy.load("../dataset/{}/test/Test_EIDIG_BL_data.npy".format(D_tag), allow_pickle=True)
#     EIDIG_R_file = "../dataset/" + D_tag + "/test/Test_EIDIG_BL_data_{}_{}.npy"
#     EIDIG_result = check_similarity(Seed_data, get_generate_fair_data(EIDIG_data), EIDIG_R_file)
#
#     DICE_data = numpy.load("../dataset/{}/test/Test_DICE_BL_data.npy".format(D_tag))
#     DICE_R_file = "../dataset/" + D_tag + "/test/Test_DICE_BL_data_{}_{}.npy"
#     DICE_result = check_similarity(Seed_data, DICE_data, DICE_R_file)
#
#     RF_data = numpy.load("../dataset/{}/test/Test_RobustFair_BL_data.npy".format(D_tag), allow_pickle=True)
#     RF_R_file = "../dataset/" + D_tag + "/test/Test_RF_BL_data_{}_{}.npy"
#     RF_result = check_similarity(Seed_data, get_generate_fair_data(RF_data), RF_R_file)
#
#     similarity_result = [["ACG"] + ACG_result,
#                          ["APGD"] + APGD_result,
#                          ["ADF"] + ADF_result,
#                          ["EIDIG"] + EIDIG_result,
#                          ["DICE"] + DICE_result,
#                          ["RobustFair"] + RF_result]
#     similarity_head = ["method", "0-0.7", "0.7-0.75", "0.75-0.8", "0.8-0.85", "0.85-0.9", "0.9-0.95", "0.95-1.0"]
#     workbook_name = xlsxwriter.Workbook(Result_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(similarity_head, worksheet)
#     write_worksheet_2d_data(similarity_result, worksheet)
#     workbook_name.close()
#
#
# def train_datat_similarity_evaluation(D_tag, Result_file):
#     """
#
#     :return:
#     """
#     Seed_data = numpy.load("../dataset/{}/test/retrain_avg_clusters.npy".format(D_tag))
#
#     ACG_data = numpy.load("../dataset/{}/test/Retrain_ACG_BL_data.npy".format(D_tag))
#     ACG_R_file = "../dataset/" + D_tag + "/test/Retrain_ACG_BL_data_{}_{}.npy"
#     ACG_result = check_similarity(Seed_data, ACG_data, ACG_R_file)
#
#     APGD_data = numpy.load("../dataset/{}/test/Retrain_APGD_BL_data.npy".format(D_tag))
#     APGD_R_file = "../dataset/" + D_tag + "/test/Retrain_APGD_BL_data_{}_{}.npy"
#     APGD_result = check_similarity(Seed_data, APGD_data, APGD_R_file)
#
#     ADF_data = numpy.load("../dataset/{}/test/Retrain_ADF_BL_data.npy".format(D_tag), allow_pickle=True)
#     ADF_R_file = "../dataset/" + D_tag + "/test/Retrain_ADF_BL_data_{}_{}.npy"
#     ADF_result = check_similarity(Seed_data, get_generate_fair_data(ADF_data), ADF_R_file)
#
#     EIDIG_data = numpy.load("../dataset/{}/test/Retrain_EIDIG_BL_data.npy".format(D_tag), allow_pickle=True)
#     EIDIG_R_file = "../dataset/" + D_tag + "/test/Retrain_EIDIG_BL_data_{}_{}.npy"
#     EIDIG_result = check_similarity(Seed_data, get_generate_fair_data(EIDIG_data), EIDIG_R_file)
#
#     DICE_data = numpy.load("../dataset/{}/test/Retrain_DICE_BL_data.npy".format(D_tag))
#     DICE_R_file = "../dataset/" + D_tag + "/test/Retrain_DICE_BL_data_{}_{}.npy"
#     DICE_result = check_similarity(Seed_data, DICE_data, DICE_R_file)
#
#     RF_data = numpy.load("../dataset/{}/test/Retrain_RobustFair_BL_data.npy".format(D_tag), allow_pickle=True)
#     RF_R_file = "../dataset/" + D_tag + "/test/Retrain_RF_BL_data_{}_{}.npy"
#     RF_result = check_similarity(Seed_data, get_generate_fair_data(RF_data), RF_R_file)
#
#     similarity_result = [["ACG"] + ACG_result,
#                          ["APGD"] + APGD_result,
#                          ["ADF"] + ADF_result,
#                          ["EIDIG"] + EIDIG_result,
#                          ["DICE"] + DICE_result,
#                          ["RobustFair"] + RF_result]
#     similarity_head = ["method", "0-0.7", "0.7-0.75", "0.75-0.8", "0.8-0.85", "0.85-0.9", "0.9-0.95", "0.95-1.0"]
#     workbook_name = xlsxwriter.Workbook(Result_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(similarity_head, worksheet)
#     write_worksheet_2d_data(similarity_result, worksheet)
#     workbook_name.close()
#
#
# def check_similarity2(cluster_data, generate_data, Result_file):
#     """
#     检查生成样本的相似性
#     :return:
#     """
#     result = []
#     Bounds = [0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
#     for r in range(len(Bounds)):
#         similarity = get_similar_data(cluster_data, generate_data, Bounds[r], 1.0)
#         numpy.save(Result_file.format(Bounds[r], 1.0), similarity)
#         result.append(numpy.sum(similarity))
#     return result
#
#
# def train_datat_similarity_evaluation2(D_tag, Result_file):
#     """
#
#     :return:
#     """
#     Seed_data = numpy.load("../dataset/{}/test/retrain_avg_clusters.npy".format(D_tag))
#     ACG_data = numpy.load("../dataset/{}/test/Retrain_ACG_BL_data.npy".format(D_tag))
#     ACG_R_file = "../dataset/" + D_tag + "/test/Retrain_ACG_BL_data_{}_{}2.npy"
#     ACG_result = check_similarity2(Seed_data, ACG_data, ACG_R_file)
#     APGD_data = numpy.load("../dataset/{}/test/Retrain_APGD_BL_data.npy".format(D_tag))
#     APGD_R_file = "../dataset/" + D_tag + "/test/Retrain_APGD_BL_data_{}_{}2.npy"
#     APGD_result = check_similarity2(Seed_data, APGD_data, APGD_R_file)
#     ADF_data = numpy.load("../dataset/{}/test/Retrain_ADF_BL_data.npy".format(D_tag), allow_pickle=True)
#     ADF_R_file = "../dataset/" + D_tag + "/test/Retrain_ADF_BL_data_{}_{}2.npy"
#     ADF_result = check_similarity2(Seed_data, get_generate_fair_data(ADF_data), ADF_R_file)
#     EIDIG_data = numpy.load("../dataset/{}/test/Retrain_EIDIG_BL_data.npy".format(D_tag), allow_pickle=True)
#     EIDIG_R_file = "../dataset/" + D_tag + "/test/Retrain_EIDIG_BL_data_{}_{}2.npy"
#     EIDIG_result = check_similarity2(Seed_data, get_generate_fair_data(EIDIG_data), EIDIG_R_file)
#
#     DICE_data = numpy.load("../dataset/{}/test/Retrain_DICE_BL_data.npy".format(D_tag))
#     DICE_R_file = "../dataset/" + D_tag + "/test/Retrain_DICE_BL_data_{}_{}2.npy"
#     DICE_result = check_similarity2(Seed_data, DICE_data, DICE_R_file)
#
#     RF_data = numpy.load("../dataset/{}/test/Retrain_RobustFair_BL_data.npy".format(D_tag), allow_pickle=True)
#     RF_R_file = "../dataset/" + D_tag + "/test/Retrain_RF_BL_data_{}_{}2.npy"
#     RF_result = check_similarity2(Seed_data, get_generate_fair_data(RF_data), RF_R_file)
#
#     similarity_result = [["ACG"] + ACG_result,
#                          ["APGD"] + APGD_result,
#                          ["ADF"] + ADF_result,
#                          ["EIDIG"] + EIDIG_result,
#                          ["DICE"] + DICE_result,
#                          ["RobustFair"] + RF_result]
#     similarity_head = ["method", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95"]
#     workbook_name = xlsxwriter.Workbook(Result_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(similarity_head, worksheet)
#     write_worksheet_2d_data(similarity_result, worksheet)
#     workbook_name.close()
#
#
# def get_retrain_robustness_fairness_data(D_tag, R_name, F_name, low_S, up_S, P_attr, R_num):
#     """
#     获取重训练所需的对抗样本与歧视样本
#     :return:
#     """
#     R_data = numpy.load("../dataset/{}/test/Retrain_{}_BL_data.npy".format(D_tag, R_name))
#     R_S_D = numpy.load("../dataset/{}/test/Retrain_{}_BL_data_{}_{}2.npy".format(D_tag, R_name, low_S, up_S))
#     F_data = numpy.load("../dataset/{}/test/Retrain_{}_BL_data.npy".format(D_tag, F_name), allow_pickle=True)
#     F_data = get_generate_fair_data(F_data)
#     F_S_D = numpy.load("../dataset/{}/test/Retrain_{}_BL_data_{}_{}2.npy".format(D_tag, F_name, low_S, up_S))
#
#     # # 1.测试过程中的相似样本
#     # F_data_S = numpy.load("../dataset/{}/test/Retrain_{}_BL_data_similar.npy".format(D_tag, F_name),
#     #                       allow_pickle=True)
#     # F_data_S = get_generate_fair_data(F_data_S)
#
#     R_data = R_data[R_S_D]
#     numpy.random.shuffle(R_data)
#     R_data, _ = numpy.split(R_data, [R_num, ], axis=0)
#
#     # # 2.数据增强生成的相似样本
#     # R_data_s = generate_similar_items(R_data, D_tag, P_attr)
#
#     F_data = F_data[F_S_D]
#     numpy.random.shuffle(F_data)
#     F_data, _ = numpy.split(F_data, [R_num, ], axis=0)
#
#     # # 2.数据增强生成的相似样本
#     # F_data_S = generate_similar_items(F_data, D_tag, P_attr)
#
#     # # 1.测试过程中的相似样本
#     # F_data_S = F_data_S[F_S_D]
#     # numpy.random.shuffle(F_data_S)
#     # F_data_S, _ = numpy.split(F_data_S, [R_num, ], axis=0)
#
#     Retrain_data = numpy.concatenate((R_data, F_data), axis=0)
#
#     # # 数据增强，添加相似样本
#     # _, y_S = numpy.split(F_data, [-1, ], axis=1)
#     # for i in range(F_data_S.shape[1]):
#     #     # # 1.测试过程中的相似样本
#     #     # S_data = numpy.concatenate((F_data_S[:, i, :], y_S), axis=1)
#     #     # Retrain_data = numpy.concatenate((Retrain_data, S_data), axis=0)
#     #
#     #     # 2.数据增强生成的相似样本
#     #     S_data = F_data_S[:, i, :]
#     #     Retrain_data = numpy.concatenate((Retrain_data, S_data), axis=0)
#
#     # # 数据增强，添加相似样本
#     # _, y_S = numpy.split(R_data, [-1, ], axis=1)
#     # for i in range(R_data_s.shape[1]):
#     #     # # 1.测试过程中的相似样本
#     #     # S_data = numpy.concatenate((R_data_s[:, i, :], y_S), axis=1)
#     #     # Retrain_data = numpy.concatenate((Retrain_data, S_data), axis=0)
#     #
#     #     # # 2.数据增强生成的相似样本
#     #     # S_data = R_data_s[:, i, :]
#     #     # Retrain_data = numpy.concatenate((Retrain_data, S_data), axis=0)
#
#     return numpy.split(Retrain_data, [-1, ], axis=1)
#
#
# def get_retrain_robustness_data(D_tag, R_name, F_name, low_S, up_S, P_attr, R_num):
#     """
#     获取重训练所需的对抗样本与歧视样本
#     :return:
#     """
#     R_data = numpy.load("../dataset/{}/test/Retrain_{}_BL_data.npy".format(D_tag, R_name))
#     R_S_D = numpy.load("../dataset/{}/test/Retrain_{}_BL_data_{}_{}2.npy".format(D_tag, R_name, low_S, up_S))
#
#     R_data = R_data[R_S_D]
#     numpy.random.shuffle(R_data)
#     R_data, _ = numpy.split(R_data, [R_num, ], axis=0)
#
#     return numpy.split(R_data, [-1, ], axis=1)
#
#
# def get_retrain_fairness_data(D_tag, R_name, F_name, low_S, up_S, P_attr, R_num):
#     """
#     获取重训练所需的对抗样本与歧视样本
#     :return:
#     """
#     F_data = numpy.load("../dataset/{}/test/Retrain_{}_BL_data.npy".format(D_tag, F_name), allow_pickle=True)
#     F_data = get_generate_fair_data(F_data)
#     F_S_D = numpy.load("../dataset/{}/test/Retrain_{}_BL_data_{}_{}2.npy".format(D_tag, F_name, low_S, up_S))
#
#     F_data = F_data[F_S_D]
#     numpy.random.shuffle(F_data)
#     F_data, _ = numpy.split(F_data, [R_num, ], axis=0)
#
#     return numpy.split(F_data, [-1, ], axis=1)
#
#
# def get_retrain_robust_fair_data(D_tag, low_S, up_S, P_attr, R_num):
#     """
#     获取重训练所需的对抗样本与歧视样本
#     :return:
#     """
#     # 生成样本与相似样本
#     F_data = numpy.load("../dataset/{}/test/Retrain_RobustFair_BL_data.npy".format(D_tag), allow_pickle=True)
#     F_data = get_generate_fair_data(F_data)
#     # similarity相似位置
#     F_S_D = numpy.load("../dataset/{}/test/Retrain_RF_BL_data_{}_{}2.npy".format(D_tag, low_S, up_S))
#
#     # # 1.测试过程中的相似样本
#     # F_data_S = numpy.load("../dataset/{}/test/Retrain_RobustFair_BL_data_similar.npy".format(D_tag),
#     #                       allow_pickle=True)
#     # F_data_S = get_generate_fair_data(F_data_S)
#
#     # similarity数据与相似数据
#     F_data = F_data[F_S_D]
#     numpy.random.shuffle(F_data)
#     F_data, _ = numpy.split(F_data, [R_num, ], axis=0)
#
#     # # 2.数据增强生成的相似样本
#     # F_data_S = generate_similar_items(F_data, D_tag, P_attr)
#
#     # # 1.测试过程中的相似样本
#     # F_data_S = F_data_S[F_S_D]
#     # numpy.random.shuffle(F_data_S)
#     # F_data_S, _ = numpy.split(F_data_S, [R_num, ], axis=0)
#
#     # # # 数据增强，添加相似样本
#     # # _, y_S = numpy.split(F_data, [-1, ], axis=1)
#     # for i in range(F_data_S.shape[1]):
#     #     # # 1.测试过程中的相似样本
#     #     # S_data = numpy.concatenate((F_data_S[:, i, :], y_S), axis=1)
#     #     # F_data = numpy.concatenate((F_data, S_data), axis=0)
#     #
#     #     # 2.数据增强生成的相似样本
#     #     S_data = F_data_S[:, i, :]
#     #     F_data = numpy.concatenate((F_data, S_data), axis=0)
#     return numpy.split(F_data, [-1, ], axis=1)
#
#
# def get_retrain_robustness_data_random(D_tag, R_name, F_name, low_S, up_S, P_attr, R_num):
#     """
#     获取重训练所需的对抗样本与歧视样本
#     :return:
#     """
#     R_data = numpy.load("../dataset/{}/test/Retrain_{}_BL_data.npy".format(D_tag, R_name))
#
#     numpy.random.shuffle(R_data)
#     R_data, _ = numpy.split(R_data, [R_num, ], axis=0)
#
#     return numpy.split(R_data, [-1, ], axis=1)
#
#
# def get_retrain_fairness_data_random(D_tag, R_name, F_name, low_S, up_S, P_attr, R_num):
#     """
#     获取重训练所需的对抗样本与歧视样本
#     :return:
#     """
#     F_data = numpy.load("../dataset/{}/test/Retrain_{}_BL_data.npy".format(D_tag, F_name), allow_pickle=True)
#     F_data = get_generate_fair_data(F_data)
#
#     numpy.random.shuffle(F_data)
#     F_data, _ = numpy.split(F_data, [R_num, ], axis=0)
#
#     return numpy.split(F_data, [-1, ], axis=1)
#
#
# def get_retrain_robustness_fairness_data_random(D_tag, R_name, F_name, P_attr, R_num):
#     """
#     获取重训练所需的对抗样本与歧视样本
#     :return:
#     """
#     R_data = numpy.load("../dataset/{}/test/Retrain_{}_BL_data.npy".format(D_tag, R_name))
#     F_data = numpy.load("../dataset/{}/test/Retrain_{}_BL_data.npy".format(D_tag, F_name), allow_pickle=True)
#     F_data = get_generate_fair_data(F_data)
#
#     # # 1.测试过程中的相似样本
#     # F_data_S = numpy.load("../dataset/{}/test/Retrain_{}_BL_data_similar.npy".format(D_tag, F_name),
#     #                       allow_pickle=True)
#     # F_data_S = get_generate_fair_data(F_data_S)
#
#     numpy.random.shuffle(R_data)
#     R_data, _ = numpy.split(R_data, [R_num, ], axis=0)
#
#     # # 2.数据增强生成的相似样本
#     # R_data_s = generate_similar_items(R_data, D_tag, P_attr)
#
#     numpy.random.shuffle(F_data)
#     F_data, _ = numpy.split(F_data, [R_num, ], axis=0)
#
#     # # 2.数据增强生成的相似样本
#     # F_data_S = generate_similar_items(F_data, D_tag, P_attr)
#
#     # # 1.测试过程中的相似样本
#     # F_data_S = F_data_S[F_S_D]
#     # numpy.random.shuffle(F_data_S)
#     # F_data_S, _ = numpy.split(F_data_S, [R_num, ], axis=0)
#
#     Retrain_data = numpy.concatenate((R_data, F_data), axis=0)
#
#     # # 数据增强，添加相似样本
#     # _, y_S = numpy.split(F_data, [-1, ], axis=1)
#     # for i in range(F_data_S.shape[1]):
#     #     # # 1.测试过程中的相似样本
#     #     # S_data = numpy.concatenate((F_data_S[:, i, :], y_S), axis=1)
#     #     # Retrain_data = numpy.concatenate((Retrain_data, S_data), axis=0)
#     #
#     #     # 2.数据增强生成的相似样本
#     #     S_data = F_data_S[:, i, :]
#     #     Retrain_data = numpy.concatenate((Retrain_data, S_data), axis=0)
#
#     # # 数据增强，添加相似样本
#     # _, y_S = numpy.split(R_data, [-1, ], axis=1)
#     # for i in range(R_data_s.shape[1]):
#     #     # # 1.测试过程中的相似样本
#     #     # S_data = numpy.concatenate((R_data_s[:, i, :], y_S), axis=1)
#     #     # Retrain_data = numpy.concatenate((Retrain_data, S_data), axis=0)
#     #
#     #     # # 2.数据增强生成的相似样本
#     #     # S_data = R_data_s[:, i, :]
#     #     # Retrain_data = numpy.concatenate((Retrain_data, S_data), axis=0)
#
#     return numpy.split(Retrain_data, [-1, ], axis=1)
#
#
# def data_augmentation_adv_adult(data, protected_index):
#     """
#     对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
#     :return:
#     """
#     aug = []
#     for i in range(data.shape[0]):
#         age = [uniform(17, 30), uniform(30, 60), uniform(60, 90)]
#         race = [uniform(0, 2), uniform(2, 4)]
#         sex = [uniform(0, 0.5), uniform(0.5, 1.0)]
#         a_0 = age[random.randint(0, 2)]
#         a_1 = race[random.randint(0, 1)]
#         a_2 = sex[random.randint(0, 1)]
#         aug_data = data[i].tolist()
#         aug_data[protected_index[0]] = round(a_0) / (90 - 17)  # 归一化
#         aug_data[protected_index[1]] = round(a_1) / (4)
#         aug_data[protected_index[2]] = round(a_2) / (1)
#         aug.append(aug_data)
#     return numpy.squeeze(numpy.array(aug))
#
#
# def data_augmentation_adv_credit(data, protected_index):
#     """
#     对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
#     :return:
#     """
#     aug = []
#     for i in range(data.shape[0]):
#         sex = [0, 1]
#         age = [uniform(19, 35), uniform(35, 55), uniform(55, 75)]
#         a_0 = sex[random.randint(0, 1)]
#         a_1 = age[random.randint(0, 2)]
#         aug_data = data[i].tolist()
#         aug_data[protected_index[0]] = a_0
#         aug_data[protected_index[1]] = round(a_1) / (75 - 19)  # 归一化
#         aug.append(aug_data)
#     return numpy.squeeze(numpy.array(aug))
#
#
# def data_augmentation_adv_compas(data, protected_index):
#     """
#     对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
#     :return:
#     """
#     aug = []
#     for i in range(data.shape[0]):
#         age = [uniform(18, 30), uniform(30, 60), uniform(60, 96)]
#         race = [uniform(0, 2.5), uniform(2.5, 5)]
#         sex = [uniform(0, 0.5), uniform(0.5, 1.0)]
#         a_0 = age[random.randint(0, 1)]
#         a_1 = race[random.randint(0, 1)]
#         a_2 = sex[random.randint(0, 1)]
#         aug_data = data[i].tolist()
#         aug_data[protected_index[0]] = round(a_0) / (96 - 18)  # 归一化
#         aug_data[protected_index[1]] = round(a_1) / (5)
#         aug_data[protected_index[2]] = round(a_2) / (1)
#         aug.append(aug_data)
#     return numpy.squeeze(numpy.array(aug))
#
#
# def data_augmentation_adv_bank(data, protected_index):
#     """
#     对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
#     :return:
#     """
#     aug = []
#     for i in range(data.shape[0]):
#         age = [uniform(18, 35), uniform(35, 55), uniform(55, 75), uniform(75, 95)]
#         a_1 = age[random.randint(0, 2)]
#         aug_data = data[i].tolist()
#         aug_data[protected_index[0]] = round(a_1) / (95 - 18)  # 归一化
#         aug.append(aug_data)
#     return numpy.squeeze(numpy.array(aug))
#
#
# def generate_similar_adv_dataset(D_tag, dataset, protected_attr):
#     """
#     生成样本的相似样本
#     :return:
#     """
#     # 生成相似样本
#     if D_tag == "adult":
#         similar_dataset = data_augmentation_adv_adult(dataset, protected_attr)
#     elif D_tag == "compas":
#         similar_dataset = data_augmentation_adv_compas(dataset, protected_attr)
#     elif D_tag == "credit":
#         similar_dataset = data_augmentation_adv_credit(dataset, protected_attr)
#     else:
#         similar_dataset = data_augmentation_adv_bank(dataset, protected_attr)
#     return similar_dataset
#
#
# def get_retrain_robust_fair_data_random(D_tag, P_attr, R_num):
#     """
#     获取重训练所需的对抗样本与歧视样本
#     :return:
#     """
#     # 选取生成生成样本
#     F_data = numpy.load("../dataset/{}/test/Retrain_RobustFair_BL_data.npy".format(D_tag), allow_pickle=True)
#     F_data = get_generate_fair_data(F_data)
#     numpy.random.shuffle(F_data)
#     F_data, _ = numpy.split(F_data, [R_num, ], axis=0)
#
#     # 2.数据增强生成样本的相似样本,合并生成样本与相似样本
#     F_data_S = generate_similar_adv_dataset(D_tag, F_data, P_attr)
#     F_data = numpy.concatenate((F_data, F_data_S), axis=0)
#     return numpy.split(F_data, [-1, ], axis=1)
#
#
# def get_model_performance(D_tag):
#     """
#
#     :return:
#     """
#     ACC = []
#     IF = []
#     AF = []
#     F_F1 = []
#     for cos in [0.75, 0.8, 0.85, 0.9, 0.95]:
#         eval_data = pandas.read_excel("../dataset/{}/result/Result_retrain_eval_{}.xlsx".format(D_tag, cos)).values
#         ACC.append(eval_data[1:, 1])
#         IF.append(eval_data[1:, 5])
#         AF.append(eval_data[1:, 9])
#         F_F1.append(eval_data[1:, 19])
#         name = eval_data[1:, 0]
#     return name.tolist(), numpy.array(ACC, dtype=float), numpy.array(IF, dtype=float), numpy.array(AF, dtype=float), \
#            numpy.array(F_F1, dtype=float)
#
#
# def combine_model_evaluations(D_tag, result_file):
#     """
#     将BL模型、重训练模型的评估结果合并
#     :return:
#     """
#     eval_data = []
#     # BL model
#     BL_f = "../dataset/{}/result/Train_BL_evaluation.xlsx".format(D_tag)
#     BL_eval = pandas.read_excel(BL_f)
#     eval_data.append(["Model"] + BL_eval.columns.values.reshape(1, -1).tolist()[0])
#     eval_data.append(["BL"] + BL_eval.values[0, :].reshape(1, -1).tolist()[0])
#     # check model
#     for r in ["APGD", "ACG", "ADF", "DICE"]:
#         CheckFile = "../dataset/{}/result/Test_{}_evaluation.xlsx".format(D_tag, r)
#         R_eval = pandas.read_excel(CheckFile)
#         eval_data.append(["{}".format(r)] + R_eval.values[0, :].reshape(1, -1).tolist()[0])
#
#     # for f in []:
#     #     F_eval = pandas.read_excel("../dataset/{}/result/Retrain_Random_BL_{}_evaluation.xlsx".format(D_tag, f))
#     #     eval_data.append(["{}".format(f)] + F_eval.values[0, :].reshape(1, -1).tolist()[0])
#
#     for r_f in ["TB", "FF", "FB"]:
#         ResultFile = "../dataset/{}/result/Test_RobustFair_{}_evaluation.xlsx".format(D_tag, r_f)
#         eval_RF = pandas.read_excel(ResultFile)
#         eval_data.append(["{}".format(r_f)] + eval_RF.values[0, :].reshape(1, -1).tolist()[0])
#
#     eval_data = numpy.squeeze(numpy.array(eval_data))
#     workbook_name = xlsxwriter.Workbook(result_file)
#     worksheet = workbook_name.add_worksheet("result Details")
#     write_worksheet_2d_data(eval_data, worksheet)
#     workbook_name.close()
#
#
# def combine_model_evaluations2(D_tag, result_file):
#     """
#     将BL模型、重训练模型的评估结果合并
#     :return:
#     """
#     eval_data = []
#     # BL model
#     BL_f = "../dataset/{}/result/Test_RobustFair_TF_evaluation.xlsx".format(D_tag)
#     BL_eval = pandas.read_excel(BL_f)
#     eval_data.append(["Model"] + BL_eval.columns.values.reshape(1, -1).tolist()[0])
#     eval_data.append(["BL"] + BL_eval.values[0, :].reshape(1, -1).tolist()[0])
#     # check model
#     for r in ["APGD", "ACG", "ADF", "DICE"]:
#         CheckFile = "../dataset/{}/result/Test_DICE_TF_evaluation.xlsx".format(D_tag, r)
#         R_eval = pandas.read_excel(CheckFile)
#         eval_data.append(["{}".format(r)] + R_eval.values[0, :].reshape(1, -1).tolist()[0])
#
#     for r_f in ["TB", "FF", "FB"]:
#         ResultFile = "../dataset/{}/result/Test_RobustFair_{}_TF_evaluation.xlsx".format(D_tag, r_f)
#         eval_RF = pandas.read_excel(ResultFile)
#         eval_data.append(["{}".format(r_f)] + eval_RF.values[0, :].reshape(1, -1).tolist()[0])
#
#     eval_data = numpy.squeeze(numpy.array(eval_data))
#     workbook_name = xlsxwriter.Workbook(result_file)
#     worksheet = workbook_name.add_worksheet("result Details")
#     write_worksheet_2d_data(eval_data, worksheet)
#     workbook_name.close()
