import numpy

from utils.utils_evaluate import robust_accurate_fairness_result_evaluation


def numpy_equal_to_value(inputs, value):
    """
    计算inputs中属性取值为value的位置
    :return:
    """
    condition = []
    for i in range(inputs.shape[0]):
        if (inputs[i] == value).all():
            condition.append(True)
        else:
            condition.append(False)
    return numpy.array(condition).astype(bool)


def numpy_range_from_start_to_end(inputs, start_value, end_value):
    """
    计算inputs中属性取值为value的位置
    :return:
    """
    condition = []
    for i in range(inputs.shape[0]):
        if start_value <= inputs[i] < end_value:
            condition.append(True)
        else:
            condition.append(False)
    return numpy.array(condition).astype(bool)


def get_evaluation_data1(data_file, position_file):
    """
    获取对抗扰动后预测结果仍然准确且公平的样本 Robustness,DICE
    :return:
    """
    data = numpy.load(data_file)
    position = numpy.load(position_file)

    return data[position], data[~position_file]


def get_evaluation_data2(data_file, position_file):
    """
    获取对抗扰动后预测结果仍然准确且公平的样本,ADF EIDIG
    :return:
    """
    data = numpy.load(data_file, allow_pickle=True)
    position = numpy.load(position_file)

    return data[position], data[~position_file]


def get_evaluation_data3(data_file, position_file, accurate_fairness_confusion_file):
    """
    获取对抗扰动后预测结果仍然准确且公平的样本,RobustFair
    :return:
    """
    data = numpy.load(data_file, allow_pickle=True)
    position = numpy.load(position_file)
    afc = numpy.load(accurate_fairness_confusion_file)
    TF_data = data[~position]
    TB_data = data[afc[0]]
    FF_data = data[afc[1]]
    FB_data = data[afc[2]]

    return TF_data, TB_data, FF_data, FB_data


def order_diff_data(original_data, name_data):
    """
    将 treatment difference 数据按从小到大的顺序排列，并将对应的group数据对应
    :return:
    """
    order_data = []
    order_name = []
    for i in range(len(original_data)):
        min_data = 100
        min_index = i
        for j in range(len(original_data)):
            if original_data[j] < min_data:
                min_data = original_data[j]
                min_index = j
        original_data[min_index] = 100
        order_data.append(min_data)
        order_name.append(name_data[min_index])
    return order_data, order_name


def analysis_adult(age, race, sex, Position):
    """

    :return:
    """
    # 保护属性分组
    Rate_Results = []
    Name_Results = []
    age_start = [10, 30, 60]
    age_end = [30, 60, 100]
    for a_i in range(3):
        G_age = numpy_range_from_start_to_end(age, age_start[a_i] / (90 - 17), age_end[a_i] / (90 - 17))
        for r_i in range(5):
            G_race = numpy_equal_to_value(race, (r_i / 4))
            for s_i in range(2):
                G_sex = numpy_equal_to_value(sex, s_i)
                Group_Position = numpy.logical_and(G_age, numpy.logical_and(G_race, G_sex))
                if numpy.sum(Group_Position) == 0:
                    continue
                Group_Bias = numpy.logical_and(Group_Position, Position)
                Group_Bias_Rate = numpy.sum(Group_Bias) / numpy.sum(Group_Position)
                Rate_Results.append(Group_Bias_Rate)
                Name_Results.append("age:{},Race:{},Sex:{}".format(age_start[a_i], r_i, s_i))
    a, b = order_diff_data(Rate_Results, Name_Results)
    return order_diff_data(Rate_Results, Name_Results)


def get_evaluation_rate(AF_P, AF_C):
    """

    :return:
    """
    Robust_fair_rate = [numpy.sum(numpy.logical_or(AF_C[:, 1], AF_C[:, 2])) / AF_P.shape[0],
                        numpy.sum(numpy.logical_or(AF_C[:, 0], AF_C[:, 2])) / AF_P.shape[0],
                        numpy.sum(AF_P) / AF_P.shape[0],
                        numpy.sum(~AF_P) / AF_P.shape[0],
                        numpy.sum(AF_C[:, 0]) / AF_P.shape[0],
                        numpy.sum(AF_C[:, 1]) / AF_P.shape[0],
                        numpy.sum(AF_C[:, 2]) / AF_P.shape[0]]
    return Robust_fair_rate


def analysis_adult_result(model_file, test_file, data_file, similar_file, position_file, detail_file):
    """

    :return:
    """

    data = numpy.load(data_file, allow_pickle=True)
    similar = numpy.load(similar_file, allow_pickle=True)
    num_result = []
    for i in range(data.shape[0]):
        num_result.append(robust_accurate_fairness_result_evaluation(model_file, data[i], similar[i], test_file))

    # feature, label = numpy.split(test_data, [-1, ], axis=1)
    # age = feature[:, 10]
    # race = feature[:, 11]
    # sex = feature[:, 12]
    position = numpy.load(position_file)
    AF_confusion = numpy.load(detail_file)
    rate_result = []
    for i in range(position.shape[0]):
        rate_result.append(get_evaluation_rate(position[i], AF_confusion[i]))

    print()


def analysis_bank(age, Position):
    """

    :return:
    """
    # 保护属性分组
    Rate_Results = []
    Name_Results = []
    age_start = [10, 30, 60]
    age_end = [30, 60, 100]
    for a_i in range(3):
        Group_Position = numpy_range_from_start_to_end(age, age_start[a_i] / (95 - 18), age_end[a_i] / (95 - 18))
        if numpy.sum(Group_Position) == 0:
            continue
        Group_Bias = numpy.logical_and(Group_Position, Position)
        print(numpy.sum(Group_Position))
        Group_Bias_Rate = numpy.sum(Group_Bias) / numpy.sum(Group_Position)
        Rate_Results.append(Group_Bias_Rate)
        Name_Results.append("age:{}".format(age_start[a_i]))
    a, b = order_diff_data(Rate_Results, Name_Results)
    return order_diff_data(Rate_Results, Name_Results)


def analysis_bank_result(test_file, position_file, detail_file):
    """

    :return:
    """
    test_data = numpy.load(test_file)
    feature, label = numpy.split(test_data, [-1, ], axis=1)
    age = feature[:, 0]

    position = numpy.load(position_file)
    AF_confusion = numpy.load(detail_file)
    TF_Position = ~position
    FB_Position = position
    TB_Position = AF_confusion[0]
    FF_Position = AF_confusion[1]
    FB_Position = AF_confusion[2]

    analysis_bank(age, TF_Position)
    analysis_bank(age, FB_Position)

# test_file = "../dataset/bank/test/test_seeds.npz.npy"
# position_file = "../dataset/bank/test/Test_RobustFair_BL_FB_P.npz.npy"
# detail_file = "../dataset/bank/test/Test_RobustFair_BL_FB_D.npz.npy"
# analysis_bank_result(test_file, position_file, detail_file)
