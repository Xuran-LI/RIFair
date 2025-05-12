import keras
import numpy
import pandas
import xlsxwriter
from tensorflow.python.keras.models import load_model

from utils.utils_Transform_AutoInt import BankEmbedding, AutoIntTransformerBlock
from utils.utils_evaluate import check_items_AF, get_first_true_item, check_group_fairness, model_performance
from utils.utils_input_output import get_search_instances, get_search_labels, get_search_times, write_worksheet_header, \
    write_worksheet_2d_data


def get_equal_position(item1, numpy1, item2, numpy2, start_position):
    """
    获取元素在numpy中相等元素的位置
    :return:
    """
    for i in range(start_position, numpy1.shape[0]):
        if numpy.all(item1 == numpy1[i, :]) and numpy.all(item2 == numpy2[i, :]):
            return i


def get_social_impact(search_type, dataset, protect, model_name):
    """
    各个攻击方法对准确性、公平性造成的社会影响
    :return:
    """
    # 模型
    M_file = "../dataset/{}/model/{}.h5".format(dataset, model_name)
    custom_layers = {'BankEmbedding': BankEmbedding, 'AutoIntTransformerBlock': AutoIntTransformerBlock}
    model = keras.models.load_model(M_file, custom_objects=custom_layers)
    # # 测试集
    data_file = ["../dataset/{}/data/test_i.npy".format(dataset),
                 "../dataset/{}/data/test_v.npy".format(dataset),
                 "../dataset/{}/data/test_y.npy".format(dataset)]
    test_index = numpy.load(data_file[0])
    test_value = numpy.load(data_file[1])
    test_label = numpy.load(data_file[2]).reshape(-1, 1)

    # 敏感属性测试集
    sim_file = ["../dataset/bank/data/{}_test_i.npy".format(protect),
                "../dataset/bank/data/{}_test_v.npy".format(protect),
                "../dataset/bank/data/{}_test_y.npy".format(protect)]
    sim_index = numpy.load(sim_file[0])
    sim_value = numpy.load(sim_file[1])
    sim_label = numpy.load(sim_file[2])

    # 操纵结果文件
    manipulate_file = [
        "../dataset/{}/manipulate/SocialImpact_{}_{}_{}_i.npy".format(dataset, search_type, protect, model_name),
        "../dataset/{}/manipulate/SocialImpact_{}_{}_{}_v.npy".format(dataset, search_type, protect, model_name),
        "../dataset/{}/manipulate/SocialImpact_{}_{}_{}_i_s.npy".format(dataset, search_type, protect, model_name),
        "../dataset/{}/manipulate/SocialImpact_{}_{}_{}_v_s.npy".format(dataset, search_type, protect, model_name),
        "../dataset/{}/manipulate/SocialImpact_{}_{}_{}_y.npy".format(dataset, search_type, protect, model_name)]

    generate_files = ["../dataset/{}/adv/{}_{}_{}_i.txt".format(dataset, search_type, protect, model_name),
                      "../dataset/{}/adv/{}_{}_{}_s.txt".format(dataset, search_type, protect, model_name),
                      "../dataset/{}/adv/{}_{}_{}_y.txt".format(dataset, search_type, protect, model_name),
                      "../dataset/{}/adv/{}_{}_{}_t.txt".format(dataset, search_type, protect, model_name)]

    adv_index, adv_value = get_search_instances(generate_files[0])
    adv_index_sim, adv_value_sim = get_search_instances(generate_files[1])
    adv_label = get_search_labels(generate_files[2]).reshape(-1, 1)
    test_time = get_search_times(generate_files[3]).reshape(-1, 1)

    adv_pre = model.predict([adv_index, adv_value])
    adv_pre_sim = model.predict([adv_index_sim, adv_value_sim])

    replace = 0
    manipulate_index = test_index
    manipulate_value = test_value
    manipulate_index_sim = sim_index[0].copy()
    manipulate_value_sim = sim_value[0].copy()

    equal_i = 0
    for i in range(1, len(test_time)):
        test_time[i] += test_time[i - 1]
    # 分析每轮扰动时样本的扰动信息
    for i in range(test_time.shape[0]):
        if i == 0:
            start_index = 0
            end_index = test_time[i][0]
        else:
            start_index = test_time[i - 1][0]
            end_index = test_time[i][0]
        # 获取本轮扰动结果：扰动后样本、相似样本、标签、扰动后预测结果、扰动后准确公平性检测结果
        search_data = [adv_index[start_index:end_index], adv_value[start_index:end_index]]
        search_data_pre = adv_pre[start_index:end_index]
        search_data_sim = [adv_index_sim[start_index:end_index], adv_value_sim[start_index:end_index]]
        search_data_sim_pre = adv_pre_sim[start_index:end_index]
        search_label = adv_label[start_index:end_index]
        search_AF = check_items_AF(search_data, search_data_sim, search_data_pre, search_data_sim_pre, search_label)
        if numpy.sum(search_AF[1]) > 0 and search_type == "TB":  # 搜索结果True bias
            adv_id = get_first_true_item(search_AF[1])
            equal_i = get_equal_position(search_data[0][0, :], test_index, search_data[1][0, :], test_value, equal_i)
            manipulate_index[equal_i, :] = search_data[0][adv_id, :]
            manipulate_value[equal_i, :] = search_data[1][adv_id, :]
            manipulate_index_sim[equal_i, :] = search_data_sim[0][adv_id, :]
            manipulate_value_sim[equal_i, :] = search_data_sim[1][adv_id, :]
            replace += 1
            print("replace:{:.2f}".format(replace))
        elif numpy.sum(search_AF[2]) > 0 and search_type == "FF":  # 搜索结果false fair
            adv_id = get_first_true_item(search_AF[2])
            equal_i = get_equal_position(search_data[0][0, :], test_index, search_data[1][0, :], test_value, equal_i)
            manipulate_index[equal_i, :] = search_data[0][adv_id, :]
            manipulate_value[equal_i, :] = search_data[1][adv_id, :]
            manipulate_index_sim[equal_i, :] = search_data_sim[0][adv_id, :]
            manipulate_value_sim[equal_i, :] = search_data_sim[1][adv_id, :]
            replace += 1
            print("replace:{:.2f}".format(replace))
        elif numpy.sum(search_AF[3]) > 0 and search_type == "FB":  # 搜索结果false bias
            adv_id = get_first_true_item(search_AF[3])
            equal_i = get_equal_position(search_data[0][0, :], test_index, search_data[1][0, :], test_value, equal_i)
            manipulate_index[equal_i, :] = search_data[0][adv_id, :]
            manipulate_value[equal_i, :] = search_data[1][adv_id, :]
            manipulate_index_sim[equal_i, :] = search_data_sim[0][adv_id, :]
            manipulate_value_sim[equal_i, :] = search_data_sim[1][adv_id, :]
            replace += 1
            print("replace:{:.2f}".format(replace))
        else:
            # 搜索失败，将原相似样本更新
            equal_i = get_equal_position(search_data[0][0, :], test_index, search_data[1][0, :], test_value, equal_i)
            # manipulate_index[equal_i, :] = search_data[0][-1, :]
            # manipulate_value[equal_i, :] = search_data[1][-1, :]
            # manipulate_index_sim[equal_i, :] = search_data_sim[0][0, :]
            # manipulate_value_sim[equal_i, :] = search_data_sim[1][0, :]

    numpy.save(manipulate_file[0], manipulate_index)
    numpy.save(manipulate_file[1], manipulate_value)
    numpy.save(manipulate_file[2], manipulate_index_sim)
    numpy.save(manipulate_file[3], manipulate_value_sim)
    numpy.save(manipulate_file[4], test_label)


def get_social_impact_dataset(dataset, search_type, protect, test_model):
    """
    获取测试集数据
    :return:
    """
    model_file = "../dataset/{}/model/{}.h5".format(dataset, test_model)
    custom_layers = {'BankEmbedding': BankEmbedding, 'AutoIntTransformerBlock': AutoIntTransformerBlock}
    model = load_model(model_file, custom_objects=custom_layers)

    data_file = [
        "../dataset/{}/manipulate/SocialImpact_{}_{}_AutoInt_BL0_i.npy".format(dataset, search_type, protect),
        "../dataset/{}/manipulate/SocialImpact_{}_{}_AutoInt_BL0_v.npy".format(dataset, search_type, protect),
        "../dataset/{}/manipulate/SocialImpact_{}_{}_AutoInt_BL0_i_s.npy".format(dataset, search_type, protect),
        "../dataset/{}/manipulate/SocialImpact_{}_{}_AutoInt_BL0_v_s.npy".format(dataset, search_type, protect),
        "../dataset/{}/manipulate/SocialImpact_{}_{}_AutoInt_BL0_y.npy".format(dataset, search_type, protect)]

    index_data = numpy.load(data_file[0]).astype(float)
    value_data = numpy.load(data_file[1]).astype(float)
    index_data_sim = numpy.load(data_file[2]).astype(float)
    value_data_sim = numpy.load(data_file[3]).astype(float)
    label_data = numpy.load(data_file[4]).astype(float).reshape(-1, 1)
    pre_data = numpy.argmax(model.predict([index_data, value_data]), axis=1).reshape(-1, 1)

    sim_data = []
    sim_pre = []
    for j in range(1):
        sim_data.append([index_data_sim, value_data_sim])
        sim_pre.append(numpy.argmax(model.predict([index_data_sim, value_data_sim]), axis=1).reshape(-1, 1))

    return [index_data, value_data], label_data, pre_data, sim_data, sim_pre


def get_original_dataset(dataset, protect_name, test_model):
    """
    获取测试集数据
    :return:
    """
    model_file = "../dataset/{}/model/{}.h5".format(dataset, test_model)
    custom_layers = {'BankEmbedding': BankEmbedding, 'AutoIntTransformerBlock': AutoIntTransformerBlock}
    model = load_model(model_file, custom_objects=custom_layers)

    data_file = ["../dataset/{}/data/{}_i.npy".format(dataset, "test"),
                 "../dataset/{}/data/{}_v.npy".format(dataset, "test"),
                 "../dataset/{}/data/{}_y.npy".format(dataset, "test")]
    index = numpy.load(data_file[0]).astype(float)
    value = numpy.load(data_file[1]).astype(float)
    label = numpy.load(data_file[2]).astype(float).reshape(-1, 1)
    D_input = [index, value]
    pre = numpy.argmax(model.predict(D_input), axis=1).reshape(-1, 1)

    sim_file = ["../dataset/{}/data/{}_test_i.npy".format(dataset, protect_name),
                "../dataset/{}/data/{}_test_v.npy".format(dataset, protect_name),
                "../dataset/{}/data/{}_test_y.npy".format(dataset, protect_name)]
    sim_index = numpy.load(sim_file[0]).astype(float)
    sim_value = numpy.load(sim_file[1]).astype(float)
    sim_label = numpy.load(sim_file[2]).astype(float)
    S_input = []
    S_pre = []
    for j in range(1):
        S_input.append([sim_index[0, :, :], sim_value[0, :, :]])
        S_pre.append(numpy.argmax(model.predict([sim_index[0, :, :], sim_value[0, :, :]]), axis=1).reshape(-1, 1))

    return D_input, label, pre, S_input, S_pre


def get_social_impact_result(dataset, test_models, result_file):
    """
    分析RAFair对社会的影响，包括影响测试集的准确性、个体公平性、操控不同群体间的acc差异
    :return:
    """
    age_header = ["Acc", "age IF", "age TFR", "age TBR", "age FFR", "age FBR", 'F_recall', 'F_precision', 'F_F1', "SUM",
                  "SPD", "EOD", "ACCD"]
    age_result = []

    for i in range(len(test_models)):
        data = pandas.read_csv("../dataset/bank/data/test_data.csv").values
        age_data = get_original_dataset(dataset, "age", test_models[i])
        protected_index = data[:, 0].reshape(-1, 1)
        group_fairness = check_group_fairness(protected_index, "youth", age_data[2], age_data[1])
        acc_if_af = model_performance(age_data)
        age_result.append(acc_if_af + group_fairness[0])
        for search_type in ["TB", "FF", "FB"]:
            data = pandas.read_csv("../dataset/bank/data/test_data.csv").values
            age_data = get_social_impact_dataset(dataset, search_type, "age", test_models[i])
            protected_index = data[:, 0].reshape(-1, 1)
            group_fairness = check_group_fairness(protected_index, "youth", age_data[2], age_data[1])
            acc_if_af = model_performance(age_data)
            age_result.append(acc_if_af + group_fairness[0])

    workbook_name = xlsxwriter.Workbook(result_file)
    worksheet = workbook_name.add_worksheet("age")
    write_worksheet_header(age_header, worksheet)
    write_worksheet_2d_data(age_result, worksheet)

    workbook_name.close()


if __name__ == "__main__":
    # 各个攻击方法对准确性、公平性造成的社会影响
    get_social_impact("TB", "bank", "age", "AutoInt_BL0")
    get_social_impact("FF", "bank", "age", "AutoInt_BL0")
    get_social_impact("FB", "bank", "age", "AutoInt_BL0")

    # 检查模型对精度、个体公平性、ACCD的影响
    dataset_name = "bank"
    test_models = ["AutoInt_BL0"]
    result_file = "../dataset/bank/evaluation/Train_BL_evaluation_social_impact.xlsx"
    get_social_impact_result(dataset_name, test_models, result_file)

    test_models = ["Re_age_AutoInt_BL0"]
    result_file = "../dataset/bank/evaluation/ReTrain_BL_evaluation_social_impact.xlsx"
    get_social_impact_result(dataset_name, test_models, result_file)
