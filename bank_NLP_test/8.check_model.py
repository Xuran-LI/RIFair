import numpy
import pandas
import xlsxwriter
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils.utils_RIFair import split_NLP_model_by_embedding_layer
from utils.utils_Transform_AutoInt import TokenAndPositionEmbeddingBank, TransformerBlock
from utils.utils_evaluate import NLP_adversarial_analysis, NLP_model_performance, check_group_fairness, \
    calculate_average_perturbation_effect

from utils.utils_input_output import write_worksheet_header, write_worksheet_2d_data, get_search_times, \
    get_search_labels, get_search_NLP_instances


def get_original_dataset(dataset, protect_name, test_model):
    """
    获取测试集数据
    :return:
    """
    model_file = "../dataset/NLP/{}/model/{}.h5".format(dataset, test_model)
    custom_layers = {'TokenAndPositionEmbeddingBank': TokenAndPositionEmbeddingBank,
                     'TransformerBlock': TransformerBlock}
    model = load_model(model_file, custom_objects=custom_layers)

    data_file = ["../dataset/NLP/{}/data/code_text_test_bank.npy".format(dataset),
                 "../dataset/NLP/{}/data/text_test_label.npy".format(dataset)]
    input_data = numpy.load(data_file[0]).astype(float)
    label_data = numpy.load(data_file[1]).astype(float).reshape(-1, 1)
    pre_data = numpy.argmax(model.predict(input_data), axis=1).reshape(-1, 1)

    sim_file = ["../dataset/NLP/{}/data/code_text_{}_test_data.npy".format(dataset, protect_name)]
    sim_input = numpy.load(sim_file[0]).astype(float)
    sim_data = []
    sim_pre = []
    for j in range(sim_input.shape[0]):
        sim_data.append(sim_input[j])
        sim_pre.append(numpy.argmax(model.predict(sim_input[j]), axis=1).reshape(-1, 1))

    return input_data, label_data, pre_data, sim_data, sim_pre


def check_model_performance(dataset, test_models, result_file):
    """
    检查模型在测试集上的表现
    :return:
    """
    age_header = ["Acc", "Age IF", "Age TFR", "Age TBR", "Age FFR", "Age FBR", 'F_recall', 'F_precision', 'F_F1', "SUM",
                  "SPD", "EOD", "ACCD"]

    age_result = []

    for i in range(len(test_models)):
        data = pandas.read_csv("../dataset/NLP/bank/data/test_data.csv").values
        age_data = get_original_dataset(dataset, "age", test_models[i])
        protected_index = data[:, 0].reshape(-1, 1)
        group_fairness = check_group_fairness(protected_index, "youth", age_data[2], age_data[1])
        acc_if_af = NLP_model_performance(age_data)
        age_result.append(acc_if_af + group_fairness[0])

    workbook_name = xlsxwriter.Workbook(result_file)
    worksheet = workbook_name.add_worksheet("Age")
    write_worksheet_header(age_header, worksheet)
    write_worksheet_2d_data(age_result, worksheet)

    workbook_name.close()


def get_adversarial_result(dataset, protect_name, test_model, attack_model):
    """
    获取对抗生成结果数据
    :return:
    """
    model_file = "../dataset/NLP/{}/model/{}.h5".format(dataset, test_model)
    custom_layers = {'TokenAndPositionEmbeddingBank': TokenAndPositionEmbeddingBank,
                     'TransformerBlock': TransformerBlock}
    model = load_model(model_file, custom_objects=custom_layers)

    data1 = None
    data2 = None
    label = None
    times = None
    # 获取 TB FF FB 搜索结果
    for name1 in ["TB", "FF", "FB"]:
        generate_files = ["../dataset/NLP/{}/adv/{}_{}_{}_d.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/NLP/{}/adv/{}_{}_{}_s.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/NLP/{}/adv/{}_{}_{}_y.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/NLP/{}/adv/{}_{}_{}_t.txt".format(dataset, name1, protect_name, attack_model)]

        adv_data = get_search_NLP_instances(generate_files[0])
        sim_data = get_search_NLP_instances(generate_files[1])
        adv_label = get_search_labels(generate_files[2]).reshape(-1, 1)
        search_times = get_search_times(generate_files[3]).reshape(-1, 1)

        if label is None:
            data1 = adv_data
            data2 = sim_data
            label = adv_label
            times = search_times
        else:
            data1 = numpy.concatenate((data1, adv_data), axis=0)
            data2 = numpy.concatenate((data2, sim_data), axis=0)
            label = numpy.concatenate((label, adv_label), axis=0)
            times = numpy.concatenate((times, search_times), axis=0)

    for i in range(1, len(times)):
        times[i] += times[i - 1]
    # 计算原始数据位置，使用numpy.delete 删除
    delete_index = []
    for j in range(len(times)):
        if j == 0:
            delete_index.append(0)
        else:
            delete_index.append(times[j - 1][0])

    adv_input = numpy.delete(data1, delete_index, axis=0)
    adv_pre = numpy.argmax(model.predict(adv_input), axis=1).reshape(-1, 1)

    sim_input = [numpy.delete(data2, delete_index, axis=0)]
    sim_pre = [numpy.argmax(model.predict(numpy.delete(data2, delete_index, axis=0)), axis=1).reshape(-1, 1)]

    adv_label = numpy.delete(label, delete_index, axis=0)
    return adv_input, adv_label, adv_pre, sim_input, sim_pre


def analysis_model_adversarial_result(dataset, test_models, attack_model, result_file):
    """
    分析对抗生成结果
    :return:
    """
    age_header = ["age TF", "age TB", "age FF", "age FB", "SUM"]

    age_result = []

    for i in range(len(test_models)):
        # 马氏距离生成结果
        age_adv = get_adversarial_result(dataset, "age", test_models[i], attack_model)
        age_analysis = NLP_adversarial_analysis(age_adv)
        age_result.append(age_analysis)

    workbook_name = xlsxwriter.Workbook(result_file)
    worksheet = workbook_name.add_worksheet("age")
    write_worksheet_header(age_header, worksheet)
    write_worksheet_2d_data(age_result, worksheet)

    workbook_name.close()


def get_perturbation_and_predication(dataset, protect_name, test_model, attack_model):
    """
    获取embedding层模型输出结果，整个模型的输出结果
    :return:
    """
    model_file = "../dataset/NLP/{}/model/{}.h5".format(dataset, test_model)
    custom_layers = {'TokenAndPositionEmbeddingBank': TokenAndPositionEmbeddingBank,
                     'TransformerBlock': TransformerBlock}
    model = load_model(model_file, custom_objects=custom_layers)
    models = split_NLP_model_by_embedding_layer(model, 90, 32)

    data1 = None
    data2 = None
    label = None
    times = None
    for name1 in ["TB", "FF", "FB"]:
        generate_files = ["../dataset/NLP/{}/adv/{}_{}_{}_d.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/NLP/{}/adv/{}_{}_{}_s.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/NLP/{}/adv/{}_{}_{}_y.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/NLP/{}/adv/{}_{}_{}_t.txt".format(dataset, name1, protect_name, attack_model)]

        adv_index = get_search_NLP_instances(generate_files[0])
        sim_index = get_search_NLP_instances(generate_files[1])
        adv_label = get_search_labels(generate_files[2]).reshape(-1, 1)
        search_times = get_search_times(generate_files[3]).reshape(-1, 1)

        if label is None:
            data1 = adv_index
            data2 = sim_index
            label = adv_label
            times = search_times
        else:
            data1 = numpy.concatenate((data1, adv_index), axis=0)
            data2 = numpy.concatenate((data2, sim_index), axis=0)
            label = numpy.concatenate((label, adv_label), axis=0)
            times = numpy.concatenate((times, search_times), axis=0)

    embedding_output0 = models[1].predict(data1)
    embedding_output1 = models[1].predict(data2)

    model_predication0 = models[0].predict(data1)
    model_predication1 = models[0].predict(data2)

    for i in range(1, len(times)):
        times[i] += times[i - 1]

    label = to_categorical(label, num_classes=2)

    return embedding_output0, embedding_output1, model_predication0, model_predication1, label, times


def analysis_perturbation_effect(dataset, test_models, attack_model, result_file):
    """
    分析扰动对样本及其相似样本输出结果的影响
    :return:
    """
    age_header = ["age TF", "age TB", "age FF", "age FB", "SUM"]

    age_result = []

    for i in range(len(test_models)):
        age_data = get_perturbation_and_predication(dataset, "age", test_models[i], attack_model)
        age_effect = calculate_average_perturbation_effect(age_data)
        age_result.append(age_effect)

    workbook_name = xlsxwriter.Workbook(result_file)
    worksheet = workbook_name.add_worksheet("age")
    write_worksheet_header(age_header, worksheet)
    write_worksheet_2d_data(age_result, worksheet)

    workbook_name.close()


def get_social_impact_dataset(dataset, search_type, protect, test_model):
    """
    获取测试集数据
    :return:
    """
    model_file = "../dataset/NLP/{}/model/{}.h5".format(dataset, test_model)
    custom_layers = {'TokenAndPositionEmbeddingBank': TokenAndPositionEmbeddingBank,
                     'TransformerBlock': TransformerBlock}
    model = load_model(model_file, custom_objects=custom_layers)

    data_file = [
        "../dataset/NLP/{}/manipulate/SocialImpact_{}_{}_{}_d.npy".format(dataset, search_type, protect, test_model),
        "../dataset/NLP/{}/manipulate/SocialImpact_{}_{}_{}_s.npy".format(dataset, search_type, protect, test_model),
        "../dataset/NLP/{}/manipulate/SocialImpact_{}_{}_{}_y.npy".format(dataset, search_type, protect, test_model)]

    input_data = numpy.load(data_file[0]).astype(float)
    sim_input = numpy.load(data_file[1]).astype(float)
    label_data = numpy.load(data_file[2]).astype(float).reshape(-1, 1)
    pre_data = numpy.argmax(model.predict(input_data), axis=1).reshape(-1, 1)

    sim_data = []
    sim_pre = []
    for j in range(1):
        sim_data.append(sim_input)
        sim_pre.append(numpy.argmax(model.predict(sim_input), axis=1).reshape(-1, 1))

    return input_data, label_data, pre_data, sim_data, sim_pre


def get_social_impact_result(dataset, test_models):
    """
    分析RAFair对社会的影响，包括影响测试集的准确性、个体公平性、操控不同群体间的acc差异
    :return:
    """
    age_header = ["Acc", "age IF", "age TFR", "age TBR", "age FFR", "age FBR", 'F_recall', 'F_precision',
                  'F_F1', "SUM", "SPD", "EOD", "ACCD"]

    age_result = []

    for i in range(len(test_models)):
        for search_type in ["TB", "FF", "FB"]:
            data = pandas.read_csv("../dataset/NLP/bank/data/test_data.csv").values
            age_data = get_social_impact_dataset(dataset, search_type, "age", test_models[i])
            protected_index = data[:, 0].reshape(-1, 1)
            group_fairness = check_group_fairness(protected_index, "youth", age_data[2], age_data[1])
            acc_if_af = NLP_model_performance(age_data)
            age_result.append(acc_if_af + group_fairness[0])

    workbook_name = xlsxwriter.Workbook(result_file)
    worksheet = workbook_name.add_worksheet("age")
    write_worksheet_header(age_header, worksheet)
    write_worksheet_2d_data(age_result, worksheet)

    workbook_name.close()


if __name__ == "__main__":
    # # BL 模型在原始测试集和生成测试集上的表现
    dataset_name = "bank"
    model_names = ["Transformer_BL0"]
    result_file = "../dataset/NLP/bank/evaluation/Train_BL_evaluation_original.xlsx"
    check_model_performance(dataset_name, model_names, result_file)

    test_models = ["Transformer_BL0"]
    gen_model = "Transformer_BL0"
    result_file = "../dataset/NLP/bank/evaluation/Train_BL_evaluation_generate.xlsx"
    analysis_model_adversarial_result(dataset_name, test_models, gen_model, result_file)

    test_models = ["Transformer_BL0"]
    gen_model = "Transformer_BL0"
    result_file = "../dataset/NLP/bank/evaluation/Train_BL_evaluation_bug_analysis.xlsx"
    analysis_perturbation_effect(dataset_name, test_models, gen_model, result_file)

    # Retrain model 在原始测试集和生成测试集上的表现
    model_names = ["Re_age_Transformer_BL0"]
    result_file = "../dataset/NLP/bank/evaluation/ReTrain_BL_evaluation_original.xlsx"
    check_model_performance(dataset_name, model_names, result_file)

    test_models = ["Re_age_Transformer_BL0"]
    gen_model = "Transformer_BL0"
    result_file = "../dataset/NLP/bank/evaluation/ReTrain_BL_evaluation_generate.xlsx"
    analysis_model_adversarial_result(dataset_name, test_models, gen_model, result_file)

    test_models = ["Re_age_Transformer_BL0"]
    gen_model = "Transformer_BL0"
    result_file = "../dataset/NLP/bank/evaluation/ReTrain_BL_evaluation_bug_analysis.xlsx"
    analysis_perturbation_effect(dataset_name, test_models, gen_model, result_file)
