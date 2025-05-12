import keras
import numpy
import pandas
import xlsxwriter
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils.utils_RIFair import split_model_by_embedding_layer
from utils.utils_Transform_AutoInt import BankEmbedding, AutoIntTransformerBlock
from utils.utils_evaluate import model_performance, adversarial_analysis, \
    calculate_average_mahalanobis_distance, calculate_average_perturbation_effect, get_privilege_encode, \
    check_group_fairness, get_first_true_item, check_items_AF
from utils.utils_input_output import write_worksheet_header, write_worksheet_2d_data, get_search_instances, \
    get_search_labels, get_search_times


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
    for j in range(sim_index.shape[0]):
        S_input.append([sim_index[j, :, :], sim_value[j, :, :]])
        S_pre.append(numpy.argmax(model.predict([sim_index[j, :, :], sim_value[j, :, :]]), axis=1).reshape(-1, 1))

    return D_input, label, pre, S_input, S_pre


def check_model_performance(dataset, test_models, result_file):
    """
    检查模型在测试集上的表现
    :return:
    """
    age_header = ["Acc", "age IF", "age TFR", "age TBR", "age FFR", "age FBR", 'F_recall', 'F_precision',
                  'F_F1', "SUM", "SPD", "EOD", "ACCD"]
    age_result = []

    for i in range(len(test_models)):
        age_data = get_original_dataset(dataset, "age", test_models[i])
        privilege_value = get_privilege_encode(dataset, 0, "youth")
        protected_index = age_data[0][0][:, 0].reshape(-1, 1)
        group_fairness = check_group_fairness(protected_index, privilege_value, age_data[2], age_data[1])
        acc_if_af = model_performance(age_data)
        age_result.append(acc_if_af + group_fairness[0])

    workbook_name = xlsxwriter.Workbook(result_file)
    worksheet = workbook_name.add_worksheet("age")
    write_worksheet_header(age_header, worksheet)
    write_worksheet_2d_data(age_result, worksheet)
    workbook_name.close()


def get_adversarial_result(dataset, protect_name, test_model, attack_model):
    """
    获取对抗生成结果数据
    :return:
    """
    model_file = "../dataset/{}/model/{}.h5".format(dataset, test_model)
    custom_layers = {'BankEmbedding': BankEmbedding, 'AutoIntTransformerBlock': AutoIntTransformerBlock}
    model = load_model(model_file, custom_objects=custom_layers)

    index1 = None
    value1 = None
    index2 = None
    value2 = None
    label = None
    times = None
    # 获取 TB FF FB 搜索结果
    for name1 in ["TB", "FF", "FB"]:
        generate_files = ["../dataset/{}/adv/{}_{}_{}_i.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/{}/adv/{}_{}_{}_s.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/{}/adv/{}_{}_{}_y.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/{}/adv/{}_{}_{}_t.txt".format(dataset, name1, protect_name, attack_model)]

        adv_label = get_search_labels(generate_files[2]).reshape(-1, 1)
        adv_index, adv_value = get_search_instances(generate_files[0])
        sim_index, sim_value = get_search_instances(generate_files[1])
        search_times = get_search_times(generate_files[3]).reshape(-1, 1)

        if label is None:
            index1 = adv_index
            value1 = adv_value
            index2 = sim_index
            value2 = sim_value
            label = adv_label
            times = search_times
        else:
            index1 = numpy.concatenate((index1, adv_index), axis=0)
            value1 = numpy.concatenate((value1, adv_value), axis=0)
            index2 = numpy.concatenate((index2, sim_index), axis=0)
            value2 = numpy.concatenate((value2, sim_value), axis=0)
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

    adv_index1 = numpy.delete(index1, delete_index, axis=0)
    adv_value1 = numpy.delete(value1, delete_index, axis=0)
    adv_input = [adv_index1, adv_value1]
    adv_pre = numpy.argmax(model.predict(adv_input), axis=1).reshape(-1, 1)

    sim_index2 = numpy.delete(index2, delete_index, axis=0)
    sim_value2 = numpy.delete(value2, delete_index, axis=0)
    sim_input = [[sim_index2, sim_value2]]
    sim_pre = [numpy.argmax(model.predict([sim_index2, sim_value2]), axis=1).reshape(-1, 1)]

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
        age_analysis = adversarial_analysis(age_adv)
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
    model_file = "../dataset/{}/model/{}.h5".format(dataset, test_model)
    custom_layers = {'BankEmbedding': BankEmbedding, 'AutoIntTransformerBlock': AutoIntTransformerBlock}
    model = load_model(model_file, custom_objects=custom_layers)
    models = split_model_by_embedding_layer(model, 16, 32)

    index1 = None
    value1 = None
    index2 = None
    value2 = None
    label = None
    times = None
    for name1 in ["TB", "FF", "FB"]:
        generate_files = ["../dataset/{}/adv/{}_{}_{}_i.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/{}/adv/{}_{}_{}_s.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/{}/adv/{}_{}_{}_y.txt".format(dataset, name1, protect_name, attack_model),
                          "../dataset/{}/adv/{}_{}_{}_t.txt".format(dataset, name1, protect_name, attack_model)]

        adv_label = get_search_labels(generate_files[2]).reshape(-1, 1)
        adv_index, adv_value = get_search_instances(generate_files[0])
        sim_index, sim_value = get_search_instances(generate_files[1])
        search_times = get_search_times(generate_files[3]).reshape(-1, 1)

        if label is None:
            index1 = adv_index
            value1 = adv_value
            index2 = sim_index
            value2 = sim_value
            label = adv_label
            times = search_times
        else:
            index1 = numpy.concatenate((index1, adv_index), axis=0)
            value1 = numpy.concatenate((value1, adv_value), axis=0)
            index2 = numpy.concatenate((index2, sim_index), axis=0)
            value2 = numpy.concatenate((value2, sim_value), axis=0)
            label = numpy.concatenate((label, adv_label), axis=0)
            times = numpy.concatenate((times, search_times), axis=0)

    embedding_output0 = models[1].predict([index1, value1])
    embedding_output1 = models[1].predict([index2, value2])

    model_predication0 = models[0].predict([index1, value1])
    model_predication1 = models[0].predict([index2, value2])

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


if __name__ == "__main__":
    # BL 模型在原始测试集和生成测试集上的表现
    dataset_name = "bank"
    # model_names = ["AutoInt_BL0"]
    # result_file = "../dataset/bank/evaluation/Train_BL_evaluation_original.xlsx"
    # check_model_performance(dataset_name, model_names, result_file)
    #
    # test_models = ["AutoInt_BL0"]
    # gen_model = "AutoInt_BL0"
    # result_file = "../dataset/bank/evaluation/Train_BL_evaluation_generate.xlsx"
    # analysis_model_adversarial_result(dataset_name, test_models, gen_model, result_file)

    test_models = ["AutoInt_BL0"]
    gen_model = "AutoInt_BL0"
    result_file = "../dataset/bank/evaluation/Train_BL_evaluation_bug_analysis.xlsx"
    analysis_perturbation_effect(dataset_name, test_models, gen_model, result_file)

    # Retrain model 在原始测试集和生成测试集上的表现
    # model_names = ["Re_age_AutoInt_BL0"]
    # result_file = "../dataset/bank/evaluation/ReTrain_BL_evaluation_original.xlsx"
    # check_model_performance(dataset_name, model_names, result_file)
    #
    # test_models = ["Re_age_AutoInt_BL0"]
    # gen_model = "AutoInt_BL0"
    # result_file = "../dataset/bank/evaluation/ReTrain_BL_evaluation_generate.xlsx"
    # analysis_model_adversarial_result(dataset_name, test_models, gen_model, result_file)

    test_models = ["Re_age_AutoInt_BL0"]
    gen_model = "AutoInt_BL0"
    result_file = "../dataset/bank/evaluation/ReTrain_BL_evaluation_bug_analysis.xlsx"
    analysis_perturbation_effect(dataset_name, test_models, gen_model, result_file)
