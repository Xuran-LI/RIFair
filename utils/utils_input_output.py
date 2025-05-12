import pandas
import numpy
from tensorflow.python.keras.utils.np_utils import to_categorical


def adjust_ctrip_data(test_file):
    """
    获取ctrip的numerical数据
    :return:
    """
    data = pandas.read_csv(test_file, header=None).values
    hotel, user_service = numpy.split(data, [6, ], axis=1)
    return numpy.concatenate((user_service, hotel), axis=1)


def write_worksheet_2d_data(data, worksheet):
    """
    输出2维数据至worksheet
    :return:
    """
    for i in range(len(data)):
        for j in range(len(data[i])):
            worksheet.write(i + 1, j, data[i][j])


def write_worksheet_header(headers, worksheet):
    """
    输出header至worksheet
    :return:
    """
    for i in range(len(headers)):
        worksheet.write(0, i, headers[i])


def write_predication(output_file, predications):
    """

    :return:
    """
    for i in range(len(predications)):
        for j in range(len(predications[i])):
            output_file.write(",".join('%.4f' % float(d) for d in predications[i][j][0]) + ";")
        output_file.write("|")
    output_file.write("\n")


def write_NLP(output_file, NLPs):
    """
    输出NLP结果
    :return:
    """
    for i in range(len(NLPs)):
        for j in range(len(NLPs[i])):
            output_file.write(",".join('{}'.format(d) for d in NLPs[i][j]) + ";")
        output_file.write("|")
    output_file.write("\n")


def write_attentions(output_file, attentions):
    """
    输出attention权重
    :return:
    """
    for i in range(len(attentions)):
        for j in range(len(attentions[i])):
            att = attentions[i][j][0]
            for h in range(att.shape[0]):
                output_file.write(",".join('%.4f' % float(d) for d in att[h]) + "$")
            output_file.write(";")
        output_file.write("|")
    output_file.write("\n")


def write_search_instances(output_file, adv_instance):
    """
    将扰动结果的index、value输出至文件中
    :param output_file:
    :param adv_instance:
    :return:
    """
    for i in range(len(adv_instance)):
        output_file.write(",".join('%.f' % float(d) for d in adv_instance[i][0]) + ";")
        output_file.write(",".join('%.f' % float(d) for d in adv_instance[i][1]) + "\n")


def get_search_instances(file):
    """

    :return:
    """
    result1 = []
    result2 = []
    with open(file, "r") as input_file:
        for data in input_file:
            index, value = data.strip("\n").split(";")
            result1.append(index.split(","))
            result2.append(value.split(","))
    input_file.close()
    return numpy.array(result1).astype(float), numpy.array(result2).astype(float)


def write_search_NLP_instances(output_file, adv_instance):
    """
    将扰动结果的index、value输出至文件中
    :param output_file:
    :param adv_instance:
    :return:
    """
    for i in range(len(adv_instance)):
        # output_file.write(",".join('%.f' % float(d) for d in adv_instance[i][0]) + ";")
        output_file.write(",".join('%.f' % float(d) for d in adv_instance[i]) + "\n")


def get_search_NLP_instances(file):
    """

    :return:
    """
    result1 = []
    with open(file, "r") as input_file:
        for data in input_file:
            result1.append(data.strip("\n").split(","))
    input_file.close()
    return numpy.array(result1).astype(float)


def write_search_labels(output_file, adv_label):
    """
    将扰动结果的标签输出至文件中
    :param output_file:
    :param adv_label:
    :return:
    """
    for i in range(len(adv_label)):
        output_file.write(",".join('%.f' % float(d) for d in adv_label[i]) + "\n")


def get_search_labels(file):
    """

    :return:
    """
    result = []
    with open(file, "r") as input_file:
        for data in input_file:
            result.append(data.strip("\n"))
    input_file.close()
    return numpy.array(result).astype(float).reshape(1, -1)


def write_search_times(output_file, adv_times):
    """
    将扰动结果的标签输出至文件中
    :param output_file:
    :param adv_times:
    :return:
    """
    output_file.write("{}\n".format(adv_times))


def get_search_times(file):
    """

    :return:
    """
    result = []
    with open(file, "r") as input_file:
        for data in input_file:
            result.append(data.strip("\n"))
    input_file.close()
    return numpy.array(result).astype(int).reshape(1, -1)


def get_search_result(index, value, label, sim_index, sim_value, search_times):
    """
    获取对对抗攻击结果
    :return:
    """
    result_index = []
    result_value = []
    result_label = []
    result_sim_index = []
    result_sim_value = []

    for i in range(1, len(search_times)):
        search_times[i] += search_times[i - 1]
    for i in range(search_times.shape[0]):
        if i == 0:
            start_index = 0
            end_index = search_times[i][0]
        else:
            start_index = search_times[i - 1][0]
            end_index = search_times[i][0]
        for ii in range(start_index + 1, end_index):
            result_index.append(index[ii])
            result_value.append(value[ii])
            result_label.append(label[ii])
            result_sim_index.append(sim_index[ii])
            result_sim_value.append(sim_value[ii])
    return result_index, result_value, result_label, result_sim_index, result_sim_value


def get_test_data(dataset):
    """
    获取验证集合
    :return:
    """
    index = numpy.load("../dataset/{}/data/test_i.npy".format(dataset), allow_pickle=True)
    value = numpy.load("../dataset/{}/data/test_v.npy".format(dataset), allow_pickle=True)
    label = numpy.load("../dataset/{}/data/test_y.npy".format(dataset), allow_pickle=True).reshape(-1, 1)
    label = to_categorical(label, num_classes=2)

    return index, value, label


def write_min_max(output_file, min_max):
    """
    将value取值的最小值、最大值输出至文件中
    :return:
    """
    for i in range(len(min_max)):
        output_file.write(",".join('%.f' % float(d) for d in min_max[i]) + "\n")


def get_min_max(file):
    """
    获取value取值的最小值、最大值
    :param file:
    :return:
    """
    min_list = []
    max_list = []
    with open(file, "r") as input_file:
        for data in input_file:
            min_max = data.strip("\n").split(",")
            min_list.append(int(min_max[0]))
            max_list.append(int(min_max[1]))
    input_file.close()
    return min_list, max_list


def get_NLP_test_data(dataset):
    """
    获取验证集合
    :return:
    """
    data = numpy.load("../dataset/NLP/{}/data/code_text_train_{}.npy".format(dataset, dataset), allow_pickle=True)
    label = numpy.load("../dataset/NLP/{}/data/text_train_label.npy".format(dataset), allow_pickle=True).reshape(-1, 1)
    label = to_categorical(label, num_classes=2)

    return data, label


def get_ACS_NLP_test_data(dataset):
    """
    获取验证集合
    :return:
    """
    data = numpy.load("../dataset/ACS/NLP/{}/data/code_text_train_{}.npy".format(dataset, dataset), allow_pickle=True)
    label = numpy.load("../dataset/ACS/NLP/{}/data/text_train_label.npy".format(dataset), allow_pickle=True).reshape(-1, 1)
    label = to_categorical(label, num_classes=2)

    return data, label
# def get_item_label_by_condition(conditions, items, labels):
#     """
#     根据状态获取对应的item label
#     :return:
#     """
#     result_items = []
#     result_labels = []
#     for i in range(conditions.shape[0]):
#         if conditions[i]:
#             result_items.append(items[i])
#             result_labels.append(labels[i])
#     if len(result_labels) < 1:
#         return numpy.empty((1, items.shape[1])).astype(float), numpy.empty((1, labels.shape[1])).astype(float)
#     else:
#         return numpy.array(result_items), numpy.array(result_labels)
#
#
# def get_similar_data_by_condition(conditions, similar_data):
#     """
#     根据状态获取对应的item label
#     :return:
#     """
#     result_items = []
#     for i in range(conditions.shape[0]):
#         if conditions[i]:
#             result_items.append(similar_data[i])
#     return numpy.array(result_items)
#
#
# # 去重数据
# def unique_data(data, similar_data):
#     """
#     去重数据
#     :return:
#     """
#     unique, unique_index = numpy.unique(data, axis=0, return_index=True)
#     unique_similar = similar_data[unique_index]
#     return unique, unique_similar
#
#
# def unique_search_data(data1, similar_data1, data2, similar_data2, file1, file2):
#     """
#     将本轮搜索结果相对于之前搜索结果进行去重
#     :return:
#     """
#     pass_len = data1.shape[0]
#
#     generated_data = numpy.concatenate((data1, data2), axis=0)
#     generated_data_similar = numpy.concatenate((similar_data1, similar_data2), axis=0)
#
#     unique, unique_index = numpy.unique(generated_data, axis=0, return_index=True)
#     unique_similar = generated_data_similar[unique_index, :, :]
#
#     data2 = []
#     similar_data2 = []
#     for u_i in unique_index:
#         if u_i >= pass_len:
#             data2.append(generated_data[u_i])
#             similar_data2.append(generated_data_similar[u_i])
#
#     numpy.save(file1, data2)
#     numpy.save(file2, similar_data2)
#     return numpy.array(unique), numpy.array(unique_similar)
#
#
# # 合并多个list
# def combine_lists(list1_0, list1_1, list1_2, list2_0, list2_1, list2_2):
#     """
#     链接多个list
#     :return:
#     """
#     return list1_0 + list2_0, list1_1 + list2_1, list1_2 + list2_2
