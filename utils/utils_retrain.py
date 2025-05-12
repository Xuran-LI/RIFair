import numpy
from tensorflow.python.keras.models import load_model

from utils.utils_evaluate import calculate_mse, check_dist, calculate_cos


def get_retrain_data_similar(train_file, generation_file1, generation_file2, percentage):
    """
    将生成样本及其相似样本作为重训练样本
    :return:
    """
    # 原始训练集
    data = numpy.load(train_file)
    data_x, data_y = numpy.split(data, [-1, ], axis=1)

    # 生成集合
    data1 = numpy.load(generation_file1)
    data1_x, data1_y = numpy.split(data1, [-1, ], axis=1)

    # 随机选择生成集合内数据
    data_index = numpy.arange(data1_x.shape[0])
    numpy.random.shuffle(data_index)
    retrain_num = round(percentage * data1_x.shape[0])
    select_index = data_index[:retrain_num]
    select_x = data1_x[select_index]
    select_y = data1_y[select_index]

    # 生成集合的相似集
    similar_data = numpy.load(generation_file2)
    data2_y = data1_y
    for j in select_index:
        i = numpy.random.randint(0, similar_data.shape[1] - 1)
        select_x = numpy.concatenate((select_x, similar_data[j, i, :].reshape(1, -1)), axis=0)
        select_y = numpy.concatenate((select_y, data2_y[j].reshape(1, -1)), axis=0)
        # for i in range(similar_data.shape[1]):
        #     select_x = numpy.concatenate((select_x, similar_data[j, i, :].reshape(1, -1)), axis=0)
        #     select_y = numpy.concatenate((select_y, data2_y[j].reshape(1, -1)), axis=0)

    # 生成重训练集合
    retrain_x = numpy.concatenate((data_x, select_x), axis=0)
    retrain_y = numpy.concatenate((data_y, select_y), axis=0)

    return retrain_x, retrain_y


def get_retrain_data1(train_file, generation_file, generation_similar_file, percentage):
    """
    将生成样本作为重训练样本
    :return:
    """
    train_data = numpy.load(train_file)
    retrain_x, retrain_y = numpy.split(train_data, [-1, ], axis=1)

    # 生成样本
    data = numpy.load(generation_file)
    data_x, data_y = numpy.split(data, [-1, ], axis=1)

    # 生成相似样本
    similar_data_x = numpy.load(generation_similar_file)

    # 随机选择生成集合内数据
    data_index = numpy.arange(data_x.shape[0])
    numpy.random.shuffle(data_index)
    retrain_num = round(percentage * data_x.shape[0])
    select_index = data_index[:retrain_num]
    select_x = data_x[select_index]
    select_y = data_y[select_index]
    select_similar_x = similar_data_x[select_index]

    retrain_x = numpy.concatenate((retrain_x, select_x), axis=0)
    retrain_y = numpy.concatenate((retrain_y, select_y), axis=0)

    for i in range(select_similar_x.shape[1]):
        retrain_x = numpy.concatenate((retrain_x, select_similar_x[:, i, :]), axis=0)
        retrain_y = numpy.concatenate((retrain_y, select_y), axis=0)

    return retrain_x, retrain_y


def get_retrain_data(generate_data, similar_data, retrain_num):
    """
    将生成样本作为重训练样本
    :return:
    """
    retrain_data = generate_data[similar_data]
    numpy.random.shuffle(retrain_data)
    if retrain_data.shape[0] > retrain_num:
        retrain_data, _ = numpy.split(retrain_data, [retrain_num], axis=0)
        retrain_x, retrain_y = numpy.split(retrain_data, [-1, ], axis=1)
    else:
        retrain_x, retrain_y = numpy.split(retrain_data, [-1, ], axis=1)

    return retrain_x, retrain_y
    # return select_x, select_y


# def get_retrain_data(generation_file, generation_similar_file, percentage):
#     """
#     将生成样本作为重训练样本
#     :return:
#     """
#     # 生成样本
#     data = numpy.load(generation_file)
#     data_x, data_y = numpy.split(data, [-1, ], axis=1)
#
#     # 生成相似样本
#     data1 = numpy.load(generation_similar_file)
#     data1_x, data1_y = numpy.split(data1, [-1, ], axis=1)
#
#     # 随机选择生成集合内数据
#     data_index = numpy.arange(data1_x.shape[0])
#     numpy.random.shuffle(data_index)
#     retrain_num = round(percentage * data1_x.shape[0])
#     select_index = data_index[:retrain_num]
#     select_x = data1_x[select_index]
#     select_y = data1_y[select_index]
#
#     # 生成重训练集合
#     retrain_x = numpy.concatenate((data_x, select_x), axis=0)
#     retrain_y = numpy.concatenate((data_y, select_y), axis=0)
#
#     return retrain_x, retrain_y


def get_retrain_acc_data(model_file, train_file, generation_file, chose_size, dist=0):
    """
    使用数据集中的 acc data进行重训练
    :return:
    """
    model = load_model(model_file)
    data1 = numpy.load(generation_file)
    x1, y1 = numpy.split(data1, [-1, ], axis=1)
    pre = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)
    MSE = calculate_mse(pre, y1)
    # 预测结果与真实标记MSE小于dist的比例
    acc_cond = check_dist(MSE, dist)
    acc_data = data1[acc_cond]
    numpy.random.shuffle(acc_data)
    chose_data1, chose_data2 = numpy.split(acc_data, [chose_size, ], axis=0)

    data = numpy.load(train_file)
    data_x, data_y = numpy.split(data, [-1, ], axis=1)
    acc_x, acc_y = numpy.split(chose_data1, [-1, ], axis=1)

    retrain_x = numpy.concatenate((data_x, acc_x), axis=0)
    retrain_y = numpy.concatenate((data_y, acc_y), axis=0)

    return retrain_x, retrain_y


def get_retrain_fair_data(model_file, train_file, test_file, similar_file, chose_size, K=0, dist=0):
    """
    使用数据集中的 fair data进行重训练
    :return:
    """
    model = load_model(model_file)
    data1 = numpy.load(test_file)

    x1, y1 = numpy.split(data1, [-1, ], axis=1)
    pre1 = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)

    data2 = numpy.squeeze(numpy.load(similar_file))
    x2 = []
    pre2 = []
    for j in range(data2.shape[1]):
        x2.append(data2[:, j, :])
        pre2.append(numpy.argmax(model.predict(data2[:, j, :]), axis=1).reshape(-1, 1))

    IF_cond = numpy.ones(pre1.shape[0])
    for i in range(len(pre2)):
        # D(f(x1),f(x2))<=Kd(x1,x2)
        D_distance = calculate_mse(pre1, pre2[i])
        Kd_distance = K * calculate_mse(x1, x2[i])
        IF_cond = numpy.logical_and(IF_cond, check_dist(D_distance - Kd_distance, dist))

    IF_data = data1[IF_cond]
    numpy.random.shuffle(IF_data)
    chose_data1, chose_data2 = numpy.split(IF_data, [chose_size, ], axis=0)

    data = numpy.load(train_file)
    data_x, data_y = numpy.split(data, [-1, ], axis=1)
    IF_x, IF_y = numpy.split(chose_data1, [-1, ], axis=1)

    retrain_x = numpy.concatenate((data_x, IF_x), axis=0)
    retrain_y = numpy.concatenate((data_y, IF_y), axis=0)

    return retrain_x, retrain_y


def get_retrain_acc_and_fair_data(model_file, train_file, test_file, similar_file, chose_size, K=0, dist=0):
    """
    使用数据集中的 acc&fair data进行重训练
    :return:
    """
    model = load_model(model_file)
    data1 = numpy.load(test_file)

    x1, y1 = numpy.split(data1, [-1, ], axis=1)
    pre1 = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)

    data2 = numpy.squeeze(numpy.load(similar_file))
    x2 = []
    pre2 = []
    for j in range(data2.shape[1]):
        x2.append(data2[:, j, :])
        pre2.append(numpy.argmax(model.predict(data2[:, j, :]), axis=1).reshape(-1, 1))

    MSE = calculate_mse(pre1, y1)
    AF_cond1 = check_dist(MSE, dist)
    AF_cond2 = numpy.ones(AF_cond1.shape)
    for i in range(len(pre2)):
        # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
        D_distance = calculate_mse(y1, pre2[i])
        Kd_distance = K * calculate_mse(x1, x2[i])
        AF_cond2 = numpy.logical_and(AF_cond2, check_dist(D_distance - Kd_distance, dist))
    AF_cond = numpy.logical_and(AF_cond1, AF_cond2)
    AF_data = data1[AF_cond]
    numpy.random.shuffle(AF_data)
    chose_data1, chose_data2 = numpy.split(AF_data, [chose_size, ], axis=0)

    data = numpy.load(train_file)
    data_x, data_y = numpy.split(data, [-1, ], axis=1)
    AF_x, AF_y = numpy.split(chose_data1, [-1, ], axis=1)

    retrain_x = numpy.concatenate((data_x, AF_x), axis=0)
    retrain_y = numpy.concatenate((data_y, AF_y), axis=0)

    return retrain_x, retrain_y
