import random
import keras
import numpy
from utils.utils_RIFair import split_model_by_embedding_layer, false_bias
from utils.utils_Transform_AutoInt import BankEmbedding, AutoIntTransformerBlock
from utils.utils_input_output import write_search_instances, write_search_times, write_search_labels, get_min_max


def get_embedding_vector_direction_unit(model, input_size):
    """
    计算模型embedding层的输出结果embedding vector
    计算各embedding vector到3其余embedding vector的单位方向向量
    :return:
    """
    # 获取编码信息
    fea_dim = numpy.load("../dataset/bank/data/fea_dim.npy").tolist()
    embedding_vector = []
    # 根据编码信息及embedding模型获取编码后的编码向量vector
    raw_value = numpy.array([1] * input_size)
    for i in range(fea_dim[-1] + 1):
        raw_index = numpy.array([i] * input_size)
        raw_data = [raw_index, raw_value]
        embedding_vector.append(model(raw_data)[0, 0, :])
    # 计算各编码向量到其余向的单位方向向量（向量终点-向量起点）
    direction_unit = []
    for i in range(len(embedding_vector)):
        direction_unit_vector = []
        for j in range(len(embedding_vector)):
            vector = embedding_vector[j] - embedding_vector[i]
            direction_unit_vector.append(vector)
        direction_unit.append(direction_unit_vector)
    return direction_unit


def FalseBiasedAdv(file1, files2, files3, files4, times, size1, size2, index):
    """
    进行准确公平性测试
    :return:
    """
    custom_layers = {'BankEmbedding': BankEmbedding, 'AutoIntTransformerBlock': AutoIntTransformerBlock}
    model = keras.models.load_model(file1, custom_objects=custom_layers)

    models = split_model_by_embedding_layer(model, size1, size2)
    delts = get_embedding_vector_direction_unit(models[1], size1)

    index1 = numpy.load(files2[0])
    value1 = numpy.load(files2[1])
    label1 = numpy.load(files2[2]).reshape(-1, 1)

    sim_index = numpy.load(files3[0])
    sim_value = numpy.load(files3[1])
    sim_label = numpy.load(files3[2])

    adv_x_i_file = open(files4[0], "w")
    adv_x_s_file = open(files4[1], "w")
    adv_label_file = open(files4[2], "w")
    adv_time_file = open(files4[3], "w")

    min_list, max_list = get_min_max("../dataset/bank/data/min_max.txt")
    for i in range(label1.shape[0]):
        print("check {}th item".format(i))
        select_id = random.randint(0, sim_value.shape[0] - 1)
        index2 = sim_index[select_id, i, :]
        value2 = sim_value[select_id, i, :]
        result = false_bias(models, label1[i], [index1[i], value1[i]], [index2, value2],
                            times, delts, index, min_list, max_list)
        if len(result[0]) > 1:
            write_search_instances(adv_x_i_file, result[0])
            write_search_instances(adv_x_s_file, result[1])
            write_search_labels(adv_label_file, result[2])
            write_search_times(adv_time_file, result[3])
    adv_x_i_file.close()
    adv_x_s_file.close()
    adv_label_file.close()
    adv_time_file.close()


def FalseBiasedRetrain(file1, files2, files3, files4, times, size1, size2, index):
    """
    进行准确公平性测试
    :return:
    """
    custom_layers = {'BankEmbedding': BankEmbedding, 'AutoIntTransformerBlock': AutoIntTransformerBlock}
    model = keras.models.load_model(file1, custom_objects=custom_layers)

    models = split_model_by_embedding_layer(model, size1, size2)
    delts = get_embedding_vector_direction_unit(models[1], size1)

    # 对样本进行打乱，然后截取前20%样本进行 adversarial retrain experiment
    label1 = numpy.load(files2[2]).reshape(-1, 1)
    shuffle_index = numpy.arange(label1.shape[0])
    numpy.random.shuffle(shuffle_index)
    split_index = int(label1.shape[0] * 0.20)

    index1 = numpy.load(files2[0])[shuffle_index][:split_index]
    value1 = numpy.load(files2[1])[shuffle_index][:split_index]
    label1 = label1[shuffle_index][:split_index]
    cov = [numpy.cov(index1, rowvar=False), numpy.cov(value1, rowvar=False)]

    sim_index = numpy.load(files3[0])[:, shuffle_index, :][:, : split_index, :]
    sim_value = numpy.load(files3[1])[:, shuffle_index, :][:, : split_index, :]
    sim_label = numpy.load(files3[2])[:, shuffle_index][:, : split_index]

    adv_x_i_file = open(files4[0], "w")
    adv_x_s_file = open(files4[1], "w")
    adv_label_file = open(files4[2], "w")
    adv_time_file = open(files4[3], "w")

    min_list, max_list = get_min_max("../dataset/bank/data/min_max.txt")
    for i in range(label1.shape[0]):
        print("check {}th item".format(i))
        select_id = random.randint(0, sim_value.shape[0] - 1)
        index2 = sim_index[select_id, i, :]
        value2 = sim_value[select_id, i, :]
        result = false_bias(models, label1[i], [index1[i], value1[i]], [index2, value2],
                            times, delts, index, min_list, max_list)
        if len(result[0]) > 1:
            write_search_instances(adv_x_i_file, result[0])
            write_search_instances(adv_x_s_file, result[1])
            write_search_labels(adv_label_file, result[2])
            write_search_times(adv_time_file, result[3])
    adv_x_i_file.close()
    adv_x_s_file.close()
    adv_label_file.close()
    adv_time_file.close()


if __name__ == "__main__":
    search_time = 10
    feature_size = 16
    embed_size = 32
    age = [0]

    # test dataset adv experiments
    M_file = "../dataset/bank/model/AutoInt_BL0.h5"
    data_file = ["../dataset/bank/data/test_i.npy", "../dataset/bank/data/test_v.npy",
                 "../dataset/bank/data/test_y.npy"]

    age_file = ["../dataset/bank/data/age_test_i.npy", "../dataset/bank/data/age_test_v.npy",
                "../dataset/bank/data/age_test_y.npy"]
    age_result = ["../dataset/bank/adv/FB_{}_i.txt".format("age_AutoInt_BL0"),
                  "../dataset/bank/adv/FB_{}_s.txt".format("age_AutoInt_BL0"),
                  "../dataset/bank/adv/FB_{}_y.txt".format("age_AutoInt_BL0"),
                  "../dataset/bank/adv/FB_{}_t.txt".format("age_AutoInt_BL0")]
    FalseBiasedAdv(M_file, data_file, age_file, age_result, search_time, feature_size, embed_size, age)

    # train dataset adv retrain experiments
    M_file = "../dataset/bank/model/AutoInt_BL0.h5"
    data_file = ["../dataset/bank/data/train_i.npy", "../dataset/bank/data/train_v.npy",
                 "../dataset/bank/data/train_y.npy"]

    age_file = ["../dataset/bank/data/age_train_i.npy", "../dataset/bank/data/age_train_v.npy",
                "../dataset/bank/data/age_train_y.npy"]
    age_result = ["../dataset/bank/retrain/FB_{}_i.txt".format("age_AutoInt_BL0"),
                  "../dataset/bank/retrain/FB_{}_s.txt".format("age_AutoInt_BL0"),
                  "../dataset/bank/retrain/FB_{}_y.txt".format("age_AutoInt_BL0"),
                  "../dataset/bank/retrain/FB_{}_t.txt".format("age_AutoInt_BL0")]
    FalseBiasedRetrain(M_file, data_file, age_file, age_result, search_time, feature_size, embed_size, age)
