import pickle
import random
import keras
import numpy
from utils.utils_RIFair import split_NLP_model_by_embedding_layer, NLP_false_bias, \
    get_bank_embedding_vector_direction_unit
from utils.utils_Transform_AutoInt import TokenAndPositionEmbeddingBank, TransformerBlock
from utils.utils_input_output import write_search_NLP_instances, write_search_times, write_search_labels


def FalseBiasedAdv(file1, files2, files3, files4, times, size1, size2):
    """
    进行准确公平性测试
    :return:
    """
    custom_layers = {'TokenAndPositionEmbeddingBank': TokenAndPositionEmbeddingBank,
                     'TransformerBlock': TransformerBlock}
    model = keras.models.load_model(file1, custom_objects=custom_layers)

    models = split_NLP_model_by_embedding_layer(model, size1, size2)
    delts = get_bank_embedding_vector_direction_unit(models[1], size1)

    test_data = numpy.load(files2[0])
    test_label = numpy.load(files2[1]).reshape(-1, 1)
    sim_data = numpy.load(files3[0])

    adv_data_file = open(files4[0], "w")
    adv_sim_file = open(files4[1], "w")
    adv_label_file = open(files4[2], "w")
    adv_time_file = open(files4[3], "w")

    for i in range(test_label.shape[0]):
        print("check {}th item".format(i))
        select_id = random.randint(0, sim_data.shape[0] - 1)
        sim_test = sim_data[select_id, i, :]
        protect = test_data[i] == sim_test
        result = NLP_false_bias(models, test_label[i], test_data[i], sim_test, times, delts, protect)
        # 保存搜索结果
        write_search_NLP_instances(adv_data_file, result[0])
        write_search_NLP_instances(adv_sim_file, result[1])
        write_search_labels(adv_label_file, result[2])
        write_search_times(adv_time_file, result[3])

    adv_data_file.close()
    adv_sim_file.close()
    adv_label_file.close()
    adv_time_file.close()


def FalseBiasedRetrain(file1, files2, files3, files4, times, size1, size2):
    """
    进行准确公平性测试
    :return:
    """
    custom_layers = {'TokenAndPositionEmbeddingBank': TokenAndPositionEmbeddingBank,
                     'TransformerBlock': TransformerBlock}
    model = keras.models.load_model(file1, custom_objects=custom_layers)
    models = split_NLP_model_by_embedding_layer(model, size1, size2)
    delts = get_bank_embedding_vector_direction_unit(models[1], size1)
    # 对样本进行打乱，然后截取前20%样本进行 adversarial retrain experiment
    train_label = numpy.load(files2[1]).reshape(-1, 1)
    shuffle_index = numpy.arange(train_label.shape[0])
    numpy.random.shuffle(shuffle_index)
    split_index = int(train_label.shape[0] * 0.20)

    train_data = numpy.load(files2[0])[shuffle_index][:split_index]
    train_label = train_label[shuffle_index][:split_index]
    sim_data = numpy.load(files3[0])[:, shuffle_index, :][:, : split_index, :]

    adv_data_file = open(files4[0], "w")
    adv_sim_file = open(files4[1], "w")
    adv_label_file = open(files4[2], "w")
    adv_time_file = open(files4[3], "w")

    for i in range(train_label.shape[0]):
        print("check {}th item".format(i))
        select_id = random.randint(0, sim_data.shape[0] - 1)
        sim_test = sim_data[select_id, i, :]
        protect = train_data[i] == sim_test
        result = NLP_false_bias(models, train_label[i], train_data[i], sim_test, times, delts, protect)
        # 保存搜索结果
        write_search_NLP_instances(adv_data_file, result[0])
        write_search_NLP_instances(adv_sim_file, result[1])
        write_search_labels(adv_label_file, result[2])
        write_search_times(adv_time_file, result[3])
    adv_data_file.close()
    adv_sim_file.close()
    adv_label_file.close()
    adv_time_file.close()


if __name__ == "__main__":
    search_time = 10
    feature_size = 90
    embed_size = 32

    # test dataset adv experiments
    M_file = "../dataset/NLP/bank/model/Transformer_BL0.h5"
    data_file = ["../dataset/NLP/bank/data/code_text_test_bank.npy",
                 "../dataset/NLP/bank/data/text_test_label.npy"]

    age_file = ["../dataset/NLP/bank/data/code_text_age_test_data.npy"]
    age_result = ["../dataset/NLP/bank/adv/FB_{}_d.txt".format("age_Transformer_BL0"),
                  "../dataset/NLP/bank/adv/FB_{}_s.txt".format("age_Transformer_BL0"),
                  "../dataset/NLP/bank/adv/FB_{}_y.txt".format("age_Transformer_BL0"),
                  "../dataset/NLP/bank/adv/FB_{}_t.txt".format("age_Transformer_BL0")]
    FalseBiasedAdv(M_file, data_file, age_file, age_result, search_time, feature_size, embed_size)

    # train dataset adv retrain experiments
    M_file = "../dataset/NLP/bank/model/Transformer_BL0.h5"
    data_file = ["../dataset/NLP/bank/data/code_text_train_bank.npy",
                 "../dataset/NLP/bank/data/text_train_label.npy"]

    age_file = ["../dataset/NLP/bank/data/code_text_age_train_data.npy"]
    age_result = ["../dataset/NLP/bank/retrain/FB_{}_d.txt".format("age_Transformer_BL0"),
                  "../dataset/NLP/bank/retrain/FB_{}_s.txt".format("age_Transformer_BL0"),
                  "../dataset/NLP/bank/retrain/FB_{}_y.txt".format("age_Transformer_BL0"),
                  "../dataset/NLP/bank/retrain/FB_{}_t.txt".format("age_Transformer_BL0")]
    FalseBiasedRetrain(M_file, data_file, age_file, age_result, search_time, feature_size, embed_size)
