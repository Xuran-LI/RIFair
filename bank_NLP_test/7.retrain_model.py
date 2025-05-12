import numpy
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils.utils_Transform_AutoInt import TokenAndPositionEmbeddingBank, TransformerBlock
from utils.utils_input_output import get_search_NLP_instances, get_search_labels, get_NLP_test_data,get_search_times


def get_retrain_data(name2, name3):
    """
    获取重训练数据
    :return:
    """
    data = numpy.load("../dataset/NLP/bank/data/code_text_train_bank.npy", allow_pickle=True)
    label = numpy.load("../dataset/NLP/bank/data/text_train_label.npy", allow_pickle=True).reshape(-1, 1)

    for name1 in ["TB", "FF", "FB"]:
        generate_files = ["../dataset/NLP/bank/retrain/{}_{}_{}_d.txt".format(name1, name2, name3),
                          "../dataset/NLP/bank/retrain/{}_{}_{}_s.txt".format(name1, name2, name3),
                          "../dataset/NLP/bank/retrain/{}_{}_{}_y.txt".format(name1, name2, name3),
                          "../dataset/NLP/bank/retrain/{}_{}_{}_t.txt".format(name1, name2, name3)]

        adv_label = get_search_labels(generate_files[2]).reshape(-1, 1)
        # 加载对抗样本
        adv_data = get_search_NLP_instances(generate_files[0])
        data = numpy.concatenate((data, adv_data), axis=0)
        label = numpy.concatenate((label, adv_label), axis=0)
        a=get_search_times(generate_files[3])
        # 加载对抗样本的相似样本
        sim_data = get_search_NLP_instances(generate_files[1])
        data = numpy.concatenate((data, sim_data), axis=0)
        label = numpy.concatenate((label, adv_label), axis=0)

    label = to_categorical(label, num_classes=2)

    return data, label


def retrain_model(P_name, M_name, retrain_data, val_data):
    """
    加载原模型进行重训练
    :return:
    """
    custom_layers = {'TokenAndPositionEmbeddingBank': TokenAndPositionEmbeddingBank, 'TransformerBlock': TransformerBlock}
    model_file = "../dataset/NLP/bank/model/{}.h5".format(M_name)
    model = load_model(model_file, custom_objects=custom_layers)
    model.summary()
    model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=["acc"])
    C_point = ModelCheckpoint("../dataset/NLP/bank/model/Re_{}_{}.h5".format(P_name, M_name), monitor='val_acc',
                              save_best_only=True)
    model.fit(x=retrain_data[0], y=retrain_data[1], validation_data=(val_data[0], val_data[1]),
              batch_size=128, epochs=10, callbacks=C_point)


if __name__ == "__main__":
    model_name = "Transformer_BL0"
    V_data = get_NLP_test_data("bank")
    R_data = get_retrain_data("age", model_name)
    retrain_model("age", model_name, R_data, V_data)

    