import numpy
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils.utils_Transform_AutoInt import BankEmbedding, AutoIntTransformerBlock
from utils.utils_input_output import get_search_instances, get_search_labels, get_test_data


def get_retrain_data(name2, name3):
    """
    获取重训练数据
    :return:
    """
    index = numpy.load("../dataset/bank/data/train_i.npy", allow_pickle=True)
    value = numpy.load("../dataset/bank/data/train_v.npy", allow_pickle=True)
    label = numpy.load("../dataset/bank/data/train_y.npy", allow_pickle=True).reshape(-1, 1)

    for name1 in ["TB", "FF", "FB"]:
        generate_files = ["../dataset/bank/retrain/{}_{}_{}_i.txt".format(name1, name2, name3),
                          "../dataset/bank/retrain/{}_{}_{}_s.txt".format(name1, name2, name3),
                          "../dataset/bank/retrain/{}_{}_{}_y.txt".format(name1, name2, name3),
                          "../dataset/bank/retrain/{}_{}_{}_t.txt".format(name1, name2, name3)]

        adv_label = get_search_labels(generate_files[2]).reshape(-1, 1)

        adv_index, adv_value = get_search_instances(generate_files[0])
        index = numpy.concatenate((index, adv_index), axis=0)
        value = numpy.concatenate((value, adv_value), axis=0)
        label = numpy.concatenate((label, adv_label), axis=0)

        sim_index, sim_value = get_search_instances(generate_files[1])
        index = numpy.concatenate((index, sim_index), axis=0)
        value = numpy.concatenate((value, sim_value), axis=0)
        label = numpy.concatenate((label, adv_label), axis=0)

    label = to_categorical(label, num_classes=2)

    return index, value, label


def retrain_model(P_name, M_name, retrain_data, val_data):
    """
    加载原模型进行重训练
    :return:
    """
    custom_layers = {'BankEmbedding': BankEmbedding, 'AutoIntTransformerBlock': AutoIntTransformerBlock}
    model_file = "../dataset/bank/model/{}.h5".format(M_name)
    model = load_model(model_file, custom_objects=custom_layers)
    model.summary()
    model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=["acc"])
    C_point = ModelCheckpoint("../dataset/bank/model/Re_{}_{}.h5".format(P_name, M_name), monitor='val_acc',
                              save_best_only=True)
    model.fit(x=[retrain_data[0], retrain_data[1]], y=retrain_data[2],
              validation_data=([val_data[0], val_data[1]], val_data[2]),
              batch_size=32, epochs=10, callbacks=C_point)


if __name__ == "__main__":
    model_name = "AutoInt_BL0"
    R_data = get_retrain_data("age", model_name)
    V_data = get_test_data("bank")
    retrain_model("age", model_name, R_data,V_data)
