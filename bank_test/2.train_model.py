import numpy
import keras
from keras import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.utils_Transform_AutoInt import BankEmbedding, AutoIntTransformerBlock


def train_bank_model(M_name, D_index, D_value, D_label):
    """
    训练模型
    :return:
    """
    vocab_size = 54  # 字典大小
    embed_size = 32  # Embedding 大小
    index_size = 16  # index 大小
    value_size = 16  # value 大小
    num_heads = 4  # 注意力头 大小
    ff_dim = 32  # transform层中前馈神经网络大小
    # 输入index、value
    index_inputs = layers.Input(shape=(index_size,))
    value_inputs = layers.Input(shape=(value_size,))
    # 对index、value进行编码
    embedding_layer = BankEmbedding(vocab_size, embed_size)
    embeddings = embedding_layer(index_inputs, value_inputs)
    # 使用AutoInt学习数据间的3-order组合属性
    transformer_block1 = AutoIntTransformerBlock(embed_size, num_heads, ff_dim)
    transformer1 = transformer_block1(embeddings)
    transformer_block2 = AutoIntTransformerBlock(embed_size, num_heads, ff_dim)
    transformer2 = transformer_block2(transformer1)
    transformer_block3 = AutoIntTransformerBlock(embed_size, num_heads, ff_dim)
    transformer3 = transformer_block3(transformer2)
    # 使用AutoInt提取的特征进行深度神经网络学习
    x = layers.GlobalAveragePooling1D()(transformer3)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs=[index_inputs, value_inputs], outputs=outputs)
    model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=["acc"])
    C_point = ModelCheckpoint("../dataset/bank/model/AutoInt_{}.h5".format(M_name), monitor='acc', save_best_only=True)
    model.fit(x=[D_index, D_value], y=D_label, batch_size=32, epochs=10, callbacks=C_point)


if __name__ == "__main__":
    for i in range(5):
        train_index = numpy.load("../dataset/bank/data/train_i.npy", allow_pickle=True)
        train_value = numpy.load("../dataset/bank/data/train_v.npy", allow_pickle=True)
        train_label = numpy.load("../dataset/bank/data/train_y.npy", allow_pickle=True)
        train_label = to_categorical(train_label, num_classes=2)
        train_bank_model("BL{}".format(i), train_index, train_value, train_label)