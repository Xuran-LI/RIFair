import numpy
import keras
from keras import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.utils_Transform_AutoInt import TokenAndPositionEmbeddingBank, TransformerBlock


def train_bank_model(model_file, inputs_data, label_data):
    """
    训练模型
    :return:
    """
    vocab_size = 1500  # 字典大小
    embed_size = 32  # Embedding 大小
    num_heads = 4  # 注意力头 大小
    ff_dim = 32  # transform层中前馈神经网络大小
    maxlen = 90
    # 输入index、value
    inputs = layers.Input(shape=(maxlen,))
    # 对index、value进行编码
    embedding_layer = TokenAndPositionEmbeddingBank(maxlen, vocab_size, embed_size)
    embeddings = embedding_layer(inputs)
    # 使用AutoInt学习数据间的3-order组合属性
    transformer_block1 = TransformerBlock(embed_size, num_heads, ff_dim)
    transformer1 = transformer_block1(embeddings)
    transformer_block2 = TransformerBlock(embed_size, num_heads, ff_dim)
    transformer2 = transformer_block2(transformer1)
    transformer_block3 = TransformerBlock(embed_size, num_heads, ff_dim)
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
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=["acc"])
    C_point = ModelCheckpoint(model_file, monitor='acc', save_best_only=True)
    model.fit(x=inputs_data, y=label_data, batch_size=128, epochs=10, callbacks=C_point)


if __name__ == "__main__":
    for i in range(5):
        train_input = numpy.load("../dataset/NLP/bank/data/code_text_train_bank.npy", allow_pickle=True)
        train_label = numpy.load("../dataset/NLP/bank/data/text_train_label.npy", allow_pickle=True)
        train_label = to_categorical(train_label, num_classes=2)
        save_file = "../dataset/NLP/bank/model/Transformer_BL{}.h5".format(i)
        train_bank_model(save_file, train_input, train_label)
