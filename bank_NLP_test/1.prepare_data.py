import pickle
import numpy
from tensorflow.python.keras.preprocessing.text import Tokenizer

from utils.utils_NLP_prepare_data import split_NLP_bank_data, generate_bank_text_data, save_sequence_data, \
    bank_text_data_augmentation_age, save_augmentation_sequence_data, generate_bank_adv_replace_synonyms

if __name__ == "__main__":
    split_NLP_bank_data("../dataset/NLP/bank/data/train_data.csv", "../dataset/NLP/bank/data/test_data.csv")
    # 对测试集进行公平数据增强
    bank_text_data_augmentation_age("../dataset/NLP/bank/data/test_data.csv",
                                    "../dataset/NLP/bank/data/age_test_data.npy")
    # 对训练集进行公平数据增强
    bank_text_data_augmentation_age("../dataset/NLP/bank/data/train_data.csv",
                                    "../dataset/NLP/bank/data/age_train_data.npy")

    # 将训练集、测试集、数据增强后的表格数据集转换为文本数据
    generate_bank_text_data("../dataset/NLP/bank/data/train_data.csv",
                            "../dataset/NLP/bank/data/age_train_data.npy",
                            "../dataset/NLP/bank/data/text_train_bank.npy",
                            "../dataset/NLP/bank/data/text_train_label.npy",
                            "../dataset/NLP/bank/data/text_age_train_data.npy")

    generate_bank_text_data("../dataset/NLP/bank/data/test_data.csv",
                            "../dataset/NLP/bank/data/age_test_data.npy",
                            "../dataset/NLP/bank/data/text_test_bank.npy",
                            "../dataset/NLP/bank/data/text_test_label.npy",
                            "../dataset/NLP/bank/data/text_age_test_data.npy")

    # 初始化 Tokenizer，设置词汇表大小和保留特殊标记
    tokenizer = Tokenizer(num_words=1500, filters='-()', lower=False, oov_token="<UNK>")
    text_train_bank = numpy.load("../dataset/NLP/bank/data/text_train_bank.npy")
    text_test_bank = numpy.load("../dataset/NLP/bank/data/text_test_bank.npy")
    text_bank = numpy.concatenate((text_train_bank, text_test_bank), axis=0)
    # 获取各个单词的同义词替换，进行对抗扰动
    with open("../dataset/bank/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    adv_synonyms_text = generate_bank_adv_replace_synonyms()
    text_data = numpy.concatenate((text_bank, adv_synonyms_text), axis=0)
    tokenizer.fit_on_texts(text_data)
    # 保存字典
    word_index = tokenizer.word_index
    with open("../dataset/NLP/bank/data/vocab_dic.pkl", 'wb') as f:
        pickle.dump(word_index, f)
    # 对测试集训练进行编码及填充
    save_sequence_data("../dataset/NLP/bank/data/text_test_bank.npy", tokenizer, 90,
                       "../dataset/NLP/bank/data/code_text_test_bank.npy")
    save_sequence_data("../dataset/NLP/bank/data/text_train_bank.npy", tokenizer, 90,
                       "../dataset/NLP/bank/data/code_text_train_bank.npy")
    # 对公平性test数据集进行进行编码及填充
    save_augmentation_sequence_data("../dataset/NLP/bank/data/text_age_test_data.npy", tokenizer, 90,
                                    "../dataset/NLP/bank/data/code_text_age_test_data.npy")
    # 对公平性train数据集进行进行编码及填充
    save_augmentation_sequence_data("../dataset/NLP/bank/data/text_age_train_data.npy", tokenizer, 90,
                                    "../dataset/NLP/bank/data/code_text_age_train_data.npy")
