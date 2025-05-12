import pickle
import numpy
import pandas
import random
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime
from nltk.corpus import wordnet

from utils.utils_input_output import write_min_max


# adult dataset
def get_adult_voca_dic_and_fea_dim():
    """
    根据数据集中的数据获取编码所需字典，及各个属性维度
    :return:
    """
    adult_df = pandas.read_csv("../dataset/adult/data/adult.data")
    raw_data = adult_df[['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation',
                         'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                         'native-country', 'income']].values

    # 初始化各属性字典
    vocab_dic = {}
    for i in range(13):
        vocab_dic[i] = {}
    for j in [0, 3, 9, 10, 11]:  # numerical feature 位置
        vocab_dic[j]["number"] = [len(vocab_dic[j]) + 1]
    for h in range(raw_data.shape[0]):
        item = raw_data[h]
        for hh in [1, 2, 4, 5, 6, 7, 8, 12]:  # category feature 位置
            if item[hh] not in vocab_dic[hh]:  # 特征值item[hh]是否在字典feature_dictionary[h] 中
                # 特征值不在，为字典feature_dictionary[hh]增加key值item[hh]，value值初始化为len(vocab_dic[hh])+1
                vocab_dic[hh][item[hh]] = [len(vocab_dic[hh]) + 1]

    pandas.DataFrame(raw_data).to_csv("../dataset/adult/data/data.csv", index=False)

    with open("../dataset/adult/data/vocab_dic.pkl", 'wb') as f:
        pickle.dump(vocab_dic, f)
    # calculate number of category/numerical features of every dimension
    fea_dim = [len(vocab_dic[vd]) for vd in vocab_dic]
    for i in range(1, len(fea_dim)):
        fea_dim[i] += fea_dim[i - 1]
    numpy.save("../dataset/adult/data/fea_dim.npy", fea_dim)


def split_adult_data():
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv("../dataset/adult/data/data.csv").values
    numpy.random.shuffle(raw_data)
    tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    pandas.DataFrame(tran_data).to_csv("../dataset/adult/data/train_data.csv", index=False)
    pandas.DataFrame(test_data).to_csv("../dataset/adult/data/test_data.csv", index=False)


def reCode_adult_data(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv(data_file).values
    with open("../dataset/adult/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/adult/data/fea_dim.npy").tolist()
    # 对数据进行重新编码
    data_values = []
    data_index = []
    data_label = []
    for j in range(raw_data.shape[0]):
        item = raw_data[j]
        value = [0] * 13
        index = [None] * 13
        label = item[13]
        if label.find('>50K') > -1:
            label = 1
        else:
            label = 0
        for i in [0, 3, 9, 10, 11]:
            if item[i] != '':  # numerical feature 位置，取原值，空值取0
                value[i] = int(item[i])
            index[i] = fea_dim[i]
        for i in [1, 2, 4, 5, 6, 7, 8, 12]:  # category feature 位置, 非空值取1，空值取0
            if item[i] != '':
                value[i] = 1
            index[i] = (vocab_dic[i][item[i]][0])  # index为对应特征值的在该属性阈的index
        for i in [1, 2, 4, 5, 6, 7, 8, 12]:
            index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
        data_index.append(index)
        data_values.append(value)
        data_label.append(label)
    numpy.save("../dataset/adult/data/{}_i.npy".format(file_name), data_index)
    numpy.save("../dataset/adult/data/{}_v.npy".format(file_name), data_values)
    numpy.save("../dataset/adult/data/{}_y.npy".format(file_name), data_label)

    min_max = {}
    for j in range(13):
        min_max[j] = [100000, 0]
    for j in range(len(data_values)):
        item = data_values[j]
        for hh in range(len(item)):  # 获取value取值的最小值、最大值
            if int(item[hh]) < min_max[hh][0]:
                min_max[hh][0] = int(item[hh])
            if int(item[hh]) > min_max[hh][1]:
                min_max[hh][1] = int(item[hh])
    with open("../dataset/adult/data/min_max.txt", 'w') as f:
        write_min_max(f, min_max)
    f.close()


def adult_data_augmentation_race(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    with open("../dataset/adult/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    data = pandas.read_csv(data_file).values
    aug = []
    races = vocab_dic[7].keys()
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            aug_data = data[i].tolist()
            if race != aug_data[7]:
                aug_data[7] = race
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def adult_data_augmentation_gender(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    with open("../dataset/adult/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    data = pandas.read_csv(data_file).values
    aug = []
    genders = vocab_dic[8].keys()
    for i in range(data.shape[0]):
        data_list = []
        for gender in genders:
            aug_data = data[i].tolist()
            if gender != aug_data[8]:
                aug_data[8] = gender
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def adult_data_augmentation_multiple(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    with open("../dataset/adult/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    data = pandas.read_csv(data_file).values
    aug = []
    races = vocab_dic[7].keys()
    genders = vocab_dic[8].keys()
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            for gender in genders:
                aug_data = data[i].tolist()
                if race != aug_data[7] or gender != aug_data[8]:
                    aug_data[7] = race
                    aug_data[8] = gender
                    data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def reCode_adult_data_similar(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    aug_data = numpy.load(data_file)
    with open("../dataset/adult/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/adult/data/fea_dim.npy").tolist()
    aug_data_values = []
    aug_data_index = []
    aug_data_label = []
    for x in range(aug_data.shape[1]):
        # 对数据进行重新编码
        raw_data = aug_data[:, x, :]
        data_values = []
        data_index = []
        data_label = []
        for j in range(raw_data.shape[0]):
            item = raw_data[j]
            value = [0] * 13
            index = [None] * 13
            label = item[13]
            if label.find('>50K') > -1:
                label = 1
            else:
                label = 0
            for i in [0, 3, 9, 10, 11]:
                if item[i] != '':  # numerical feature 位置，取原值，空值取0
                    value[i] = int(item[i])
                index[i] = fea_dim[i]
            for i in [1, 2, 4, 5, 6, 7, 8, 12]:  # category feature 位置, 非空值取1，空值取0
                if item[i] != '':
                    value[i] = 1
                index[i] = (vocab_dic[i][item[i]][0])  # index为对应特征值的在该属性阈的index
            for i in [1, 2, 4, 5, 6, 7, 8, 12]:
                index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
            data_values.append(value)
            data_index.append(index)
            data_label.append(label)
        aug_data_values.append(data_values)
        aug_data_index.append(data_index)
        aug_data_label.append(data_label)
    numpy.save("../dataset/adult/data/{}_i.npy".format(file_name), aug_data_index)
    numpy.save("../dataset/adult/data/{}_v.npy".format(file_name), aug_data_values)
    numpy.save("../dataset/adult/data/{}_y.npy".format(file_name), aug_data_label)


# bank dataset
def get_bank_voca_dic_and_fea_dim():
    """
    读取compas数据集，将其保存为text格式
    :return:
    """
    bank_file = "../dataset/bank/data/bank-full.csv"
    raw_data = pandas.read_csv(bank_file).values
    bank_data = []
    for i in range(len(raw_data)):
        item = raw_data[i, 0].replace('"', '').replace('.', '').split(";")
        if int(item[0]) < 45:
            item[0] = "youth"
        elif int(item[0]) < 60:
            item[0] = "middle-age"
        else:
            item[0] = "the old"

        bank_data.append(item)

    # 初始化各属性字典
    vocab_dic = {}
    for i in range(16):
        vocab_dic[i] = {}
    for j in [5, 9, 11, 12, 13, 14]:  # numerical feature 位置
        vocab_dic[j]["number"] = [len(vocab_dic[j]) + 1]

    for h in range(len(bank_data)):
        item = bank_data[h]
        for hh in [0, 1, 2, 3, 4, 6, 7, 8, 10, 15]:  # category feature 位置
            if str(item[hh]) not in vocab_dic[hh]:  # 特征值item[i]是否在字典feature_dictionary[i] 中
                # 特征值不在，为字典feature_dictionary[i]增加key值item[i]，value值初始化为len(vocab_dic[i])+1
                vocab_dic[hh][str(item[hh])] = [len(vocab_dic[hh]) + 1]

    pandas.DataFrame(bank_data).to_csv("../dataset/bank/data/data.csv", index=False)

    with open("../dataset/bank/data/vocab_dic.pkl", 'wb') as f:
        pickle.dump(vocab_dic, f)
    # calculate number of category/numerical features of every dimension
    fea_dim = [len(vocab_dic[vd]) for vd in vocab_dic]
    for i in range(1, len(fea_dim)):
        fea_dim[i] += fea_dim[i - 1]
    numpy.save("../dataset/bank/data/fea_dim.npy", fea_dim)


def split_bank_data():
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv("../dataset/bank/data/data.csv").values
    numpy.random.shuffle(raw_data)
    tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    pandas.DataFrame(tran_data).to_csv("../dataset/bank/data/train_data.csv", index=False)
    pandas.DataFrame(test_data).to_csv("../dataset/bank/data/test_data.csv", index=False)


def reCode_bank_data(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv(data_file).values
    with open("../dataset/bank/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/bank/data/fea_dim.npy").tolist()
    # 对数据进行重新编码
    data_values = []
    data_index = []
    data_label = []
    for j in range(raw_data.shape[0]):
        item = raw_data[j]
        value = [0] * 16
        index = [None] * 16
        label = item[16]
        if label.find('yes') > -1:
            label = 1
        else:
            label = 0
        for i in [5, 9, 11, 12, 13, 14]:
            if item[i] != '':  # numerical feature 位置，取原值，空值取0
                value[i] = float(item[i])
            else:
                break
            index[i] = fea_dim[i]
        for i in [0, 1, 2, 3, 4, 6, 7, 8, 10, 15]:  # category feature 位置, 非空值取1，空值取0
            if item[i] != '':
                value[i] = 1
            index[i] = (vocab_dic[i][str(item[i])][0])  # index为对应特征值的在该属性阈的index
        for i in [1, 2, 3, 4, 6, 7, 8, 10, 15]:
            index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
        data_index.append(index)
        data_values.append(value)
        data_label.append(label)
    numpy.save("../dataset/bank/data/{}_i.npy".format(file_name), data_index)
    numpy.save("../dataset/bank/data/{}_v.npy".format(file_name), data_values)
    numpy.save("../dataset/bank/data/{}_y.npy".format(file_name), data_label)

    min_max = {}
    for j in range(16):
        min_max[j] = [100000, 0]
    for j in range(len(data_values)):
        item = data_values[j]
        for hh in range(len(item)):  # 获取value取值的最小值、最大值
            if int(item[hh]) < min_max[hh][0]:
                min_max[hh][0] = int(item[hh])
            if int(item[hh]) > min_max[hh][1]:
                min_max[hh][1] = int(item[hh])
    with open("../dataset/bank/data/min_max.txt", 'w') as f:
        write_min_max(f, min_max)
    f.close()


def bank_data_augmentation_age(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    with open("../dataset/bank/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    data = pandas.read_csv(data_file).values
    aug = []
    ages = vocab_dic[0].keys()
    for i in range(data.shape[0]):
        data_list = []
        for age in ages:
            aug_data = data[i].tolist()
            if age != aug_data[0]:
                aug_data[0] = age
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def reCode_bank_data_similar(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    aug_data = numpy.load(data_file)
    with open("../dataset/bank/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/bank/data/fea_dim.npy").tolist()
    aug_data_values = []
    aug_data_index = []
    aug_data_label = []
    for x in range(aug_data.shape[1]):
        # 对数据进行重新编码
        raw_data = aug_data[:, x, :]
        data_values = []
        data_index = []
        data_label = []
        for j in range(raw_data.shape[0]):
            item = raw_data[j]
            value = [0] * 16
            index = [None] * 16
            label = item[16]
            if label.find('yes') > -1:
                label = 1
            else:
                label = 0
            for i in [5, 9, 11, 12, 13, 14]:
                if item[i] != '':  # numerical feature 位置，取原值，空值取0
                    value[i] = float(item[i])
                else:
                    break
                index[i] = fea_dim[i]
            for i in [0, 1, 2, 3, 4, 6, 7, 8, 10, 15]:  # category feature 位置, 非空值取1，空值取0
                if item[i] != '':
                    value[i] = 1
                index[i] = (vocab_dic[i][str(item[i])][0])  # index为对应特征值的在该属性阈的index
            for i in [1, 2, 3, 4, 6, 7, 8, 10, 15]:
                index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
            data_index.append(index)
            data_values.append(value)
            data_label.append(label)
        aug_data_values.append(data_values)
        aug_data_index.append(data_index)
        aug_data_label.append(data_label)
    numpy.save("../dataset/bank/data/{}_i.npy".format(file_name), aug_data_index)
    numpy.save("../dataset/bank/data/{}_v.npy".format(file_name), aug_data_values)
    numpy.save("../dataset/bank/data/{}_y.npy".format(file_name), aug_data_label)


def generate_bank_text_data():
    """
    使用template模板将adult表格数据转换为文本text数据
    template：The individual is a {age_group} {marital} individual, employed as a {job} with a {education} degree.
    Their credit defaults record as {default}, annual balances as €{balance}, housing loans as {housing_loan},
    and personal loan as {personal_loan} . During the current campaign, They were contacted {campaign} times,
    most recently via {contact} on {month} {day} (duration: {duration} seconds). In previous campaigns,
    They were contacted {previous} times {pdays} days ago, resulting in a {poutcome} outcome.
    → Target subscription decision: ‌"{yes/no}"‌.
    :return:
    """


# COMPAS dataset
def get_compas_voca_dic_and_fea_dim():
    """
    读取compas数据集，将其保存为text格式
    :return:
    """
    compas_file = "../dataset/compas/data/compas-scores-two-years.csv"
    compas_df = pandas.read_csv(compas_file)
    compas_df = compas_df[
        ['sex', 'age_cat', 'race', 'decile_score', 'priors_count', 'c_jail_in', 'c_jail_out', 'c_charge_degree',
         'c_charge_desc', 'is_recid', 'r_charge_degree', 'r_charge_desc', 'is_violent_recid', 'vr_charge_degree',
         'vr_charge_desc', 'decile_score.1', 'score_text', 'v_decile_score', 'v_score_text', 'priors_count.1',
         'two_year_recid']]
    # # Indices of data samples to keep
    # ix = compas_df['days_b_screening_arrest'] > -30
    # # ix = (compas_df['days_b_screening_arrest'] >  -30) & ix
    # ix = (compas_df['is_recid'] != -1) & ix
    # ix = (compas_df['c_charge_degree'] != "O") & ix
    # # ix = (df['score_text'] != 'N/A') & ix
    # # compas_df = compas_df.loc[ix, :]
    # compas_df['length_of_stay'] = (
    #         pandas.to_datetime(compas_df['c_jail_out']) - pandas.to_datetime(compas_df['c_jail_in'])).apply(
    #     lambda x: x.days)
    # ix = compas_df['length_of_stay'] > 0
    # compas_df = compas_df.loc[ix, :]
    raw_data = compas_df[
        ['sex', 'age_cat', 'race', 'decile_score', 'priors_count', 'c_charge_degree', 'c_charge_desc', 'is_recid',
         'r_charge_degree', 'r_charge_desc', 'is_violent_recid', 'vr_charge_degree', 'vr_charge_desc', 'decile_score.1',
         'score_text', 'v_decile_score', 'v_score_text', 'priors_count.1', 'two_year_recid']].values

    # 初始化各属性字典
    vocab_dic = {}
    for i in range(18):
        vocab_dic[i] = {}
    for j in [3, 4, 7, 10, 13, 15, 17]:  # numerical feature 位置
        vocab_dic[j]["number"] = [len(vocab_dic[j]) + 1]

    for h in range(raw_data.shape[0]):
        item = raw_data[h]
        for hh in [0, 1, 2, 5, 6, 8, 9, 11, 12, 14, 16]:  # category feature 位置
            if str(item[hh]) not in vocab_dic[hh]:  # 特征值item[i]是否在字典feature_dictionary[i] 中
                # 特征值不在，为字典feature_dictionary[i]增加key值item[i]，value值初始化为len(vocab_dic[i])+1
                vocab_dic[hh][str(item[hh])] = [len(vocab_dic[hh]) + 1]

    pandas.DataFrame(raw_data).to_csv("../dataset/compas/data/data.csv", index=False)

    with open("../dataset/compas/data/vocab_dic.pkl", 'wb') as f:
        pickle.dump(vocab_dic, f)
    # calculate number of category/numerical features of every dimension
    fea_dim = [len(vocab_dic[vd]) for vd in vocab_dic]
    for i in range(1, len(fea_dim)):
        fea_dim[i] += fea_dim[i - 1]
    numpy.save("../dataset/compas/data/fea_dim.npy", fea_dim)


def split_compas_data():
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv("../dataset/compas/data/data.csv").values
    numpy.random.shuffle(raw_data)
    tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    pandas.DataFrame(tran_data).to_csv("../dataset/compas/data/train_data.csv", index=False)
    pandas.DataFrame(test_data).to_csv("../dataset/compas/data/test_data.csv", index=False)


def reCode_compas_data(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv(data_file).values
    with open("../dataset/compas/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/compas/data/fea_dim.npy").tolist()
    # 对数据进行重新编码
    data_values = []
    data_index = []
    data_label = []
    for j in range(raw_data.shape[0]):
        item = raw_data[j]
        value = [0] * 18
        index = [None] * 18
        label = item[18]
        for i in [3, 4, 7, 10, 13, 15, 17]:
            if item[i] != 'nan':  # numerical feature 位置，取原值，空值取0
                value[i] = float(item[i])
            else:
                break
            index[i] = fea_dim[i]
        for i in [0, 1, 2, 5, 6, 8, 9, 11, 12, 14, 16]:  # category feature 位置, 非空值取1，空值取0
            if str(item[i]) != 'nan':
                value[i] = 1
            index[i] = (vocab_dic[i][str(item[i])][0])  # index为对应特征值的在该属性阈的index
        for i in [1, 2, 5, 6, 8, 9, 11, 12, 14, 16]:
            index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
        data_index.append(index)
        data_values.append(value)
        data_label.append(label)
    numpy.save("../dataset/compas/data/{}_i.npy".format(file_name), data_index)
    numpy.save("../dataset/compas/data/{}_v.npy".format(file_name), data_values)
    numpy.save("../dataset/compas/data/{}_y.npy".format(file_name), data_label)

    min_max = {}
    for j in range(18):
        min_max[j] = [100000, 0]
    for j in range(len(data_values)):
        item = data_values[j]
        for hh in range(len(item)):  # 获取value取值的最小值、最大值
            if int(item[hh]) < min_max[hh][0]:
                min_max[hh][0] = int(item[hh])
            if int(item[hh]) > min_max[hh][1]:
                min_max[hh][1] = int(item[hh])
    with open("../dataset/compas/data/min_max.txt", 'w') as f:
        write_min_max(f, min_max)
    f.close()


def compas_data_augmentation_race(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    with open("../dataset/compas/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    data = pandas.read_csv(data_file).values
    aug = []
    races = vocab_dic[2].keys()
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            aug_data = data[i].tolist()
            if race != aug_data[2]:
                aug_data[2] = race
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def compas_data_augmentation_gender(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    with open("../dataset/compas/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    data = pandas.read_csv(data_file).values
    aug = []
    genders = vocab_dic[0].keys()
    for i in range(data.shape[0]):
        data_list = []
        for gender in genders:
            aug_data = data[i].tolist()
            if gender != aug_data[0]:
                aug_data[0] = gender
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def compas_data_augmentation_multiple(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    with open("../dataset/compas/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    data = pandas.read_csv(data_file).values
    aug = []
    races = vocab_dic[2].keys()
    genders = vocab_dic[0].keys()
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            for gender in genders:
                aug_data = data[i].tolist()
                if race != aug_data[2] or gender != aug_data[0]:
                    aug_data[2] = race
                    aug_data[0] = gender
                    data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def reCode_compas_data_similar(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    aug_data = numpy.load(data_file)
    with open("../dataset/compas/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/compas/data/fea_dim.npy").tolist()
    aug_data_values = []
    aug_data_index = []
    aug_data_label = []
    for x in range(aug_data.shape[1]):
        # 对数据进行重新编码
        raw_data = aug_data[:, x, :]
        data_values = []
        data_index = []
        data_label = []
        for j in range(raw_data.shape[0]):
            item = raw_data[j]
            value = [0] * 18
            index = [None] * 18
            label = item[18]
            for i in [3, 4, 7, 10, 13, 15, 17]:
                if item[i] != 'nan':  # numerical feature 位置，取原值，空值取0
                    value[i] = float(item[i])
                index[i] = fea_dim[i]
            for i in [0, 1, 2, 5, 6, 8, 9, 11, 12, 14, 16]:  # category feature 位置, 非空值取1，空值取0
                if str(item[i]) != 'nan':
                    value[i] = 1
                index[i] = (vocab_dic[i][str(item[i])][0])  # index为对应特征值的在该属性阈的index
            for i in [1, 2, 5, 6, 8, 9, 11, 12, 14, 16]:
                index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
            data_values.append(value)
            data_index.append(index)
            data_label.append(label)
        aug_data_values.append(data_values)
        aug_data_index.append(data_index)
        aug_data_label.append(data_label)
    numpy.save("../dataset/compas/data/{}_i.npy".format(file_name), aug_data_index)
    numpy.save("../dataset/compas/data/{}_v.npy".format(file_name), aug_data_values)
    numpy.save("../dataset/compas/data/{}_y.npy".format(file_name), aug_data_label)


# credit dataset
def get_credit_voca_dic_and_fea_dim():
    """
    根据数据集中的数据获取编码所需字典，及各个属性维度
    :return:
    """
    raw_data = pandas.read_csv("../dataset/credit/data/german.data").values

    credit_data = []
    for i in range(len(raw_data)):
        item = raw_data[i, 0].split()
        if item[8] == "A91":
            item[8] = "male"
            item.insert(9, "divorced")
        elif item[8] == "A92":
            item[8] = "female"
            item.insert(9, "divorced")
        elif item[8] == "A93":
            item[8] = "male"
            item.insert(9, "single")
        elif item[8] == "A94":
            item[8] = "male"
            item.insert(9, "married")
        elif item[8] == "A95":
            item[8] = "female"
            item.insert(9, "single")
        credit_data.append(item)

    # 初始化各属性字典
    vocab_dic = {}
    for i in range(21):
        vocab_dic[i] = {}
    for j in [1, 4, 7, 11, 13, 16, 18]:  # numerical feature 位置
        vocab_dic[j]["number"] = [1]
    for h in range(len(credit_data)):
        item = credit_data[h]
        for hh in [0, 2, 3, 5, 6, 8, 9, 10, 12, 14, 15, 17, 19, 20]:  # category feature 位置
            if item[hh] not in vocab_dic[hh]:  # 特征值item[hh]是否在字典feature_dictionary[h] 中
                # 特征值不在，为字典feature_dictionary[hh]增加key值item[hh]，value值初始化为len(vocab_dic[hh])+1
                vocab_dic[hh][item[hh]] = [len(vocab_dic[hh]) + 1]
    pandas.DataFrame(credit_data).to_csv("../dataset/credit/data/data.csv", index=False)
    with open("../dataset/credit/data/vocab_dic.pkl", 'wb') as f:
        pickle.dump(vocab_dic, f)
    f.close()
    # calculate number of category/numerical features of every dimension
    fea_dim = [len(vocab_dic[vd]) for vd in vocab_dic]
    for i in range(1, len(fea_dim)):
        fea_dim[i] += fea_dim[i - 1]
    numpy.save("../dataset/credit/data/fea_dim.npy", fea_dim)


def split_credit_data():
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv("../dataset/credit/data/data.csv").values
    numpy.random.shuffle(raw_data)
    tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    pandas.DataFrame(tran_data).to_csv("../dataset/credit/data/train_data.csv", index=False)
    pandas.DataFrame(test_data).to_csv("../dataset/credit/data/test_data.csv", index=False)


def reCode_credit_data(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv(data_file).values
    with open("../dataset/credit/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/credit/data/fea_dim.npy").tolist()
    # 对数据进行重新编码
    data_values = []
    data_index = []
    data_label = []
    for j in range(raw_data.shape[0]):
        item = raw_data[j]
        value = [0] * 21
        index = [None] * 21
        label = item[21]
        if label == 1:
            label = 1
        else:
            label = 0
        for i in [1, 4, 7, 11, 13, 16, 18]:
            if item[i] != '':  # numerical feature 位置，取原值，空值取0
                value[i] = float(item[i])
            else:
                break
            index[i] = fea_dim[i]
        for i in [0, 2, 3, 5, 6, 8, 9, 10, 12, 14, 15, 17, 19, 20]:  # category feature 位置, 非空值取1，空值取0
            if item[i] != '':
                value[i] = 1
            index[i] = (vocab_dic[i][str(item[i])][0])  # index为对应特征值的在该属性阈的index
        for i in [2, 3, 5, 6, 8, 9, 10, 12, 14, 15, 17, 19, 20]:
            index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
        data_index.append(index)
        data_values.append(value)
        data_label.append(label)
    numpy.save("../dataset/credit/data/{}_i.npy".format(file_name), data_index)
    numpy.save("../dataset/credit/data/{}_v.npy".format(file_name), data_values)
    numpy.save("../dataset/credit/data/{}_y.npy".format(file_name), data_label)

    min_max = {}
    for j in range(21):
        min_max[j] = [100000, 0]
    for j in range(len(data_values)):
        item = data_values[j]
        for hh in range(len(item)):  # 获取value取值的最小值、最大值
            if int(item[hh]) < min_max[hh][0]:
                min_max[hh][0] = int(item[hh])
            if int(item[hh]) > min_max[hh][1]:
                min_max[hh][1] = int(item[hh])
    with open("../dataset/credit/data/min_max.txt", 'w') as f:
        write_min_max(f, min_max)
    f.close()


def credit_data_augmentation_gender(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    with open("../dataset/credit/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    data = pandas.read_csv(data_file).values
    aug = []
    genders = vocab_dic[8].keys()
    for i in range(data.shape[0]):
        data_list = []
        for gender in genders:
            aug_data = data[i].tolist()
            if gender != aug_data[8]:
                aug_data[8] = gender
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def reCode_credit_data_similar(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    aug_data = numpy.load(data_file)
    with open("../dataset/credit/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/credit/data/fea_dim.npy").tolist()
    aug_data_values = []
    aug_data_index = []
    aug_data_label = []
    for x in range(aug_data.shape[1]):
        # 对数据进行重新编码
        raw_data = aug_data[:, x, :]
        data_values = []
        data_index = []
        data_label = []
        for j in range(raw_data.shape[0]):
            item = raw_data[j]
            value = [0] * 21
            index = [None] * 21
            label = item[21]
            if label.find('1') > -1:
                label = 1
            else:
                label = 0
            for i in [1, 4, 7, 11, 13, 16, 18]:
                if item[i] != '':  # numerical feature 位置，取原值，空值取0
                    value[i] = float(item[i])
                else:
                    break
                index[i] = fea_dim[i]
            for i in [0, 2, 3, 5, 6, 8, 9, 10, 12, 14, 15, 17, 19, 20]:  # category feature 位置, 非空值取1，空值取0
                if item[i] != '':
                    value[i] = 1
                index[i] = (vocab_dic[i][str(item[i])][0])  # index为对应特征值的在该属性阈的index
            for i in [2, 3, 5, 6, 8, 9, 10, 12, 14, 15, 17, 19, 20]:
                index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
            data_index.append(index)
            data_values.append(value)
            data_label.append(label)
        aug_data_values.append(data_values)
        aug_data_index.append(data_index)
        aug_data_label.append(data_label)
    numpy.save("../dataset/credit/data/{}_i.npy".format(file_name), aug_data_index)
    numpy.save("../dataset/credit/data/{}_v.npy".format(file_name), aug_data_values)
    numpy.save("../dataset/credit/data/{}_y.npy".format(file_name), aug_data_label)


# ACSEmployment dataset
def get_ACSEmployment_voca_dic_and_fea_dim():
    """
    ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY',
     'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P']
    根据数据集中的数据获取编码所需字典，及各个属性维度
    Target: ESR (Employment status recode): an individual’s label is 1 if ESR == 1, and 0 otherwise.
    Features:
    • AGEP (Age): Range of values:
        – 0 - 99 (integers)
        – 0 indicates less than 1 year old.
    • SCHL (Educational attainment): Range of values:
        – N/A (less than 3 years old)
        – 1: No schooling completed
        – 2: Nursery school/preschool
        – 3: Kindergarten
        – 4: Grade 1
        – 5: Grade 2
        – 6: Grade 3
        – 7: Grade 4
        – 8: Grade 5
        – 9: Grade 6
        – 10: Grade 7
        – 11: Grade 8
        – 12: Grade 9
        – 13: Grade 10
        – 14: Grade 11
        – 15: 12th Grade - no diploma
        – 16: Regular high school diploma
        – 17: GED or alternative credential
        – 18: Some college but less than 1 year
        – 19: 1 or more years of college credit but no degree
        – 20: Associate’s degree
        – 21: Bachelor’s degree
        – 22: Master’s degree
        – 23: Professional degree beyond a bachelor’s degree
        – 24: Doctorate degree
    • MAR (Marital status): Range of values:
        – 1: Married
        – 2: Widowed
        – 3: Divorced
        – 4: Separated
        – 5: Never married or under 15 years old
    • RELP (Relationship): Range of values:
        – 0: Reference person
        – 1: Husband/wife
        – 2: Biological son or daughter
        – 3: Adopted son or daughter
        – 4: Stepson or stepdaughter
        – 5: Brother or sister
        – 6: Father or mother
        – 7: Grandchild
        – 8: Parent-in-law
        – 9: Son-in-law or daughter-in-law
        – 10: Other relative
        – 11: Roomer or boarder
        – 12: Housemate or roommate
        – 13: Unmarried partner
        – 14: Foster child
        – 15: Other nonrelative
        – 16: Institutionalized group quarters population
        – 17: Noninstitutionalized group quarters population
    • DIS (Disability recode): Range of values:
        – 1: With a disability
        – 2: Without a disability
    • ESP (Employment status of parents): Range of values:
        – N/A (not own child of householder, and not child in subfamily)
        – 1: Living with two parents: both parents in labor force
        – 2: Living with two parents: Father only in labor force
        – 3: Living with two parents: Mother only in labor force
        – 4: Living with two parents: Neither parent in labor force
        – 5: Living with father: Father in the labor force
        – 6: Living with father: Father not in labor force
        – 7: Living with mother: Mother in the labor force
        – 8: Living with mother: Mother not in labor force
    • CIT (Citizenship status): Range of values:
        – 1: Born in the U.S.
        – 2: Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas
        – 3: Born abroad of American parent(s)
        – 4: U.S. citizen by naturalization
        – 5: Not a citizen of the U.S.
    • MIG (Mobility status (lived here 1 year ago): Range of values:
        – N/A (less than 1 year old)
        – 1: Yes, same house (nonmovers)
        – 2: No, outside US and Puerto Rico
        – 3: No, different house in US or Puerto Rico
    • MIL (Military service): Range of values:
        – N/A (less than 17 years old)
        – 1: Now on active duty
        – 2: On active duty in the past, but not now
        – 3: Only on active duty for training in Reserves/National Guard
        – 4: Never served in the military
    • ANC (Ancestry recode): Range of values:
        – 1: Single
        – 2: Multiple
        – 3: Unclassified
        – 4: Not reported
        – 8: Suppressed for data year 2018 for select PUMAs
    • NATIVITY (Nativity): Range of values:
        – 1: Native
        – 2: Foreign born
    • DEAR (Hearing difficulty): Range of values:
        – 1: Yes
        – 2: No
    • DEYE (Vision difficulty): Range of values:
        – 1: Yes
        – 2: No
    • DREM (Cognitive difficulty): Range of values:
        – N/A (less than 5 years old)
        – 1: Yes
        – 2: No
    • SEX (Sex): Range of values:
        – 1: Male
        – 2: Female
    • RAC1P (Recoded detailed race code): Range of values:
        – 1: White alone
        – 2: Black or African American alone
        – 3: American Indian alone
        – 4: Alaska Native alone
        – 5: American Indian and Alaska Native tribes specified, or American Indian or Alaska
        Native, not specified and no other races
        – 6: Asian alone
        – 7: Native Hawaiian and Other Pacific Islander alone
        – 8: Some Other Race alone
        – 9: Two or More Races

    :return:
    """
    data_source = ACSDataSource(root_dir="../dataset/ACS/data", survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=False)
    EmploymentFeatures, EmploymentLabel, EmploymentGroup = ACSEmployment.df_to_numpy(acs_data)
    raw_data = numpy.concatenate((EmploymentFeatures, EmploymentLabel.reshape(-1, 1)), axis=1).astype(int)
    pandas.DataFrame(raw_data).to_csv("../dataset/ACS/employment/data/data.csv", index=False)

    # 各属性字典
    vocab_dic = {
        0: {"number": 1},
        1: {  # SCHL (Educational attainment)
            "less than 3 years old": [0],
            "No schooling completed": [1],
            "Nursery school/preschool": [2],
            "Kindergarten": [3],
            "Grade 1": [4],
            "Grade 2": [5],
            "Grade 3": [6],
            "Grade 4": [7],
            "Grade 5": [8],
            "Grade 6": [9],
            "Grade 7": [10],
            "Grade 8": [11],
            "Grade 9": [12],
            "Grade 10": [13],
            "Grade 11": [14],
            "12th Grade - no diploma": [15],
            "Regular high school diploma": [16],
            "GED or alternative credential": [17],
            "Some college but less than 1 year": [18],
            "1 or more years of college credit but no degree": [19],
            "Associate's degree": [20],
            "Bachelor's degree": [21],
            "Master's degree": [22],
            "Professional degree beyond a bachelor's degree": [23],
            "Doctorate degree": [24]
        },
        2: {  # MAR (Marital status)
            "Married": [1],
            "Widowed": [2],
            "Divorced": [3],
            "Separated": [4],
            "Never married or under 15 years old": [5]
        },
        3: {  # RELP (Relationship)
            "Reference person": [0],
            "Husband/wife": [1],
            "Biological son or daughter": [2],
            "Adopted son or daughter": [3],
            "Stepson or stepdaughter": [4],
            "Brother or sister": [5],
            "Father or mother": [6],
            "Grandchild": [7],
            "Parent-in-law": [8],
            "Son-in-law or daughter-in-law": [9],
            "Other relative": [10],
            "Roomer or boarder": [11],
            "Housemate or roommate": [12],
            "Unmarried partner": [13],
            "Foster child": [14],
            "Other nonrelative": [15],
            "Institutionalized group quarters population": [16],
            "Noninstitutionalized group quarters population": [17]
        },
        4: {  # DIS (Disability recode)
            "With a disability": [1],
            "Without a disability": [2]
        },
        5: {  # ESP (Employment status of parents)
            "not own child of householder, and not child in subfamily": [0],
            "Living with two parents: both parents in labor force": [1],
            "Living with two parents: Father only in labor force": [2],
            "Living with two parents: Mother only in labor force": [3],
            "Living with two parents: Neither parent in labor force": [4],
            "Living with father: Father in the labor force": [5],
            "Living with father: Father not in labor force": [6],
            "Living with mother: Mother in the labor force": [7],
            "Living with mother: Mother not in labor force": [8]
        },
        6: {  # CIT (Citizenship status)
            "Born in the U.S.": [1],
            "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas": [2],
            "Born abroad of American parent(s)": [3],
            "U.S. citizen by naturalization": [4],
            "Not a citizen of the U.S.": [5]
        },
        7: {  # MIG (Mobility status)
            "less than 1 year old": [0],
            "Yes, same house (nonmovers)": [1],
            "No, outside US and Puerto Rico": [2],
            "No, different house in US or Puerto Rico": [3]
        },
        8: {  # MIL (Military service)
            "less than 17 years old": [0],
            "Now on active duty": [1],
            "On active duty in the past, but not now": [2],
            "Only on active duty for training in Reserves/National Guard": [3],
            "Never served in the military": [4]
        },
        9: {  # ANC (Ancestry recode)
            "Single": [1],
            "Multiple": [2],
            "Unclassified": [3],
            "Not reported": [4],
            "Suppressed for data year 2018 for select PUMAs": [8]
        },
        10: {  # NATIVITY (Nativity)
            "Native": [1],
            "Foreign born": [2]
        },
        11: {  # DEAR (Hearing difficulty)
            "Yes": [1],
            "No": [2]
        },
        12: {  # DEYE (Vision difficulty)
            "Yes": [1],
            "No": [2]
        },
        13: {  # DREM (Cognitive difficulty)
            "less than 5 years old": [0],
            "Yes": [1],
            "No": [2]
        },
        14: {  # SEX (Sex)
            "Male": [1],
            "Female": [2]
        },
        15: {  # RAC1P (Race code)
            "White alone": [1],
            "Black or African American alone": [2],
            "American Indian alone": [3],
            "Alaska Native alone": [4],
            "American Indian and Alaska Native tribes specified, or American Indian or Alaska Native, not specified and no other races": [
                5],
            "Asian alone": [6],
            "Native Hawaiian and Other Pacific Islander alone": [7],
            "Some Other Race alone": [8],
            "Two or More Races": [9]
        }
    }
    with open("../dataset/ACS/employment/data/vocab_dic.pkl", 'wb') as f:
        pickle.dump(vocab_dic, f)
    # calculate number of category/numerical features of every dimension
    fea_dim = [len(vocab_dic[vd]) for vd in vocab_dic]
    for i in range(1, len(fea_dim)):
        fea_dim[i] += fea_dim[i - 1]
    numpy.save("../dataset/ACS/employment/data/fea_dim.npy", fea_dim)


def split_ACSEmployment_data():
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv("../dataset/ACS/employment/data/data.csv").values
    numpy.random.shuffle(raw_data)
    tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    pandas.DataFrame(tran_data).to_csv("../dataset/ACS/employment/data/train_data.csv", index=False)
    pandas.DataFrame(test_data).to_csv("../dataset/ACS/employment/data/test_data.csv", index=False)


def reCode_ACSEmployment_data(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv(data_file).values
    with open("../dataset/ACS/employment/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/ACS/employment/data/fea_dim.npy").tolist()
    # 对数据进行重新编码
    data_values = []
    data_index = []
    data_label = []
    for j in range(raw_data.shape[0]):
        item = raw_data[j]
        value = [0] * 16
        index = [None] * 16
        label = item[16]
        for i in [0]:  # numerical feature 位置，取原值，空值取0
            value[i] = int(item[i])
            index[i] = fea_dim[i]
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:  # category feature 位置, 非空值取1，空值取0
            value[i] = 1
            index[i] = int(item[i])  # index为对应特征值的在该属性阈的index
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
        data_index.append(index)
        data_values.append(value)
        data_label.append(label)
    numpy.save("../dataset/ACS/employment/data/{}_i.npy".format(file_name), data_index)
    numpy.save("../dataset/ACS/employment/data/{}_v.npy".format(file_name), data_values)
    numpy.save("../dataset/ACS/employment/data/{}_y.npy".format(file_name), data_label)

    min_max = {}
    for j in range(16):
        min_max[j] = [100000, 0]
    for j in range(len(data_values)):
        item = data_values[j]
        for hh in range(len(item)):  # 获取value取值的最小值、最大值
            if int(item[hh]) < min_max[hh][0]:
                min_max[hh][0] = int(item[hh])
            if int(item[hh]) > min_max[hh][1]:
                min_max[hh][1] = int(item[hh])
    with open("../dataset/ACS/employment/data/min_max.txt", 'w') as f:
        write_min_max(f, min_max)
    f.close()


def ACSEmployment_data_augmentation_race(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            aug_data = data[i].tolist()
            if race != aug_data[15]:
                aug_data[15] = race
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def ACSEmployment_data_augmentation_gender(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    genders = [1, 2]
    for i in range(data.shape[0]):
        data_list = []
        for gender in genders:
            aug_data = data[i].tolist()
            if gender != aug_data[14]:
                aug_data[14] = gender
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def ACSEmployment_data_augmentation_multiple(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    genders = [1, 2]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            for gender in genders:
                aug_data = data[i].tolist()
                if race != aug_data[15] or gender != aug_data[14]:
                    aug_data[15] = race
                    aug_data[14] = gender
                    data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def reCode_ACSEmployment_data_similar(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    aug_data = numpy.load(data_file)
    with open("../dataset/ACS/employment/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/ACS/employment/data/fea_dim.npy").tolist()
    aug_data_values = []
    aug_data_index = []
    aug_data_label = []
    for x in range(aug_data.shape[1]):
        # 对数据进行重新编码
        raw_data = aug_data[:, x, :]
        data_values = []
        data_index = []
        data_label = []
        for j in range(raw_data.shape[0]):
            item = raw_data[j]
            value = [0] * 16
            index = [None] * 16
            label = item[16]
            for i in [0]:  # numerical feature 位置，取原值，空值取0
                value[i] = int(item[i])
                index[i] = fea_dim[i]
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:  # category feature 位置, 非空值取1，空值取0
                value[i] = 1
                index[i] = int(item[i])  # index为对应特征值的在该属性阈的index
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
                index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
            data_index.append(index)
            data_values.append(value)
            data_label.append(label)
        aug_data_values.append(data_values)
        aug_data_index.append(data_index)
        aug_data_label.append(data_label)
    numpy.save("../dataset/ACS/employment/data/{}_i.npy".format(file_name), aug_data_index)
    numpy.save("../dataset/ACS/employment/data/{}_v.npy".format(file_name), aug_data_values)
    numpy.save("../dataset/ACS/employment/data/{}_y.npy".format(file_name), aug_data_label)


# ACSIncome dataset
def get_ACSIncome_voca_dic_and_fea_dim():
    """
    ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
    根据数据集中的数据获取编码所需字典，及各个属性维度
    Target: ESR (Income status recode): an individual’s label is 1 if ESR == 1, and 0 otherwise.
    Features:
    • AGEP (Age): Range of values:
        – 0 - 99 (integers)
        – 0 indicates less than 1 year old.
    • COW (Class of worker): Range of values:
        – N/A (not in universe)
        – 1: Employee of a private for-profit company or business, or of an individual, for wages,
        salary, or commissions
        – 2: Employee of a private not-for-profit, tax-exempt, or charitable organization
        – 3: Local government employee (city, county, etc.)
        – 4: State government employee
        – 5: Federal government employee
        – 6: Self-employed in own not incorporated business, professional practice, or farm
        – 7: Self-employed in own incorporated business, professional practice or farm
        – 8: Working without pay in family business or farm
        – 9: Unemployed and last worked 5 years ago or earlier or never worked
    • SCHL (Educational attainment): Range of values:
        – N/A (less than 3 years old)
        – 1: No schooling completed
        – 2: Nursery school/preschool
        – 3: Kindergarten
        – 4: Grade 1
        – 5: Grade 2
        – 6: Grade 3
        – 7: Grade 4
        – 8: Grade 5
        – 9: Grade 6
        – 10: Grade 7
        – 11: Grade 8
        – 12: Grade 9
        – 13: Grade 10
        – 14: Grade 11
        – 15: 12th Grade - no diploma
        – 16: Regular high school diploma
        – 17: GED or alternative credential
        – 18: Some college but less than 1 year
        – 19: 1 or more years of college credit but no degree
        – 20: Associate’s degree
        – 21: Bachelor’s degree
        – 22: Master’s degree
        – 23: Professional degree beyond a bachelor’s degree
        – 24: Doctorate degree
    • MAR (Marital status): Range of values:
        – 1: Married
        – 2: Widowed
        – 3: Divorced
        – 4: Separated
        – 5: Never married or under 15 years old
    • OCCP (Occupation): Please see ACS PUMS documentation for the full list of occupation
    codes
    • POBP (Place of birth): Range of values includes most countries and individual US states;
    please see ACS PUMS documentation for the full list.
    • RELP (Relationship): Range of values:
        – 0: Reference person
        – 1: Husband/wife
        – 2: Biological son or daughter
        – 3: Adopted son or daughter
        – 4: Stepson or stepdaughter
        – 5: Brother or sister
        – 6: Father or mother
        – 7: Grandchild
        – 8: Parent-in-law
        – 9: Son-in-law or daughter-in-law
        – 10: Other relative
        – 11: Roomer or boarder
        – 12: Housemate or roommate
        – 13: Unmarried partner
        – 14: Foster child
        – 15: Other nonrelative
        – 16: Institutionalized group quarters population
        – 17: Noninstitutionalized group quarters population
    • WKHP (Usual hours worked per week past 12 months): Range of values:
        – N/A (less than 16 years old / did not work during the past 12 months)
        – 1 - 98 integer valued: usual hours worked
        – 99: 99 or more usual hours
    • SEX (Sex): Range of values:
        – 1: Male
        – 2: Female
    • RAC1P (Recoded detailed race code): Range of values:
        – 1: White alone
        – 2: Black or African American alone
        – 3: American Indian alone
        – 4: Alaska Native alone
        – 5: American Indian and Alaska Native tribes specified, or American Indian or Alaska
        Native, not specified and no other races
        – 6: Asian alone
        – 7: Native Hawaiian and Other Pacific Islander alone
        – 8: Some Other Race alone
        – 9: Two or More Races

    :return:
    """
    data_source = ACSDataSource(root_dir="../dataset/ACS/data", survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=False)
    IncomeFeatures, IncomeLabel, IncomeGroup = ACSIncome.df_to_numpy(acs_data)
    raw_data = numpy.concatenate((IncomeFeatures, IncomeLabel.reshape(-1, 1)), axis=1).astype(int)
    pandas.DataFrame(raw_data).to_csv("../dataset/ACS/income/data/data.csv", index=False)

    # 各属性字典
    vocab_dic = {
        0: {"number": 1},
        1: {  # COW (Class of worker)
            "not in universe": [0],
            "Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions": [
                1],
            "Employee of a private not-for-profit, tax-exempt, or charitable organization": [2],
            "Local government employee (city, county, etc.)": [3],
            "State government employee": [4],
            "Federal government employee": [5],
            "Self-employed in own not incorporated business, professional practice, or farm": [6],
            "Self-employed in own incorporated business, professional practice or farm": [7],
            "Working without pay in family business or farm": [8],
            "Unemployed and last worked 5 years ago or earlier or never worked": [9]
        },
        2: {  # SCHL (Educational attainment)
            "less than 3 years old": [0],
            "No schooling completed": [1],
            "Nursery school/preschool": [2],
            "Kindergarten": [3],
            "Grade 1": [4],
            "Grade 2": [5],
            "Grade 3": [6],
            "Grade 4": [7],
            "Grade 5": [8],
            "Grade 6": [9],
            "Grade 7": [10],
            "Grade 8": [11],
            "Grade 9": [12],
            "Grade 10": [13],
            "Grade 11": [14],
            "12th Grade - no diploma": [15],
            "Regular high school diploma": [16],
            "GED or alternative credential": [17],
            "Some college but less than 1 year": [18],
            "1 or more years of college credit but no degree": [19],
            "Associate's degree": [20],
            "Bachelor's degree": [21],
            "Master's degree": [22],
            "Professional degree beyond a bachelor's degree": [23],
            "Doctorate degree": [24]
        },
        3: {  # MAR (Marital status)
            "Married": [1],
            "Widowed": [2],
            "Divorced": [3],
            "Separated": [4],
            "Never married or under 15 years old": [5]
        },
        4: {  # OCCP (Occupation)
        },
        5: {  # POBP (Place of birth)
        },
        6: {  # RELP (Relationship)
            "Reference person": [0],
            "Husband/wife": [1],
            "Biological son or daughter": [2],
            "Adopted son or daughter": [3],
            "Stepson or stepdaughter": [4],
            "Brother or sister": [5],
            "Father or mother": [6],
            "Grandchild": [7],
            "Parent-in-law": [8],
            "Son-in-law or daughter-in-law": [9],
            "Other relative": [10],
            "Roomer or boarder": [11],
            "Housemate or roommate": [12],
            "Unmarried partner": [13],
            "Foster child": [14],
            "Other nonrelative": [15],
            "Institutionalized group quarters population": [16],
            "Noninstitutionalized group quarters population": [17]
        },
        7: {  # WKHP (Usual hours worked per week past 12 months)
            "number": 1
        },
        8: {  # SEX (Sex)
            "Male": [1],
            "Female": [2]
        },
        9: {  # RAC1P (Recoded detailed race code)
            "White alone": [1],
            "Black or African American alone": [2],
            "American Indian alone": [3],
            "Alaska Native alone": [4],
            "American Indian and Alaska Native tribes specified, or American Indian or Alaska Native, not specified and no other races": [
                5],
            "Asian alone": [6],
            "Native Hawaiian and Other Pacific Islander alone": [7],
            "Some Other Race alone": [8],
            "Two or More Races": [9]
        }
    }

    for h in range(len(raw_data)):
        item = raw_data[h]
        for hh in [4, 5]:  # category feature 位置
            if item[hh] not in vocab_dic[hh]:  # 特征值item[hh]是否在字典feature_dictionary[h] 中
                # 特征值不在，为字典feature_dictionary[hh]增加key值item[hh]，value值初始化为len(vocab_dic[hh])+1
                vocab_dic[hh][item[hh]] = [len(vocab_dic[hh]) + 1]

    with open("../dataset/ACS/income/data/vocab_dic.pkl", 'wb') as f:
        pickle.dump(vocab_dic, f)
    # calculate number of category/numerical features of every dimension
    fea_dim = [len(vocab_dic[vd]) for vd in vocab_dic]
    for i in range(1, len(fea_dim)):
        fea_dim[i] += fea_dim[i - 1]
    numpy.save("../dataset/ACS/income/data/fea_dim.npy", fea_dim)  # 734


def split_ACSIncome_data():
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv("../dataset/ACS/income/data/data.csv").values
    numpy.random.shuffle(raw_data)
    tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    # pandas.DataFrame(tran_data).to_csv("../dataset/ACS/income/data/train_data.csv", index=False)
    # pandas.DataFrame(test_data).to_csv("../dataset/ACS/income/data/test_data.csv", index=False)


def reCode_ACSIncome_data(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv(data_file).values
    with open("../dataset/ACS/income/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/ACS/income/data/fea_dim.npy").tolist()
    # 对数据进行重新编码
    data_values = []
    data_index = []
    data_label = []
    for j in range(raw_data.shape[0]):
        item = raw_data[j]
        value = [0] * 10
        index = [None] * 10
        label = item[10]
        for i in [0, 7]:  # numerical feature 位置，取原值，空值取0
            value[i] = int(item[i])
            index[i] = fea_dim[i]
        for i in [1, 2, 3, 6, 8, 9]:  # category feature 位置, 非空值取1，空值取0
            value[i] = 1
            index[i] = int(item[i])  # index为对应特征值的在该属性阈的index
        for i in [4, 5]:  # category feature 位置, 对第4（OCCP）、5（POBP）属性进行编码
            value[i] = 1
            index[i] = (vocab_dic[i][item[i]][0])
        for i in [1, 2, 3, 4, 5, 6, 8, 9]:
            index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
        data_index.append(index)
        data_values.append(value)
        data_label.append(label)
    numpy.save("../dataset/ACS/income/data/{}_i.npy".format(file_name), data_index)
    numpy.save("../dataset/ACS/income/data/{}_v.npy".format(file_name), data_values)
    numpy.save("../dataset/ACS/income/data/{}_y.npy".format(file_name), data_label)

    min_max = {}
    for j in range(10):
        min_max[j] = [100000, 0]
    for j in range(len(data_values)):
        item = data_values[j]
        for hh in range(len(item)):  # 获取value取值的最小值、最大值
            if int(item[hh]) < min_max[hh][0]:
                min_max[hh][0] = int(item[hh])
            if int(item[hh]) > min_max[hh][1]:
                min_max[hh][1] = int(item[hh])
    with open("../dataset/ACS/income/data/min_max.txt", 'w') as f:
        write_min_max(f, min_max)
    f.close()


def ACSIncome_data_augmentation_race(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            aug_data = data[i].tolist()
            if race != aug_data[9]:
                aug_data[9] = race
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def ACSIncome_data_augmentation_gender(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    genders = [1, 2]
    for i in range(data.shape[0]):
        data_list = []
        for gender in genders:
            aug_data = data[i].tolist()
            if gender != aug_data[8]:
                aug_data[8] = gender
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def ACSIncome_data_augmentation_multiple(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    genders = [1, 2]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            for gender in genders:
                aug_data = data[i].tolist()
                if race != aug_data[9] or gender != aug_data[8]:
                    aug_data[9] = race
                    aug_data[8] = gender
                    data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def reCode_ACSIncome_data_similar(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    aug_data = numpy.load(data_file)
    with open("../dataset/ACS/income/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/ACS/income/data/fea_dim.npy").tolist()
    aug_data_values = []
    aug_data_index = []
    aug_data_label = []
    for x in range(aug_data.shape[1]):
        # 对数据进行重新编码
        raw_data = aug_data[:, x, :]
        data_values = []
        data_index = []
        data_label = []
        for j in range(raw_data.shape[0]):
            item = raw_data[j]
            value = [0] * 10
            index = [None] * 10
            label = item[10]
            for i in [0, 7]:  # numerical feature 位置，取原值，空值取0
                value[i] = int(item[i])
                index[i] = fea_dim[i]
            for i in [1, 2, 3, 6, 8, 9]:  # category feature 位置, 非空值取1，空值取0
                value[i] = 1
                index[i] = int(item[i])  # index为对应特征值的在该属性阈的index
            for i in [4, 5]:  # category feature 位置, 对第4（OCCP）、5（POBP）属性进行编码
                value[i] = 1
                index[i] = (vocab_dic[i][item[i]][0])
            for i in [1, 2, 3, 4, 5, 6, 8, 9]:
                index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
            data_index.append(index)
            data_values.append(value)
            data_label.append(label)
        aug_data_values.append(data_values)
        aug_data_index.append(data_index)
        aug_data_label.append(data_label)
    numpy.save("../dataset/ACS/income/data/{}_i.npy".format(file_name), aug_data_index)
    numpy.save("../dataset/ACS/income/data/{}_v.npy".format(file_name), aug_data_values)
    numpy.save("../dataset/ACS/income/data/{}_y.npy".format(file_name), aug_data_label)


# ACSCoverage dataset
def get_ACSCoverage_voca_dic_and_fea_dim():
    """
    ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM',
     'PINCP', 'ESR', 'ST', 'FER', 'RAC1P']
    根据数据集中的数据获取编码所需字典，及各个属性维度
    Target: ESR (Coverage status recode): an individual’s label is 1 if ESR == 1, and 0 otherwise.
    Features:
    • AGEP (Age): Range of values:
        – 0 - 99 (integers)
        – 0 indicates less than 1 year old.
    • SCHL (Educational attainment): Range of values:
        – N/A (less than 3 years old)
        – 1: No schooling completed
        – 2: Nursery school/preschool
        – 3: Kindergarten
        – 4: Grade 1
        – 5: Grade 2
        – 6: Grade 3
        – 7: Grade 4
        – 8: Grade 5
        – 9: Grade 6
        – 10: Grade 7
        – 11: Grade 8
        – 12: Grade 9
        – 13: Grade 10
        – 14: Grade 11
        – 15: 12th Grade - no diploma
        – 16: Regular high school diploma
        – 17: GED or alternative credential
        – 18: Some college but less than 1 year
        – 19: 1 or more years of college credit but no degree
        – 20: Associate’s degree
        – 21: Bachelor’s degree
        – 22: Master’s degree
        – 23: Professional degree beyond a bachelor’s degree
        – 24: Doctorate degree
    • MAR (Marital status): Range of values:
        – 1: Married
        – 2: Widowed
        – 3: Divorced
        – 4: Separated
        – 5: Never married or under 15 years old• SEX (Sex): Range of values:
        – 1: Male
        – 2: Female
    • DIS (Disability recode): Range of values:
        – 1: With a disability
        – 2: Without a disability
    • ESP (Employment status of parents): Range of values:
        – N/A (not own child of householder, and not child in subfamily)
        – 1: Living with two parents: both parents in labor force
        – 2: Living with two parents: Father only in labor force
        – 3: Living with two parents: Mother only in labor force
        – 4: Living with two parents: Neither parent in labor force
        – 5: Living with father: Father in the labor force
        – 6: Living with father: Father not in labor force
        – 7: Living with mother: Mother in the labor force
        – 8: Living with mother: Mother not in labor force
    • CIT (Citizenship status): Range of values:
        – 1: Born in the U.S.
        – 2: Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas
        – 3: Born abroad of American parent(s)
        – 4: U.S. citizen by naturalization
        – 5: Not a citizen of the U.S.
    • MIG (Mobility status (lived here 1 year ago): Range of values:
        – N/A (less than 1 year old)
        – 1: Yes, same house (nonmovers)
        – 2: No, outside US and Puerto Rico
        – 3: No, different house in US or Puerto Rico
    • MIL (Military service): Range of values:
        – N/A (less than 17 years old)
        – 1: Now on active duty
        – 2: On active duty in the past, but not now
        – 3: Only on active duty for training in Reserves/National Guard
        – 4: Never served in the military
    • ANC (Ancestry recode): Range of values:
        – 1: Single
        – 2: Multiple
        – 3: Unclassified
        – 4: Not reported
        – 8: Suppressed for data year 2018 for select PUMAs
    • NATIVITY (Nativity): Range of values:
        – 1: Native
        – 2: Foreign born
    • DEAR (Hearing difficulty): Range of values:
        – 1: Yes
        – 2: No
    • DEYE (Vision difficulty): Range of values:
        – 1: Yes
        – 2: No
    • DREM (Cognitive difficulty): Range of values:
        – N/A (less than 5 years old)
        – 1: Yes
        – 2: No
    • PINCP (Total person’s income): Range of values:
        – integers between -19997 and 4209995 to indicate income in US dollars
        – loss of $19998 or more is coded as -19998.
        – income of $4209995 or more is coded as 4209995.
    • ESR (Employment status recode): Range of values:
        – N/A (less than 16 years old)
        – 1: Civilian employed, at work
        – 2: Civilian employed, with a job but not at work
        – 3: Unemployed
        – 4: Armed forces, at work
        – 5: Armed forces, with a job but not at work
        – 6: Not in labor force
    • ST (State code): Please see ACS PUMS documentation for the correspondence between coded
    values and state name.
    • FER (Gave birth to child within the past 12 months): Range of values:
        – N/A (less than 15 years/greater than 50 years/male)
        – 1: Yes
        – 2: No
    • RAC1P (Recoded detailed race code): Range of values:
        – 1: White alone
        – 2: Black or African American alone
        – 3: American Indian alone
        – 4: Alaska Native alone
        – 5: American Indian and Alaska Native tribes specified, or American Indian or Alaska Native, not specified and no other races
        – 6: Asian alone
        – 7: Native Hawaiian and Other Pacific Islander alone
        – 8: Some Other Race alone
        – 9: Two or More Races
    :return:
    """
    data_source = ACSDataSource(root_dir="../dataset/ACS/data", survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=False)
    CoverageFeatures, CoverageLabel, CoverageGroup = ACSPublicCoverage.df_to_numpy(acs_data)
    raw_data = numpy.concatenate((CoverageFeatures, CoverageLabel.reshape(-1, 1)), axis=1).astype(int)
    pandas.DataFrame(raw_data).to_csv("../dataset/ACS/coverage/data/data.csv", index=False)

    # 各属性字典
    vocab_dic = {
        0: {"number": 1},
        1: {  # SCHL (Educational attainment)
            "less than 3 years old": [0],
            "No schooling completed": [1],
            "Nursery school/preschool": [2],
            "Kindergarten": [3],
            "Grade 1": [4],
            "Grade 2": [5],
            "Grade 3": [6],
            "Grade 4": [7],
            "Grade 5": [8],
            "Grade 6": [9],
            "Grade 7": [10],
            "Grade 8": [11],
            "Grade 9": [12],
            "Grade 10": [13],
            "Grade 11": [14],
            "12th Grade - no diploma": [15],
            "Regular high school diploma": [16],
            "GED or alternative credential": [17],
            "Some college but less than 1 year": [18],
            "1 or more years of college credit but no degree": [19],
            "Associate's degree": [20],
            "Bachelor's degree": [21],
            "Master's degree": [22],
            "Professional degree beyond a bachelor's degree": [23],
            "Doctorate degree": [24]
        },
        2: {  # MAR (Marital status)
            "Married": [1],
            "Widowed": [2],
            "Divorced": [3],
            "Separated": [4],
            "Never married or under 15 years old": [5]
        },
        3: {  # SEX (Sex)
            "Male": [1],
            "Female": [2]
        },
        4: {  # DIS (Disability recode)
            "With a disability": [1],
            "Without a disability": [2]
        },
        5: {  # ESP (Employment status of parents)
            "not own child of householder, and not child in subfamily": [0],
            "Living with two parents: both parents in labor force": [1],
            "Living with two parents: Father only in labor force": [2],
            "Living with two parents: Mother only in labor force": [3],
            "Living with two parents: Neither parent in labor force": [4],
            "Living with father: Father in the labor force": [5],
            "Living with father: Father not in labor force": [6],
            "Living with mother: Mother in the labor force": [7],
            "Living with mother: Mother not in labor force": [8]
        },
        6: {  # CIT (Citizenship status)
            "Born in the U.S.": [1],
            "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas": [2],
            "Born abroad of American parent(s)": [3],
            "U.S. citizen by naturalization": [4],
            "Not a citizen of the U.S.": [5]
        },
        7: {  # MIG (Mobility status)
            "less than 1 year old": [0],
            "Yes, same house (nonmovers)": [1],
            "No, outside US and Puerto Rico": [2],
            "No, different house in US or Puerto Rico": [3]
        },
        8: {  # MIL (Military service)
            "less than 17 years old": [0],
            "Now on active duty": [1],
            "On active duty in the past, but not now": [2],
            "Only on active duty for training in Reserves/National Guard": [3],
            "Never served in the military": [4]
        },
        9: {  # ANC (Ancestry recode)
            "Single": [1],
            "Multiple": [2],
            "Unclassified": [3],
            "Not reported": [4],
            "Suppressed for data year 2018 for select PUMAs": [8]
        },
        10: {  # NATIVITY (Nativity)
            "Native": [1],
            "Foreign born": [2]
        },
        11: {  # DEAR (Hearing difficulty)
            "Yes": [1],
            "No": [2]
        },
        12: {  # DEYE (Vision difficulty)
            "Yes": [1],
            "No": [2]
        },
        13: {  # DREM (Cognitive difficulty)
            "less than 5 years old": [0],
            "Yes": [1],
            "No": [2]
        },
        14: {"number": 1},  # income
        15: {  # ESR (Employment status recode)
            "less than 16 years old": [0],
            "Civilian employed, at work": [1],
            "Civilian employed, with a job but not at work": [2],
            "Unemployed": [3],
            "Armed forces, at work": [4],
            "Armed forces, with a job but not at work": [5],
            "Not in labor force": [6]
        },
        16: {  # ST (State code)
        },
        17: {  # FER (Gave birth to child within the past 12 months)
            "less than 15 years/greater than 50 years/male": [0],
            "Yes": [1],
            "No": [2]
        },
        18: {  # RAC1P (Recoded detailed race code)
            "White alone": [1],
            "Black or African American alone": [2],
            "American Indian alone": [3],
            "Alaska Native alone": [4],
            "American Indian and Alaska Native tribes specified, or American Indian or Alaska Native, not specified and no other races": [
                5],
            "Asian alone": [6],
            "Native Hawaiian and Other Pacific Islander alone": [7],
            "Some Other Race alone": [8],
            "Two or More Races": [9]
        }
    }

    for h in range(len(raw_data)):
        item = raw_data[h]
        for hh in [16]:  # category feature 位置
            if item[hh] not in vocab_dic[hh]:  # 特征值item[hh]是否在字典feature_dictionary[h] 中
                # 特征值不在，为字典feature_dictionary[hh]增加key值item[hh]，value值初始化为len(vocab_dic[hh])+1
                vocab_dic[hh][item[hh]] = [len(vocab_dic[hh]) + 1]

    with open("../dataset/ACS/coverage/data/vocab_dic.pkl", 'wb') as f:
        pickle.dump(vocab_dic, f)
    # calculate number of category/numerical features of every dimension
    fea_dim = [len(vocab_dic[vd]) for vd in vocab_dic]
    for i in range(1, len(fea_dim)):
        fea_dim[i] += fea_dim[i - 1]
    numpy.save("../dataset/ACS/coverage/data/fea_dim.npy", fea_dim)  # 94


def split_ACSCoverage_data():
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv("../dataset/ACS/coverage/data/data.csv").values
    numpy.random.shuffle(raw_data)
    tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    pandas.DataFrame(tran_data).to_csv("../dataset/ACS/coverage/data/train_data.csv", index=False)
    pandas.DataFrame(test_data).to_csv("../dataset/ACS/coverage/data/test_data.csv", index=False)


def reCode_ACSCoverage_data(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv(data_file).values
    with open("../dataset/ACS/coverage/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/ACS/coverage/data/fea_dim.npy").tolist()
    # 对数据进行重新编码
    data_values = []
    data_index = []
    data_label = []
    for j in range(raw_data.shape[0]):
        item = raw_data[j]
        value = [0] * 19
        index = [None] * 19
        label = item[19]
        for i in [0, 14]:  # numerical feature 位置，取原值，空值取0
            value[i] = int(item[i])
            index[i] = fea_dim[i]
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]:  # category feature 位置, 非空值取1，空值取0
            value[i] = 1
            index[i] = int(item[i])  # index为对应特征值的在该属性阈的index
        for i in [16]:  # category feature 位置, 对第4（OCCP）、5（POBP）属性进行编码
            value[i] = 1
            index[i] = (vocab_dic[i][item[i]][0])
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]:
            index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
        data_index.append(index)
        data_values.append(value)
        data_label.append(label)
    numpy.save("../dataset/ACS/coverage/data/{}_i.npy".format(file_name), data_index)
    numpy.save("../dataset/ACS/coverage/data/{}_v.npy".format(file_name), data_values)
    numpy.save("../dataset/ACS/coverage/data/{}_y.npy".format(file_name), data_label)

    min_max = {}
    for j in range(19):
        min_max[j] = [100000, 0]
    for j in range(len(data_values)):
        item = data_values[j]
        for hh in range(len(item)):  # 获取value取值的最小值、最大值
            if int(item[hh]) < min_max[hh][0]:
                min_max[hh][0] = int(item[hh])
            if int(item[hh]) > min_max[hh][1]:
                min_max[hh][1] = int(item[hh])
    with open("../dataset/ACS/coverage/data/min_max.txt", 'w') as f:
        write_min_max(f, min_max)
    f.close()


def ACSCoverage_data_augmentation_race(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            aug_data = data[i].tolist()
            if race != aug_data[18]:
                aug_data[18] = race
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def ACSCoverage_data_augmentation_gender(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    genders = [1, 2]
    for i in range(data.shape[0]):
        data_list = []
        for gender in genders:
            aug_data = data[i].tolist()
            if gender != aug_data[3]:
                aug_data[3] = gender
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def ACSCoverage_data_augmentation_multiple(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    genders = [1, 2]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            for gender in genders:
                aug_data = data[i].tolist()
                if race != aug_data[18] or gender != aug_data[3]:
                    aug_data[18] = race
                    aug_data[3] = gender
                    data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def reCode_ACSCoverage_data_similar(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    aug_data = numpy.load(data_file)
    with open("../dataset/ACS/coverage/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/ACS/coverage/data/fea_dim.npy").tolist()
    aug_data_values = []
    aug_data_index = []
    aug_data_label = []
    for x in range(aug_data.shape[1]):
        # 对数据进行重新编码
        raw_data = aug_data[:, x, :]
        data_values = []
        data_index = []
        data_label = []
        for j in range(raw_data.shape[0]):
            item = raw_data[j]
            value = [0] * 19
            index = [None] * 19
            label = item[19]
            for i in [0, 14]:  # numerical feature 位置，取原值，空值取0
                value[i] = int(item[i])
                index[i] = fea_dim[i]
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]:  # category feature 位置, 非空值取1，空值取0
                value[i] = 1
                index[i] = int(item[i])  # index为对应特征值的在该属性阈的index
            for i in [16]:  # category feature 位置, 对第4（OCCP）、5（POBP）属性进行编码
                value[i] = 1
                index[i] = (vocab_dic[i][item[i]][0])
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]:
                index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
            data_index.append(index)
            data_values.append(value)
            data_label.append(label)
        aug_data_values.append(data_values)
        aug_data_index.append(data_index)
        aug_data_label.append(data_label)
    numpy.save("../dataset/ACS/coverage/data/{}_i.npy".format(file_name), aug_data_index)
    numpy.save("../dataset/ACS/coverage/data/{}_v.npy".format(file_name), aug_data_values)
    numpy.save("../dataset/ACS/coverage/data/{}_y.npy".format(file_name), aug_data_label)


# ACSMobility dataset
def get_ACSMobility_voca_dic_and_fea_dim():
    """
    ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIL', 'ANC', 'NATIVITY', 'RELP', 'DEAR', 'DEYE',
    'DREM', 'RAC1P', 'GCL', 'COW', 'ESR', 'WKHP', 'JWMNP', 'PINCP']
    根据数据集中的数据获取编码所需字典，及各个属性维度
    :return:
    """
    data_source = ACSDataSource(root_dir="../dataset/ACS/data", survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=False)
    MobilityFeatures, MobilityLabel, MobilityGroup = ACSMobility.df_to_numpy(acs_data)
    raw_data = numpy.concatenate((MobilityFeatures, MobilityLabel.reshape(-1, 1)), axis=1).astype(int)
    pandas.DataFrame(raw_data).to_csv("../dataset/ACS/mobility/data/data.csv", index=False)

    # 各属性字典
    vocab_dic = {
        0: {"number": 1},
        1: {  # SCHL (Educational attainment)
            "less than 3 years old": [0],
            "No schooling completed": [1],
            "Nursery school/preschool": [2],
            "Kindergarten": [3],
            "Grade 1": [4],
            "Grade 2": [5],
            "Grade 3": [6],
            "Grade 4": [7],
            "Grade 5": [8],
            "Grade 6": [9],
            "Grade 7": [10],
            "Grade 8": [11],
            "Grade 9": [12],
            "Grade 10": [13],
            "Grade 11": [14],
            "12th Grade - no diploma": [15],
            "Regular high school diploma": [16],
            "GED or alternative credential": [17],
            "Some college but less than 1 year": [18],
            "1 or more years of college credit but no degree": [19],
            "Associate's degree": [20],
            "Bachelor's degree": [21],
            "Master's degree": [22],
            "Professional degree beyond a bachelor's degree": [23],
            "Doctorate degree": [24]
        },
        2: {  # MAR (Marital status)
            "Married": [1],
            "Widowed": [2],
            "Divorced": [3],
            "Separated": [4],
            "Never married or under 15 years old": [5]
        },
        3: {  # SEX (Sex)
            "Male": [1],
            "Female": [2]
        },
        4: {  # DIS (Disability recode)
            "With a disability": [1],
            "Without a disability": [2]
        },
        5: {  # ESP (Employment status of parents)
            "not own child of householder, and not child in subfamily": [0],
            "Living with two parents: both parents in labor force": [1],
            "Living with two parents: Father only in labor force": [2],
            "Living with two parents: Mother only in labor force": [3],
            "Living with two parents: Neither parent in labor force": [4],
            "Living with father: Father in the labor force": [5],
            "Living with father: Father not in labor force": [6],
            "Living with mother: Mother in the labor force": [7],
            "Living with mother: Mother not in labor force": [8]
        },
        6: {  # CIT (Citizenship status)
            "Born in the U.S.": [1],
            "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas": [2],
            "Born abroad of American parent(s)": [3],
            "U.S. citizen by naturalization": [4],
            "Not a citizen of the U.S.": [5]
        },
        7: {  # MIL (Military service)
            "less than 17 years old": [0],
            "Now on active duty": [1],
            "On active duty in the past, but not now": [2],
            "Only on active duty for training in Reserves/National Guard": [3],
            "Never served in the military": [4]
        },
        8: {  # ANC (Ancestry recode)
            "Single": [1],
            "Multiple": [2],
            "Unclassified": [3],
            "Not reported": [4],
            "Suppressed for data year 2018 for select PUMAs": [8]
        },
        9: {  # NATIVITY (Nativity)
            "Native": [1],
            "Foreign born": [2]
        },
        10: {  # RELP (Relationship)
            "Reference person": [0],
            "Husband/wife": [1],
            "Biological son or daughter": [2],
            "Adopted son or daughter": [3],
            "Stepson or stepdaughter": [4],
            "Brother or sister": [5],
            "Father or mother": [6],
            "Grandchild": [7],
            "Parent-in-law": [8],
            "Son-in-law or daughter-in-law": [9],
            "Other relative": [10],
            "Roomer or boarder": [11],
            "Housemate or roommate": [12],
            "Unmarried partner": [13],
            "Foster child": [14],
            "Other nonrelative": [15],
            "Institutionalized group quarters population": [16],
            "Noninstitutionalized group quarters population": [17]
        },
        11: {  # DEAR (Hearing difficulty)
            "Yes": [1],
            "No": [2]
        },
        12: {  # DEYE (Vision difficulty)
            "Yes": [1],
            "No": [2]
        },
        13: {  # DREM (Cognitive difficulty)
            "less than 5 years old": [0],
            "Yes": [1],
            "No": [2]
        },
        14: {  # RAC1P (Recoded detailed race code)
            "White alone": [1],
            "Black or African American alone": [2],
            "American Indian alone": [3],
            "Alaska Native alone": [4],
            "American Indian and Alaska Native tribes specified, or American Indian or Alaska Native, not specified and no other races": [
                5],
            "Asian alone": [6],
            "Native Hawaiian and Other Pacific Islander alone": [7],
            "Some Other Race alone": [8],
            "Two or More Races": [9]
        },
        15: {  # GCL (Grandparents living with grandchildren)
            "less than 30 years/institutional GQ": [0],
            "Yes": [1],
            "No": [2]
        },
        16: {  # COW (Class of worker)
            "not in universe": [0],
            "Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions": [
                1],
            "Employee of a private not-for-profit, tax-exempt, or charitable organization": [2],
            "Local government employee (city, county, etc.)": [3],
            "State government employee": [4],
            "Federal government employee": [5],
            "Self-employed in own not incorporated business, professional practice, or farm": [6],
            "Self-employed in own incorporated business, professional practice or farm": [7],
            "Working without pay in family business or farm": [8],
            "Unemployed and last worked 5 years ago or earlier or never worked": [9]
        },
        17: {  # ESR (Employment status recode)
            "less than 16 years old": [0],
            "Civilian employed, at work": [1],
            "Civilian employed, with a job but not at work": [2],
            "Unemployed": [3],
            "Armed forces, at work": [4],
            "Armed forces, with a job but not at work": [5],
            "Not in labor force": [6]
        },
        18: {  # WKHP (Usual hours worked per week past 12 months)
            "number": 1
        },
        19: {  # JWMNP (Travel time to work)
            "number": 1
        },
        20: {  # PINCP (Total person's income)
            "number": 1
        }
    }

    with open("../dataset/ACS/mobility/data/vocab_dic.pkl", 'wb') as f:
        pickle.dump(vocab_dic, f)
    # calculate number of category/numerical features of every dimension
    fea_dim = [len(vocab_dic[vd]) for vd in vocab_dic]
    for i in range(1, len(fea_dim)):
        fea_dim[i] += fea_dim[i - 1]
    numpy.save("../dataset/ACS/mobility/data/fea_dim.npy", fea_dim)  # 119


def split_ACSMobility_data():
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv("../dataset/ACS/mobility/data/data.csv").values
    numpy.random.shuffle(raw_data)
    tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    pandas.DataFrame(tran_data).to_csv("../dataset/ACS/mobility/data/train_data.csv", index=False)
    pandas.DataFrame(test_data).to_csv("../dataset/ACS/mobility/data/test_data.csv", index=False)


def reCode_ACSMobility_data(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv(data_file).values
    with open("../dataset/ACS/mobility/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/ACS/mobility/data/fea_dim.npy").tolist()
    # 对数据进行重新编码
    data_values = []
    data_index = []
    data_label = []
    for j in range(raw_data.shape[0]):
        item = raw_data[j]
        value = [0] * 21
        index = [None] * 21
        label = item[21]
        for i in [0, 18, 19, 20]:  # numerical feature 位置，取原值，空值取0
            value[i] = int(item[i])
            index[i] = fea_dim[i]
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:  # category feature 位置, 非空值取1，空值取0
            value[i] = 1
            index[i] = int(item[i])  # index为对应特征值的在该属性阈的index
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
            index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
        data_index.append(index)
        data_values.append(value)
        data_label.append(label)
    numpy.save("../dataset/ACS/mobility/data/{}_i.npy".format(file_name), data_index)
    numpy.save("../dataset/ACS/mobility/data/{}_v.npy".format(file_name), data_values)
    numpy.save("../dataset/ACS/mobility/data/{}_y.npy".format(file_name), data_label)

    min_max = {}
    for j in range(21):
        min_max[j] = [100000, 0]
    for j in range(len(data_values)):
        item = data_values[j]
        for hh in range(len(item)):  # 获取value取值的最小值、最大值
            if int(item[hh]) < min_max[hh][0]:
                min_max[hh][0] = int(item[hh])
            if int(item[hh]) > min_max[hh][1]:
                min_max[hh][1] = int(item[hh])
    with open("../dataset/ACS/mobility/data/min_max.txt", 'w') as f:
        write_min_max(f, min_max)
    f.close()


def ACSMobility_data_augmentation_race(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            aug_data = data[i].tolist()
            if race != aug_data[14]:
                aug_data[14] = race
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def ACSMobility_data_augmentation_gender(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    genders = [1, 2]
    for i in range(data.shape[0]):
        data_list = []
        for gender in genders:
            aug_data = data[i].tolist()
            if gender != aug_data[3]:
                aug_data[3] = gender
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def ACSMobility_data_augmentation_multiple(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    genders = [1, 2]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            for gender in genders:
                aug_data = data[i].tolist()
                if race != aug_data[14] or gender != aug_data[3]:
                    aug_data[14] = race
                    aug_data[3] = gender
                    data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def reCode_ACSMobility_data_similar(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    aug_data = numpy.load(data_file)
    with open("../dataset/ACS/mobility/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/ACS/mobility/data/fea_dim.npy").tolist()
    aug_data_values = []
    aug_data_index = []
    aug_data_label = []
    for x in range(aug_data.shape[1]):
        # 对数据进行重新编码
        raw_data = aug_data[:, x, :]
        data_values = []
        data_index = []
        data_label = []
        for j in range(raw_data.shape[0]):
            item = raw_data[j]
            value = [0] * 21
            index = [None] * 21
            label = item[21]
            for i in [0, 18, 19, 20]:  # numerical feature 位置，取原值，空值取0
                value[i] = int(item[i])
                index[i] = fea_dim[i]
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:  # category feature 位置, 非空值取1，空值取0
                value[i] = 1
                index[i] = int(item[i])  # index为对应特征值的在该属性阈的index
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
                index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
            data_index.append(index)
            data_values.append(value)
            data_label.append(label)
        aug_data_values.append(data_values)
        aug_data_index.append(data_index)
        aug_data_label.append(data_label)
    numpy.save("../dataset/ACS/mobility/data/{}_i.npy".format(file_name), aug_data_index)
    numpy.save("../dataset/ACS/mobility/data/{}_v.npy".format(file_name), aug_data_values)
    numpy.save("../dataset/ACS/mobility/data/{}_y.npy".format(file_name), aug_data_label)


# ACSTravel dataset
def get_ACSTravel_voca_dic_and_fea_dim():
    """
    ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'MIG', 'RELP', 'RAC1P', 'PUMA', 'ST', 'CIT',
    'OCCP', 'JWTR', 'POWPUMA', 'POVPIP']
    根据数据集中的数据获取编码所需字典，及各个属性维度
    :return:
    """
    data_source = ACSDataSource(root_dir="../dataset/ACS/data", survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=False)
    TravelFeatures, TravelLabel, TravelGroup = ACSTravelTime.df_to_numpy(acs_data)
    raw_data = numpy.concatenate((TravelFeatures, TravelLabel.reshape(-1, 1)), axis=1).astype(int)
    pandas.DataFrame(raw_data).to_csv("../dataset/ACS/travel/data/data.csv", index=False)

    # 各属性字典
    vocab_dic = {
        0: {"number": 1},
        1: {  # SCHL (Educational attainment)
            "less than 3 years old": [0],
            "No schooling completed": [1],
            "Nursery school/preschool": [2],
            "Kindergarten": [3],
            "Grade 1": [4],
            "Grade 2": [5],
            "Grade 3": [6],
            "Grade 4": [7],
            "Grade 5": [8],
            "Grade 6": [9],
            "Grade 7": [10],
            "Grade 8": [11],
            "Grade 9": [12],
            "Grade 10": [13],
            "Grade 11": [14],
            "12th Grade - no diploma": [15],
            "Regular high school diploma": [16],
            "GED or alternative credential": [17],
            "Some college but less than 1 year": [18],
            "1 or more years of college credit but no degree": [19],
            "Associate's degree": [20],
            "Bachelor's degree": [21],
            "Master's degree": [22],
            "Professional degree beyond a bachelor's degree": [23],
            "Doctorate degree": [24]
        },
        2: {  # MAR (Marital status)
            "Married": [1],
            "Widowed": [2],
            "Divorced": [3],
            "Separated": [4],
            "Never married or under 15 years old": [5]
        },
        3: {  # SEX (Sex)
            "Male": [1],
            "Female": [2]
        },
        4: {  # DIS (Disability recode)
            "With a disability": [1],
            "Without a disability": [2]
        },
        5: {  # ESP (Employment status of parents)
            "not own child of householder, and not child in subfamily": [0],
            "Living with two parents: both parents in labor force": [1],
            "Living with two parents: Father only in labor force": [2],
            "Living with two parents: Mother only in labor force": [3],
            "Living with two parents: Neither parent in labor force": [4],
            "Living with father: Father in the labor force": [5],
            "Living with father: Father not in labor force": [6],
            "Living with mother: Mother in the labor force": [7],
            "Living with mother: Mother not in labor force": [8]
        },
        6: {  # MIG (Mobility status)
            "less than 1 year old": [0],
            "Yes, same house (nonmovers)": [1],
            "No, outside US and Puerto Rico": [2],
            "No, different house in US or Puerto Rico": [3]
        },
        7: {  # RELP (Relationship)
            "Reference person": [0],
            "Husband/wife": [1],
            "Biological son or daughter": [2],
            "Adopted son or daughter": [3],
            "Stepson or stepdaughter": [4],
            "Brother or sister": [5],
            "Father or mother": [6],
            "Grandchild": [7],
            "Parent-in-law": [8],
            "Son-in-law or daughter-in-law": [9],
            "Other relative": [10],
            "Roomer or boarder": [11],
            "Housemate or roommate": [12],
            "Unmarried partner": [13],
            "Foster child": [14],
            "Other nonrelative": [15],
            "Institutionalized group quarters population": [16],
            "Noninstitutionalized group quarters population": [17]
        },
        8: {  # RAC1P (Recoded detailed race code)
            "White alone": [1],
            "Black or African American alone": [2],
            "American Indian alone": [3],
            "Alaska Native alone": [4],
            "American Indian and Alaska Native tribes specified, or American Indian or Alaska Native, not specified and no other races": [
                5],
            "Asian alone": [6],
            "Native Hawaiian and Other Pacific Islander alone": [7],
            "Some Other Race alone": [8],
            "Two or More Races": [9]
        },
        9: {  # PUMA (Public use microdata area code)

        },
        10: {  # ST (State code)

        },
        11: {  # CIT (Citizenship status)
            "Born in the U.S.": [1],
            "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas": [2],
            "Born abroad of American parent(s)": [3],
            "U.S. citizen by naturalization": [4],
            "Not a citizen of the U.S.": [5]
        },
        12: {  # OCCP (Occupation)

        },
        13: {  # JWTR (Means of transportation to work)
            "not a worker": [0],
            "Car, truck, or van": [1],
            "Bus or trolley bus": [2],
            "Streetcar or trolley car (carro publico in Puerto Rico)": [3],
            "Subway or elevated": [4],
            "Railroad": [5],
            "Ferryboat": [6],
            "Taxicab": [7],
            "Motorcycle": [8],
            "Bicycle": [9],
            "Walked": [10],
            "Worked at home": [11],
            "Other method": [12]
        },
        14: {  # POWPUMA (Place of work PUMA)

        },
        15: {"number": 1},  # 'POVPIP'
    }

    for h in range(len(raw_data)):
        item = raw_data[h]
        for hh in [9, 10, 12, 14]:  # category feature 位置
            if item[hh] not in vocab_dic[hh]:  # 特征值item[hh]是否在字典feature_dictionary[h] 中
                # 特征值不在，为字典feature_dictionary[hh]增加key值item[hh]，value值初始化为len(vocab_dic[hh])+1
                vocab_dic[hh][item[hh]] = [len(vocab_dic[hh]) + 1]

    with open("../dataset/ACS/travel/data/vocab_dic.pkl", 'wb') as f:
        pickle.dump(vocab_dic, f)
    # calculate number of category/numerical features of every dimension
    fea_dim = [len(vocab_dic[vd]) for vd in vocab_dic]
    for i in range(1, len(fea_dim)):
        fea_dim[i] += fea_dim[i - 1]
    numpy.save("../dataset/ACS/travel/data/fea_dim.npy", fea_dim)  # 710


def split_ACSTravel_data():
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv("../dataset/ACS/travel/data/data.csv").values
    numpy.random.shuffle(raw_data)
    tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    pandas.DataFrame(tran_data).to_csv("../dataset/ACS/travel/data/train_data.csv", index=False)
    pandas.DataFrame(test_data).to_csv("../dataset/ACS/travel/data/test_data.csv", index=False)


def reCode_ACSTravel_data(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    raw_data = pandas.read_csv(data_file).values
    with open("../dataset/ACS/travel/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/ACS/travel/data/fea_dim.npy").tolist()
    # 对数据进行重新编码
    data_values = []
    data_index = []
    data_label = []
    for j in range(raw_data.shape[0]):
        item = raw_data[j]
        value = [0] * 16
        index = [None] * 16
        label = item[16]
        for i in [0, 15]:  # numerical feature 位置，取原值，空值取0
            value[i] = int(item[i])
            index[i] = fea_dim[i]
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 11, 13]:  # category feature 位置, 非空值取1，空值取0
            value[i] = 1
            index[i] = int(item[i])  # index为对应特征值的在该属性阈的index
        for i in [9, 10, 12, 14]:  # category feature 位置, 对第4（OCCP）、5（POBP）属性进行编码
            value[i] = 1
            index[i] = (vocab_dic[i][item[i]][0])
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
        data_index.append(index)
        data_values.append(value)
        data_label.append(label)
    numpy.save("../dataset/ACS/travel/data/{}_i.npy".format(file_name), data_index)
    numpy.save("../dataset/ACS/travel/data/{}_v.npy".format(file_name), data_values)
    numpy.save("../dataset/ACS/travel/data/{}_y.npy".format(file_name), data_label)

    min_max = {}
    for j in range(16):
        min_max[j] = [100000, 0]
    for j in range(len(data_values)):
        item = data_values[j]
        for hh in range(len(item)):  # 获取value取值的最小值、最大值
            if int(item[hh]) < min_max[hh][0]:
                min_max[hh][0] = int(item[hh])
            if int(item[hh]) > min_max[hh][1]:
                min_max[hh][1] = int(item[hh])
    with open("../dataset/ACS/travel/data/min_max.txt", 'w') as f:
        write_min_max(f, min_max)
    f.close()


def ACSTravel_data_augmentation_race(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            aug_data = data[i].tolist()
            if race != aug_data[8]:
                aug_data[8] = race
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def ACSTravel_data_augmentation_gender(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    genders = [1, 2]
    for i in range(data.shape[0]):
        data_list = []
        for gender in genders:
            aug_data = data[i].tolist()
            if gender != aug_data[3]:
                aug_data[3] = gender
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def ACSTravel_data_augmentation_multiple(data_file, save_file):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    genders = [1, 2]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            for gender in genders:
                aug_data = data[i].tolist()
                if race != aug_data[8] or gender != aug_data[3]:
                    aug_data[8] = race
                    aug_data[3] = gender
                    data_list.append(aug_data)
        aug.append(data_list)
    return numpy.save(save_file, aug)


def reCode_ACSTravel_data_similar(data_file, file_name):
    """
    对原始数据进行重新编码
    :return:
    """
    aug_data = numpy.load(data_file)
    with open("../dataset/ACS/travel/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)
    fea_dim = numpy.load("../dataset/ACS/travel/data/fea_dim.npy").tolist()
    aug_data_values = []
    aug_data_index = []
    aug_data_label = []
    for x in range(aug_data.shape[1]):
        # 对数据进行重新编码
        raw_data = aug_data[:, x, :]
        data_values = []
        data_index = []
        data_label = []
        for j in range(raw_data.shape[0]):
            item = raw_data[j]
            value = [0] * 16
            index = [None] * 16
            label = item[16]
            for i in [0, 15]:  # numerical feature 位置，取原值，空值取0
                value[i] = int(item[i])
                index[i] = fea_dim[i]
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 11, 13]:  # category feature 位置, 非空值取1，空值取0
                value[i] = 1
                index[i] = int(item[i])  # index为对应特征值的在该属性阈的index
            for i in [9, 10, 12, 14]:  # category feature 位置, 对第4（OCCP）、5（POBP）属性进行编码
                value[i] = 1
                index[i] = (vocab_dic[i][item[i]][0])
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
                index[i] += fea_dim[i - 1]  # index为对应特征值的在该属性阈的index + 该属性阈起始编码位置
            data_index.append(index)
            data_values.append(value)
            data_label.append(label)
        aug_data_values.append(data_values)
        aug_data_index.append(data_index)
        aug_data_label.append(data_label)
    numpy.save("../dataset/ACS/travel/data/{}_i.npy".format(file_name), aug_data_index)
    numpy.save("../dataset/ACS/travel/data/{}_v.npy".format(file_name), aug_data_values)
    numpy.save("../dataset/ACS/travel/data/{}_y.npy".format(file_name), aug_data_label)
