import pickle
import re

import numpy
import pandas
import random
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import wordnet


# NLP data
def replace_synonyms(template, synonyms):
    """
    同义词替换
    :param template:
    :param synonyms:
    :return:
    """
    # 按空格分割模板，但保留{}完整
    words = template.split(" ")
    # 检查各个单词是否存在同义词，存在则50%的概率进行同义词替换
    for i in range(len(words)):
        word = words[i].lower().strip()
        if word in synonyms.keys():
            if random.random() < 0.5:  # 50%概率替换
                replacement = random.choice(synonyms[word])
                words[i] = replacement
                # print(replacement)
    return " ".join(words)


def get_replace_synonyms(word):
    """

    :return:
    """
    synsets = wordnet.synsets(word)  # 获取所有同义词集
    synonyms = set(lemma.name() for syn in synsets for lemma in syn.lemmas())
    return synonyms


# NLP adult dataset
def split_NLP_adult_data(train_file, test_file):
    """
    将adult进行划分
    :return:
    """
    adult_df = pandas.read_csv("../dataset/NLP/adult/data/adult.data")
    raw_data = adult_df[['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation',
                         'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                         'native-country', 'income']].values
    numpy.random.shuffle(raw_data)
    tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    pandas.DataFrame(tran_data).to_csv(train_file, index=False)
    pandas.DataFrame(test_data).to_csv(test_file, index=False)


def adult_text_data_augmentation_race(data_file, save_tabular):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = ['White', 'Black', 'AsianPacIslander', 'AmerIndianEskimo', 'Other']
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            aug_data = data[i].tolist()
            if race != aug_data[7].replace("-", "").strip():
                aug_data[7] = race
                data_list.append(aug_data)
        aug.append(data_list)
    numpy.save(save_tabular, aug)


def adult_text_data_augmentation_gender(data_file, save_tabular):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    genders = ["Male", "Female"]
    for i in range(data.shape[0]):
        data_list = []
        for gender in genders:
            aug_data = data[i].tolist()
            if gender != aug_data[8].replace("-", "").strip():
                aug_data[8] = gender
                data_list.append(aug_data)
        aug.append(data_list)
    numpy.save(save_tabular, aug)


def adult_text_data_augmentation_multiple(data_file, save_tabular):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = ['White', 'Black', 'AsianPacIslander', 'AmerIndianEskimo', 'Other']
    genders = ["Male", "Female"]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            for gender in genders:
                aug_data = data[i].tolist()
                if race != aug_data[7].replace("-", "").strip() or gender != aug_data[8].strip():
                    aug_data[7] = race
                    aug_data[8] = gender
                    data_list.append(aug_data)
        aug.append(data_list)
    numpy.save(save_tabular, aug)


def adult_tabular_to_text(template, item):
    """
    将adult数据集的表格数据转化为文本数据
    :return:
    """
    adult_text = template.format(item[0], item[7].replace("-", ""), item[8], item[4], item[12], item[6], item[2],
                                 item[3], item[5], item[2], item[11], item[9], item[10])

    if adult_text.find("She/He") > -1:
        if adult_text.find("Female") > 0:
            adult_text = adult_text.replace("She/He", "She").replace("Her/His", "Her")
        else:
            adult_text = adult_text.replace("She/He", "He").replace("Her/His", "His")
    return adult_text


def generate_adult_text_data(data_file, race_file, gender_file, aug_file,
                             text_data_file, text_label_file, text_race_file, text_gender_file, text_aug_file):
    """
    使用template模板将adult表格数据转换为文本text数据
    :return:
    """
    # 定义模板
    templates = [
        "The individual is a {}-year-old {} {} , {} and residing in {} as the {} of household . They hold a {} degree ("
        " {} years of formal education) and currently work as a {} in the {} sector , typically logging {} hours "
        "weekly . Their financial profile include a capital gain of {} and loss of {} .",
        "I am a {}-year-old {} {} , {} and residing in {} as the {} of household . I hold a {} degree ( {} years of "
        "formal education ) and currently work as a {} in the {} sector , typically logging {} hours weekly . My "
        "financial profile include a capital gain of {} and loss of {} .",
        "You are a {}-year-old {} {} , {} and residing in {} as the {} of household . You hold a {} degree ( {} years "
        "of formal education ) and currently work as a {} in the {} sector , typically logging {} hours weekly . Your "
        "financial profile include a capital gain of {} and loss of {} .",
        "She/He is a {}-year-old {} {} , {} and residing in {} as the {} of household . She/He holds a {} degree ( {} "
        "years of formal education ) and currently works as a {} in the {} sector , typically logging {} hours weekly "
        ".  Her/His financial profile includes a capital gain of {} and loss of {} ."
    ]

    # 定义同义词词典
    synonyms = {
        "individual": ["person", "subject", "resident"],
        "residing": ["living", "domiciled"],
        "hold": ["possess", "have earned"],
        "holds": ["possesses", "has earned"],
        "education": ["schooling", "academic training"],
        "work": ["are employed", "serve"],
        "works": ["is employed", "serves"],
        "sector": ["industry", "field"],
        "logging": ["working", "putting in"],
        "weekly": ["per week"],
        "financial": ["fiscal", "economic"],
        "include": ["comprise", "encompass"],
        "includes": ["comprises", "encompasses"],
        "capital": ["investment", "financial", "asset"],
        "gain": ["profit"],
        "loss": ["deficit"]
    }
    tabular_data = pandas.read_csv(data_file).values
    tabular_race = numpy.load(race_file)
    tabular_gender = numpy.load(gender_file)
    tabular_aug = numpy.load(aug_file)

    text_adult = []
    text_race = []
    text_gender = []
    text_aug = []
    text_label = []
    for i in range(tabular_data.shape[0]):
        # 随机选择一个模板，替换同义词
        selected_template = random.choice(templates)
        result_template = replace_synonyms(selected_template, synonyms)
        # 对表格数据进行转换
        text_adult.append(adult_tabular_to_text(result_template, tabular_data[i]))
        # 记录标签
        if tabular_data[i][13].find(">50K") > -1:
            text_label.append(1)
        else:
            text_label.append(0)
        # 对race数据进行转换
        sim_race = tabular_race[i]
        race_result = []
        for j in range(len(sim_race)):
            race_result.append(adult_tabular_to_text(result_template, sim_race[j]))
        text_race.append(race_result)
        # 对gender数据进行转换
        sim_gender = tabular_gender[i]
        gender_result = []
        for j in range(len(sim_gender)):
            gender_result.append(adult_tabular_to_text(result_template, sim_gender[j]))
        text_gender.append(gender_result)
        # 对aug数据进行转换
        sim_aug = tabular_aug[i]
        aug_result = []
        for j in range(len(sim_aug)):
            aug_result.append(adult_tabular_to_text(result_template, sim_aug[j]))
        text_aug.append(aug_result)

    numpy.save(text_data_file, text_adult)
    numpy.save(text_label_file, text_label)
    numpy.save(text_race_file, text_race)
    numpy.save(text_gender_file, text_gender)
    numpy.save(text_aug_file, text_aug)

    # numpy.random.shuffle(raw_data)
    # tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    # test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    # pandas.DataFrame(tran_data).to_csv("../dataset/NLP/adult/data/train_data.csv", index=False)
    # pandas.DataFrame(test_data).to_csv("../dataset/NLP/adult/data/test_data.csv", index=False)

    # text_train_adult = []
    # text_train_label = []
    # for i in range(tran_data.shape[0]):
    #     text_train_adult.append(
    #         template.format(tran_data[i][0], tran_data[i][7].replace("-", ""), tran_data[i][8], tran_data[i][4],
    #                         tran_data[i][12], tran_data[i][6], tran_data[i][2], tran_data[i][3], tran_data[i][5],
    #                         tran_data[i][2], tran_data[i][11], tran_data[i][9], tran_data[i][10]))
    #     if tran_data[i][13].find(">50K") > 0:
    #         text_train_label.append(1)
    #     else:
    #         text_train_label.append(0)
    # numpy.save("../dataset/NLP/adult/data/text_train_adult.npy", text_train_adult)
    # numpy.save("../dataset/NLP/adult/data/text_train_label.npy", text_train_label)
    #
    # text_test_adult = []
    # text_test_label = []
    # for i in range(test_data.shape[0]):
    #     text_test_adult.append(
    #         template.format(test_data[i][0], test_data[i][7].replace("-", ""), test_data[i][8], test_data[i][4],
    #                         test_data[i][12], test_data[i][6], test_data[i][2], test_data[i][3], test_data[i][5],
    #                         test_data[i][2], test_data[i][11], test_data[i][9], test_data[i][10]))
    #     if test_data[i][13].find(">50K") > 0:
    #         text_test_label.append(1)
    #     else:
    #         text_test_label.append(0)
    # numpy.save("../dataset/NLP/adult/data/text_test_adult.npy", text_test_adult)
    # numpy.save("../dataset/NLP/adult/data/text_test_label.npy", text_test_label)


def generate_adult_adv_replace_synonyms():
    """
    生成adult数据集中所有文字的同义词替换，包括模板中的单词同义词，以及所有属性取值的同义词
    :return:
    """
    # adv_synonyms_dic = {}
    # # 定义模板
    # templates = [
    #     "The individual is a {}-year-old {} {} , {} and residing in {} as the {} of household . They hold a {} degree ("
    #     " {} years of formal education) and currently work as a {} in the {} sector , typically logging {} hours "
    #     "weekly . Their financial profile include a capital gain of {} and loss of {} .",
    #     "I am a {}-year-old {} {} , {} and residing in {} as the {} of household . I hold a {} degree ( {} years of "
    #     "formal education ) and currently work as a {} in the {} sector , typically logging {} hours weekly . My "
    #     "financial profile include a capital gain of {} and loss of {} .",
    #     "You are a {}-year-old {} {} , {} and residing in {} as the {} of household . You hold a {} degree ( {} years "
    #     "of formal education ) and currently work as a {} in the {} sector , typically logging {} hours weekly . Your "
    #     "financial profile include a capital gain of {} and loss of {} .",
    #     "She/He is a {}-year-old {} {} , {} and residing in {} as the {} of household . She/He holds a {} degree ( {} "
    #     "years of formal education ) and currently works as a {} in the {} sector , typically logging {} hours weekly "
    #     ".  Her/His financial profile includes a capital gain of {} and loss of {} ."
    # ]
    # for t in templates:
    #     t = re.sub(r'\{.*?\}', '', t)
    #     t = t.replace('-', ' ')
    #     t = re.sub(r'[^a-zA-Z\s]', '', t)
    #     words = t.split()
    #     for w in words:
    #         w_synonyms = get_replace_synonyms(w)
    #         if len(w_synonyms) > 0:
    #             if w not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
    #                 adv_synonyms_dic[w] = w_synonyms
    #
    # # 定义同义词词典
    # synonyms = {
    #     "individual": ["person", "subject", "resident"], "residing": ["living", "domiciled"],
    #     "hold": ["possess", "have earned"], "holds": ["possesses", "has earned"],
    #     "education": ["schooling", "academic training"], "work": ["are employed", "serve"],
    #     "works": ["is employed", "serves"], "sector": ["industry", "field"], "logging": ["working", "putting in"],
    #     "weekly": ["per week"], "financial": ["fiscal", "economic"], "include": ["comprise", "encompass"],
    #     "includes": ["comprises", "encompasses"], "capital": ["investment", "financial", "asset"], "gain": ["profit"],
    #     "loss": ["deficit"]
    # }
    # for k in synonyms:
    #     k_s = synonyms[k]
    #     for s_k in k_s:
    #         w_synonyms = get_replace_synonyms(s_k)
    #         if len(w_synonyms) > 0:
    #             if s_k not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
    #                 adv_synonyms_dic[s_k] = w_synonyms
    #
    # # 所有属性的取值
    # vocab_dic = ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay',
    #              'Never-worked', 'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm',
    #              'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th',
    #              'Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated',
    #              'Married-AF-spouse', 'Widowed', 'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
    #              'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing',
    #              'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv',
    #              'Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative', 'White', 'Black',
    #              'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Male', 'Female', 'United-States', 'Cuba',
    #              'Jamaica', 'India', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany',
    #              'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos',
    #              'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'China',
    #              'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece',
    #              'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands']
    # for v in vocab_dic:
    #     for vv in v.strip().split("-"):
    #
    #         w_synonyms = get_replace_synonyms(vv)
    #         if len(w_synonyms) > 0:
    #             if vv not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
    #                 adv_synonyms_dic[vv] = w_synonyms

    # 对生成的同义词字典进行人工选择后，选取部分恰当的单词
    adv_synonyms_dic = {'individual': {'somebody', 'someone', 'individual', 'subject', 'person'}, 'is': {'was', 'be'},
                        'a': {'a', 'A'}, 'year': {'year', 'yr', 'twelvemonth'}, 'old': {'old', 'older', 'Old'},
                        'residing': {'domicile', 'occupy', 'occupying', 'reside', 'lodge_in', 'domiciliate', 'live',
                                     'living'}, 'in': {'In', 'IN', 'in'}, 'as': {'as', 'a', 'As', 'AS'},
                        'household': {'household', 'family', 'house', 'home'},
                        'hold': {'hold_up', 'cargo_hold', 'have_got', 'take', 'carry', 'retain', 'hold_in', 'take_for',
                                 'hold', 'keep', 'contain', 'go_for', 'have', 'obtain', 'maintain', 'take_hold',
                                 'admit'}, 'degree': {'level', 'academic_degree', 'stage', 'degree', },
                        'years': {'days', 'years', 'twelvemonth', 'year', 'yr'},
                        'education': {'school', 'schooling', 'training', 'educational_activity', 'Education',
                                      'Education_Department', 'education', 'teaching', 'Department_of_Education'},
                        'currently': {'presently', 'currently'},
                        'work': {'serve', 'act', 'operate', 'body_of_work', 'make_for', 'play', 'put_to_work',
                                 'do_work', 'work', 'piece_of_work', 'study', 'function', 'employment', 'process',
                                 'act_upon', 'work_on', 'work_out', 'serving', 'acting', 'operating', 'playing',
                                 'working', 'studying', 'functioning', 'employmenting', 'processing'},
                        'hours': {'hour', 'minute', '60_minutes', 'hours', 'hr'},
                        'weekly': {'every_week', 'each_week', 'weekly'},
                        'capital': {'working_capital', 'Capital', 'capital'},
                        'gain': {'profit', 'earn', 'gain_ground', 'addition', 'benefit', 'take_in', 'gain', 'attain',
                                 'increase'}, 'I': {'1', 'I', 'i'}, 'am': {'AM', 'MA', 'Am', 'be'},
                        'holds': {'hold_up', 'cargo_hold', 'have_got', 'take', 'carry', 'retain', 'hold_in', 'take_for',
                                  'hold', 'keep', 'contain', 'go_for', 'have', 'obtain', 'maintain', 'take_hold',
                                  'admit'},
                        'works': {'act', 'operate', 'body_of_work', 'make_for', 'play', 'put_to_work', 'do_work',
                                  'work', 'piece_of_work', 'study', 'function', 'employment', 'process', 'act_upon',
                                  'work_on', 'work_out'}, 'includes': {'include', 'admit'},
                        'person': {'somebody', 'someone', 'individual', 'subject', 'person'},
                        'subject': {'somebody', 'someone', 'individual', 'subject', 'person'},
                        'resident': {'resident_physician', 'occupier', 'resident', 'house_physician', 'occupant'},
                        'living': {'domicile', 'occupy', 'occupying', 'reside', 'lodge_in', 'domiciliate', 'live',
                                   'living'},
                        'domiciled': {'domicile', 'occupy', 'occupying', 'reside', 'lodge_in', 'domiciliate', 'live',
                                      'living'},
                        'schooling': {'school', 'schooling', 'training', 'educational_activity', 'Education',
                                      'Education_Department', 'education', 'teaching', 'Department_of_Education'},
                        'serve': {'serve', 'act', 'operate', 'body_of_work', 'make_for', 'play', 'put_to_work',
                                  'do_work', 'work', 'piece_of_work', 'study', 'function', 'employment', 'process',
                                  'act_upon', 'work_on', 'work_out', 'serving', 'acting', 'operating', 'playing',
                                  'working', 'studying', 'functioning', 'employmenting', 'processing'},
                        'serves': {'serve', 'act', 'operate', 'body_of_work', 'make_for', 'play', 'put_to_work',
                                   'do_work', 'work', 'piece_of_work', 'study', 'function', 'employment', 'process',
                                   'act_upon', 'work_on', 'work_out', 'serving', 'acting', 'operating', 'playing',
                                   'working', 'studying', 'functioning', 'employmenting', 'processing'},
                        'industry': {'industry', 'industriousness', 'manufacture', 'diligence'},
                        'field': {'force_field', 'area', 'subject_field', 'field', 'field_of_operation', 'subject_area',
                                  'flying_field', 'discipline', 'field_of_honor', 'subject', 'field_of_study', 'study',
                                  'domain'},
                        'working': {'serve', 'act', 'operate', 'body_of_work', 'make_for', 'play', 'put_to_work',
                                    'do_work', 'work', 'piece_of_work', 'study', 'function', 'employment', 'process',
                                    'act_upon', 'work_on', 'work_out', 'serving', 'acting', 'operating', 'playing',
                                    'working', 'studying', 'functioning', 'employmenting', 'processing'},
                        'fiscal': {'fiscal', 'financial'}, 'economic': {'economic', 'economical'},
                        'comprise': {'comprise', 'represent', 'constitute', 'contain', 'be', 'consist', 'make_up',
                                     'incorporate'},
                        'encompass': {'embrace', 'encompass', 'comprehend', 'cover'},
                        'comprises': {'comprise', 'represent', 'constitute', 'contain', 'be', 'consist', 'make_up',
                                      'incorporate'},
                        'encompasses': {'embrace', 'encompass', 'comprehend', 'cover'},
                        'investment': {'investing', 'investment', 'investiture', 'investment_funds'},
                        'asset': {'plus', 'asset'},
                        'profit': {'profits', 'net_profit', 'net', 'benefit', 'profit', 'earnings', 'turn_a_profit',
                                   'net_income', 'gain', 'lucre'},
                        'deficit': {'shortage', 'deficit', 'shortfall'},
                        'State': {'State', 'province', 'nation', 'Department_of_State',
                                  'United_States_Department_of_State', 'state',
                                  'commonwealth', 'State_Department'}, 'not': {'not', 'non'},
                        'Private': {'secret', 'private', 'individual', 'buck_private'},
                        'Federal': {'Federal_soldier', 'Fed', 'Federal', 'federal_official', 'Union', 'federal'},
                        'pay': {'remuneration', 'compensate', 'earnings', 'pay', 'bear', 'pay_off', 'wage', 'make_up',
                                'yield',
                                'pay_up',
                                'salary', 'ante_up', 'give'},
                        'worked': {'work', 'run', 'solve', 'work_out', 'function', 'make' 'work_on', 'act', 'do_work',
                                   'figure_out',
                                   'operate'}, 'HS': {'Hs', 'h', 'H'}, 'grad': {'grade', 'graduate', 'grad'},
                        '11th': {'11th', 'eleventh'},
                        '9th': {'9th', 'ninth'}, '7th': {'7th', 'seventh'}, '8th': {'8th', 'eighth'},
                        'Doctorate': {"doctor's_degree", 'doctorate'}, 'Prof': {'prof', 'professor'},
                        'school': {'shoal', 'cultivate', 'schooling', 'train', 'schooltime', 'civilise', 'school_day',
                                   'school',
                                   'educate',
                                   'civilize', 'schoolhouse'}, '5th': {'fifth', '5th'}, '6th': {'sixth', '6th'},
                        '10th': {'tenth', '10th'}, '1st': {'1st', 'first'}, '4th': {'quaternary', '4th', 'fourth'},
                        '12th': {'12th', 'twelfth'},
                        'married': {'marital', 'married', 'matrimonial', 'conjoin', 'marry', 'espouse', 'tie', 'wed',
                                    'get_married'},
                        'Married': {'marital', 'married', 'matrimonial', 'conjoin', 'marry', 'espouse', 'tie', 'wed',
                                    'get_married'},
                        'spouse': {'mate', 'better_half', 'married_person', 'spouse', 'partner'},
                        'Divorced': {'disunite', 'split_up', 'divorce', 'disjoint', 'disassociate', 'divorced',
                                     'dissociate'},
                        'absent': {'missing', 'wanting', 'lacking', 'remove', 'absent'},
                        'Separated': {'set_apart', 'isolated', 'dislocated', 'tell_apart', 'break_up', 'detached',
                                      'separated',
                                      'split', 'break', 'single_out', 'separate'},
                        'Handlers': {'animal_trainer', 'handler', 'manager', 'coach'},
                        'cleaners': {'cleansing_agent', 'cleanser', 'dry_cleaners', 'dry_cleaner', 'cleaners',
                                     'cleaner'},
                        'specialty': {'metier', 'speciality', 'strong_suit', 'peculiarity', 'forte', 'specialisation',
                                      'specialization', 'strength', 'strong_point', 'specialness', 'long_suit',
                                      'distinctiveness',
                                      'specialism', 'specialty'},
                        'service': {'table_service', 'avail', 'military_service', 'divine_service', 'Service',
                                    'religious_service', 'inspection_and_repair', 'Robert_William_Service', 'serve',
                                    'armed_service',
                                    'servicing', 'service', 'overhaul', 'serving', 'help', 'service_of_process'},
                        'Sales': {'sales_event', 'sales_agreement', 'sales', 'gross_revenue', 'gross_sales', 'sale',
                                  'cut_rate_sale'},
                        'Craft': {'craftiness', 'craftsmanship', 'craft', 'workmanship'},
                        'repair': {'fix', 'touch_on', 'amend', 'fixing', 'stamping_ground', 'revivify', 'resort',
                                   'recreate', 'rectify',
                                   'animate', 'recompense', 'revive', 'reparation', 'mending', 'restore', 'repair',
                                   'renovate',
                                   'furbish_up', 'hangout', 'remediate'},
                        'Transport': {'transfer', 'transport', 'transmit', 'shipping', 'ship', 'send', 'conveyance',
                                      'transportation',
                                      'transferral'},
                        'moving': {'move', 'prompt', 'travel', 'go', 'affect', 'make_a_motion', 'act', 'propel',
                                   'be_active', 'moving'},
                        'Farming': {'land', 'agricultural', 'farm', 'farming', 'agriculture', 'agrarian', 'produce',
                                    'husbandry'},
                        'fishing': {'fishing', 'angle', 'fish', 'sportfishing'},
                        'Machine': {'motorcar', 'car', 'simple_machine', 'automobile', 'machine', 'political_machine',
                                    'auto'},
                        'Tech': {'technical_school', 'tech'},
                        'support': {'financial_support', 'support', 'underpin', 'back', 'patronise', 'put_up',
                                    'backing', 'funding',
                                    'supporting', 'stick_out', 'indorse', 'sustenance'},
                        'Armed': {'armed', 'arm', 'fortify', 'gird'},
                        'Forces': {'force', 'military_group', 'power', 'military_force', 'violence', 'military_unit'},
                        'house': {'theater', 'mansion', 'home', 'family', 'house', 'firm', 'household', 'domiciliate',
                                  'sign'},
                        'family': {'family_line', 'household', 'family', 'house', 'family_unit', 'home'},
                        'Husband': {'hubby', 'husband'}, 'Wife': {'married_woman', 'wife'},
                        'Own': {'have', 'possess', 'own', 'ain'},
                        'child': {'tike', 'baby', 'kid', 'child', 'nipper', 'youngster', 'small_fry', 'shaver',
                                  'minor'},
                        'Unmarried': {'single', 'unmarried'},
                        'White': {'whiten', 'White_person', 'white', 'White', 'Caucasian'},
                        'Black': {'blackamoor', 'Black_person', 'black', 'blackness', 'Negro'},
                        'Asian': {'Asian', 'Asiatic'},
                        'Pac': {'PAC', 'political_action_committee'},
                        'Islander': {'islander', 'island_dweller'},
                        'Indian': {'Amerindic', 'Red_Indian', 'Native_American', 'Amerind', 'American_Indian_language',
                                   'American_Indian',
                                   'Indian', 'Amerindian_language'}, 'Eskimo': {'Eskimo', 'Inuit', 'Esquimau'},
                        'Male': {'virile', 'Male', 'manly', 'manlike', 'male_person', 'male', 'manful'},
                        'Female': {'distaff', 'female', 'female_person'},
                        'States': {'State', 'province', 'nation', 'Department_of_State',
                                   'United_States_Department_of_State', 'state',
                                   'commonwealth', 'State_Department'},
                        'Cuba': {'Cuba', 'Republic_of_Cuba'},
                        'India': {'Bharat', 'Republic_of_India', 'India'},
                        'Mexico': {'United_Mexican_States', 'Mexico'},
                        'South': {'due_south', 'southward', 'in_the_south', 'to_the_south', 'S', 'south', 'South', },
                        'Rico': {'anti_racketeering_law', 'RICO_Act', 'RICO',
                                 'Racketeer_Influenced_and_Corrupt_Organizations_Act'},
                        'Honduras': {'Republic_of_Honduras', 'Honduras'},
                        'Germany': {'Federal_Republic_of_Germany', 'FRG', 'Germany', 'Deutschland'},
                        'Iran': {'Iran', 'Islamic_Republic_of_Iran', 'Persia'},
                        'Philippines': {'Philippines', 'Philippine_Islands', 'Philippine',
                                        'Republic_of_the_Philippines', 'Filipino'},
                        'Italy': {'Italy', 'Italia', 'Italian_Republic'},
                        'Poland': {'Republic_of_Poland', 'Polska', 'Poland'},
                        'Columbia': {'Columbia', 'Columbia_University', 'Columbia_River', 'capital_of_South_Carolina'},
                        'Cambodia': {'Kampuchea', 'Kingdom_of_Cambodia', 'Cambodia'},
                        'Thailand': {'Kingdom_of_Thailand', 'Thailand', 'Siam'},
                        'Ecuador': {'Ecuador', 'Republic_of_Ecuador'},
                        'Laos': {'Laotian', "Lao_People's_Democratic_Republic", 'Lao', 'Laos'},
                        'Taiwan': {'China', 'Republic_of_China', 'Formosa', 'Taiwan'},
                        'Haiti': {'Hayti', 'Republic_of_Haiti', 'Haiti', 'Hispaniola'},
                        'Portugal': {'Portugal', 'Portuguese_Republic'},
                        'Dominican': {'friar_preacher', 'Dominican', 'Black_Friar', 'Blackfriar'},
                        'Republic': {'commonwealth', 'republic', 'democracy'},
                        'Salvador': {'Republic_of_El_Salvador', 'El_Salvador', 'Salvador'},
                        'France': {'Anatole_France', 'France', 'Jacques_Anatole_Francois_Thibault', 'French_Republic'},
                        'Guatemala': {'Republic_of_Guatemala', 'Guatemala'},
                        'China': {"People's_Republic_of_China", 'Communist_China', 'china', 'China', 'Red_China', 'PRC',
                                  'Republic_of_China', 'Nationalist_China', 'Taiwan', 'chinaware', 'mainland_China'},
                        'Japan': {'Japanese_Archipelago', 'Japanese_Islands', 'Japan', 'Nihon', 'japan', 'Nippon'},
                        'Yugoslavia': {'Serbia_and_Montenegro', 'Union_of_Serbia_and_Montenegro', 'Yugoslavia',
                                       'Federal_Republic_of_Yugoslavia', 'Jugoslavija'},
                        'Peru': {'Republic_of_Peru', 'Peru'},
                        'Greece': {'Greece', 'Hellenic_Republic', 'Ellas'},
                        'Nicaragua': {'Nicaragua', 'Republic_of_Nicaragua'},
                        'Vietnam': {'Vietnam', 'Viet_Nam', 'Socialist_Republic_of_Vietnam', 'Vietnam_War', 'Annam'},
                        'Ireland': {'Eire', 'Republic_of_Ireland', 'Ireland', 'Irish_Republic', 'Hibernia',
                                    'Emerald_Isle'},
                        'Hungary': {'Republic_of_Hungary', 'Hungary', 'Magyarorszag'},
                        'Netherlands': {'Holland', 'Nederland', 'The_Netherlands', 'Kingdom_of_The_Netherlands',
                                        'Netherlands'}}
    with open("../dataset/NLP/adult/data/adv_synonyms_dic.pkl", 'wb') as f:
        pickle.dump(adv_synonyms_dic, f)
    adv_synonyms_text = []
    for k in adv_synonyms_dic:
        # if k not in adv_synonyms_text:
        #     adv_synonyms_text.append(k)
        for v in adv_synonyms_dic[k]:
            if v not in adv_synonyms_text:
                adv_synonyms_text.append(v)
    return [" ".join(adv_synonyms_text)]


# bank dataset
def split_NLP_bank_data(train_file, test_file):
    """
    将bank进行划分
    :return:
    """
    raw_data = pandas.read_csv("../dataset/NLP/bank/data/data.csv").values
    numpy.random.shuffle(raw_data)
    tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    pandas.DataFrame(tran_data).to_csv(train_file, index=False)
    pandas.DataFrame(test_data).to_csv(test_file, index=False)


def bank_text_data_augmentation_age(data_file, save_tabular):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    ages = ["youth", "middle-age", "the old"]
    for i in range(data.shape[0]):
        data_list = []
        for age in ages:
            aug_data = data[i].tolist()
            if age != aug_data[0]:
                aug_data[0] = age
                data_list.append(aug_data)
        aug.append(data_list)
    numpy.save(save_tabular, aug)


def bank_tabular_to_text(template, item):
    """
    将bank数据集的表格数据转化为文本数据
    :return:
    """
    bank_text = template.format(item[0].replace("-", ""), item[2], item[1], item[3], item[4], item[5], item[6],
                                item[7], item[12], item[8], item[10], item[9], item[11], item[14], item[13], item[15])

    return bank_text


def generate_bank_text_data(data_file, age_file, text_data_file, text_label_file, text_age_file):
    """
    使用template模板将bank表格数据转换为文本text数据
    :return:
    """
    # 定义模板
    templates = [
        "The individual is a {} {} individual , employed as a {} with a {} degree . Their credit defaults record as {"
        "} , annual financial balances as € {} , housing loans as {} , and personal loan  as  {} . During the current "
        "campaign , They contacted the bank {} times , most recently via  {} on  {}  {} ( duration :  {} seconds ) . "
        "In previous campaigns , They contacted  {} times  {} days ago , getting a  {} outcome .",
        "I am a {} {} individual , employed as a {} with a {} degree . My credit defaults record as {} , "
        "annual financial balances as € {} , housing loans as {} , and personal loan as  {} . During the current "
        "campaign , I contacted the bank {} times , most recently via  {} on  {}  {} ( duration :  {} seconds ) . In "
        "previous campaigns , I contacted  {} times  {} days ago , getting a  {} outcome .",
        "You are a {} {} individual , employed as a {} with a {} degree . Your credit defaults record as {} , "
        "annual financial balances as € {} , housing loans as {} , and personal loan as  {} . During the current "
        "campaign , You contacted the bank {} times , most recently via  {} on  {}  {} ( duration :  {} seconds ) . In "
        "previous campaigns , You contacted  {} times  {} days ago , getting a  {} outcome .",
        "She is a {} {} individual , employed as a {} with a {} degree . Her credit defaults record as {} , "
        "annual financial balances as € {} , housing loans as {} , and personal loan as  {} . During the current "
        "campaign , She contacted the bank {} times , most recently via  {} on  {}  {} ( duration :  {} seconds ) . In "
        "previous campaigns , She contacted  {} times  {} days ago , getting a  {} outcome .",
        "He is a {} {} individual , employed as a {} with a {} degree . His credit defaults record as {} , "
        "annual financial balances as € {} , housing loans as {} ,  and personal loan as  {} . During the current "
        "campaign , He contacted the bank {} times , most recently via  {} on  {}  {} ( duration :  {} seconds ) . In "
        "previous campaigns , He contacted  {} times  {} days ago , getting a  {} outcome ."
    ]

    # 定义同义词词典
    synonyms = {
        "individual": ["person", "subject", "resident"],
        "employed": ["hired", "worked", "engaged"],
        "degree": ["qualification", "diploma", "certification", 'grade', 'level', 'academic_degree'],
        "credit": ["loan", "payment", "debt"],
        "defaults": ["failures", "breaches", "nonpayment"],
        "annual": ["Year_end", "yearly", 'yearbook'],
        "balances": ["amounts", "summaries", 'residual', 'residue'],
        "housing": ["residential", "property", "home", "domiciliate", 'house'],
        "loans": ["mortgage", "credit", "borrowing", 'loan', 'lend', 'loanword'],
        "personal": ["individual", "private", "consumer"],
        "current": ["present", "ongoing", "active", "latest"],
        "campaign": ["promotion", 'movement'],
        "bank": ["institution", "lender", "credit_union"],
        "recently": ["lately", "newly", "freshly"],
        "contacted": ["reached_out", "called", "phoned"],
        "via": ["through", "using", "over", "on", "by"],
        "previous": ["prior", "former", "earlier", "preceding", "last"],
        "getting": ["obtaining", "receiving", "acquiring"],
        "outcome": ['final_result', 'consequence', "effect", "conclusion", "decision", 'event', 'effect', 'termination',
                    ' result', 'resultant']
    }
    tabular_data = pandas.read_csv(data_file).values
    tabular_age = numpy.load(age_file)

    text_bank = []
    text_age = []
    text_label = []
    for i in range(tabular_data.shape[0]):
        # 随机选择一个模板，替换同义词
        selected_template = random.choice(templates)
        result_template = replace_synonyms(selected_template, synonyms)
        # 对表格数据进行转换
        text_bank.append(bank_tabular_to_text(result_template, tabular_data[i]))
        # 记录标签
        if tabular_data[i][16].find("yes") > -1:
            text_label.append(1)
        else:
            text_label.append(0)
        # 对age数据进行转换
        sim_age = tabular_age[i]
        age_result = []
        for j in range(len(sim_age)):
            age_result.append(bank_tabular_to_text(result_template, sim_age[j]))
        text_age.append(age_result)

    numpy.save(text_data_file, text_bank)
    numpy.save(text_label_file, text_label)
    numpy.save(text_age_file, text_age)


def generate_bank_adv_replace_synonyms():
    """
    生成bank数据集中所有文字的同义词替换，包括模板中的单词同义词，以及所有属性取值的同义词
    :return:
    """
    # adv_synonyms_dic = {}
    # # 定义模板
    # templates = [
    #     "The individual is a {} {} individual , employed as a {} with a {} degree . Their credit defaults record as {"
    #     "} , annual financial balances as € {} , housing loans as {} , and personal loan  as  {} . During the current "
    #     "campaign , They contacted the bank {} times , most recently via  {} on  {}  {} ( duration :  {} seconds ) . "
    #     "In previous campaigns , They contacted  {} times  {} days ago , getting a  {} outcome .",
    #     "I am a {} {} individual , employed as a {} with a {} degree . My credit defaults record as {} , "
    #     "annual financial balances as € {} , housing loans as {} , and personal loan as  {} . During the current "
    #     "campaign , I contacted the bank {} times , most recently via  {} on  {}  {} ( duration :  {} seconds ) . In "
    #     "previous campaigns , I contacted  {} times  {} days ago , getting a  {} outcome .",
    #     "You are a {} {} individual , employed as a {} with a {} degree . Your credit defaults record as {} , "
    #     "annual financial balances as € {} , housing loans as {} , and personal loan as  {} . During the current "
    #     "campaign , You contacted the bank {} times , most recently via  {} on  {}  {} ( duration :  {} seconds ) . In "
    #     "previous campaigns , You contacted  {} times  {} days ago , getting a  {} outcome .",
    #     "She is a {} {} individual , employed as a {} with a {} degree . Her credit defaults record as {} , "
    #     "annual financial balances as € {} , housing loans as {} , and personal loan as  {} . During the current "
    #     "campaign , She contacted the bank {} times , most recently via  {} on  {}  {} ( duration :  {} seconds ) . In "
    #     "previous campaigns , She contacted  {} times  {} days ago , getting a  {} outcome .",
    #     "He is a {} {} individual , employed as a {} with a {} degree . His credit defaults record as {} , "
    #     "annual financial balances as € {} , housing loans as {} ,  and personal loan as  {} . During the current "
    #     "campaign , He contacted the bank {} times , most recently via  {} on  {}  {} ( duration :  {} seconds ) . In "
    #     "previous campaigns , He contacted  {} times  {} days ago , getting a  {} outcome ."
    # ]
    # for t in templates:
    #     t = re.sub(r'\{.*?\}', '', t)
    #     t = t.replace('-', ' ')
    #     t = re.sub(r'[^a-zA-Z\s]', '', t)
    #     words = t.split()
    #     for w in words:
    #         w_synonyms = get_replace_synonyms(w)
    #         if len(w_synonyms) > 0:
    #             if w not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
    #                 adv_synonyms_dic[w] = w_synonyms
    #
    # # 定义同义词词典
    # synonyms = {
    #     "individual": ["person", "subject", "resident"],
    #     "employed": ["hired", "worked", "engaged", "holding a position"],
    #     "degree": ["qualification", "diploma", "certification", "academic title", 'grade', 'level', 'academic_degree'],
    #     "credit": ["loan", "payment", "debt"],
    #     "defaults": ["failures", "breaches", "nonpayment"],
    #     "annual": ["Year_end", "yearly", "12-month ", 'one-year', 'yearbook'],
    #     "balances": ["amounts", "summaries", 'residual', 'residue'],
    #     "housing": ["residential", "property", "real_estate", "home", "domiciliate", 'house'],
    #     "loans": ["mortgage", "credit", "borrowing", 'loan', 'lend', 'loanword'],
    #     "personal": ["individual", "private", "consumer"],
    #     "current": ["present", "ongoing", "active", "latest"],
    #     "campaign": ["promotion", 'movement'],
    #     "bank": ["institution", "lender", "credit_union"],
    #     "recently": ["lately", "newly", "freshly"],
    #     "contacted": ["reached_out", "called", "phoned"],
    #     "via": ["through", "using", "over", "on", "by"],
    #     "previous": ["prior", "former", "earlier", "preceding", "last"],
    #     "getting": ["obtaining", "receiving", "acquiring"],
    #     "outcome": ['final_result', 'consequence', "effect", "conclusion", "decision", 'event', 'effect', 'termination',
    #                 ' result', 'resultant']
    # }
    # for k in synonyms:
    #     k_s = synonyms[k]
    #     for s_k in k_s:
    #         w_synonyms = get_replace_synonyms(s_k)
    #         if len(w_synonyms) > 0:
    #             if s_k not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
    #                 adv_synonyms_dic[s_k] = w_synonyms
    #
    # # 所有属性的取值
    # vocab_dic = ['middle-age'  'youth', 'the old', 'management', 'technician', 'entrepreneur', 'blue-collar', 'unknown',
    #              'retired', 'admin', 'services', 'self-employed', 'unemployed', 'housemaid', 'student', 'married',
    #              'single', 'divorced', 'tertiary', 'secondary', 'unknown', 'primary', 'no', 'yes', 'unknown',
    #              'cellular', 'telephone', "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
    #              "dec", 'unknown', 'failure', 'other', 'success']
    # for v in vocab_dic:
    #     for vv in v.strip().split("-"):
    #
    #         w_synonyms = get_replace_synonyms(vv)
    #         if len(w_synonyms) > 0:
    #             if vv not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
    #                 adv_synonyms_dic[vv] = w_synonyms

    # 对生成的同义词字典进行人工选择后，选取部分恰当的单词
    adv_synonyms_dic = {'individual': {'somebody', 'someone', 'individual', 'subject', 'person', "resident"},
                        'is': {'was', 'be'}, 'a': {'a', 'A'},
                        "employed": ["hired", "worked", "engaged"], 'as': {'as', 'a', 'As', 'AS'},
                        "degree": {"qualification", "diploma", "certification", 'grade', 'level',
                                   'academic_degree'}, "credit": ["loan", "payment", "debt"],
                        "defaults": ["failures", "breaches", "nonpayment"],
                        'record': {'phonograph_record', 'register', 'show', 'record', 'memorialize', 'put_down'},
                        "annual": {"Year_end", "yearly", 'yearbook'},
                        'financial': {'financial', 'fiscal'},
                        "balances": {"amounts", "summaries", 'residual', 'residue'},
                        "housing": {"residential", "property", "home", "domiciliate", 'house'},
                        'loans': {"mortgages", "credits", "borrowings", 'loan', 'lends', 'loanwords'},
                        'personal': {"individual", "private", "consumer"},
                        'loan': {"mortgage", "credit", "borrowing", 'loan', 'lend', 'loanword'},
                        'current': {"present", "ongoing", "active", "latest"},
                        'contacted': {"reached_out", "called", "phoned", 'get_through', 'contact'},
                        'bank': {"institution", "lender", "credit_union"},
                        'recently': {"lately", "newly", "freshly" 'recently', 'late'}, 'on': {'along', 'on'},
                        'duration': {'duration', 'continuance', 'length'}, 'seconds': {'sec', 's', 'second'},
                        'In': {'In', 'IN', 'in'}, 'previous': {"prior", "former", "earlier", "preceding", "last"},
                        'campaign': {"promotions", 'movements'},
                        'days': {'daytime', 'Day', 'twenty_four_hours', '24_hour_interval', 'days',
                                 'twenty_four_hour_period', 'day', 'daylight'},
                        'getting': {"obtaining", "receiving", "acquiring"},
                        'outcome': {'final_result', 'consequence', "effect", "conclusion", "decision", 'event',
                                    'effect', 'outcome', 'termination', ' result', 'resultant'}, 'I': {'1', 'I', 'i'},
                        'am': {'AM', 'MA', 'Am', 'be'}, 'are': {'be', 'ar', 'are'}, 'He': {'He', 'he'},
                        "via": {"through", "using", "over", "on", "by"},
                        'unknown': {'obscure', 'unidentified', 'unknown', 'strange', 'unknown_region',
                                    'unknown_quantity', 'stranger'},
                        'services': {'serving', 'table_service', 'service', 'services', 'armed_service', 'Service',
                                     'serve', 'military_service', 'servicing'},
                        'unemployed': {'unemployed', 'unemployed_people'},
                        'housemaid': {'amah', 'housemaid', 'maid', 'maidservant'},
                        'student': {'student', 'scholar', 'pupil', 'scholarly_person', 'educatee', 'bookman'},
                        'married': {'marital', 'married', 'matrimonial', 'conjoin', 'marry', 'espouse', 'tie', 'wed',
                                    'get_married'}, 'single': {'individual', 'single', 'unmarried', '1', 'one', 'I'},
                        'divorced': {'disunite', 'split_up', 'divorce', 'disjoint', 'disassociate', 'divorced',
                                     'dissociate'},
                        'tertiary': {'3rd', 'Tertiary_period', 'Tertiary', 'tertiary', 'third'},
                        'secondary': {'junior_grade', 'lower_ranking', 'lowly', 'petty', 'secondary_winding',
                                      'secondary', 'subaltern', "2th", 'secondary_coil'},
                        'primary': {'elemental', 'primary', 'primary_election', 'primary_coil', 'elementary', 'chief',
                                    'main', 'basal', 'primary_winding', 'principal', 'primary_quill', 'master',
                                    'primary_feather'},
                        'telephone': {'ring', 'call', 'telephone', 'call_up', 'phone', 'telephony', 'telephone_set'},
                        'jan': {'January', 'Jan'}, 'feb': {'Feb', 'February'}, 'mar': {'March', 'Mar', 'mar'},
                        'apr': {'Apr', 'April'}, 'may': {'may', 'May'}, 'aug': {'Aug', 'August'},
                        'sep': {'Sept', 'September', 'Sep'}, 'oct': {'Oct', 'October'}, 'nov': {'Nov', 'November'},
                        'dec': {'dec', 'Dec', 'December'}, 'failure': {'failure', 'unsuccessful'},
                        'success': {'approved', 'success'}}
    with open("../dataset/NLP/bank/data/adv_synonyms_dic.pkl", 'wb') as f:
        pickle.dump(adv_synonyms_dic, f)
    adv_synonyms_text = []
    for k in adv_synonyms_dic:
        # if k not in adv_synonyms_text:
        #     adv_synonyms_text.append(k)
        for v in adv_synonyms_dic[k]:
            if v not in adv_synonyms_text:
                adv_synonyms_text.append(v)
    return [" ".join(adv_synonyms_text)]


# COMPAS dataset
def split_NLP_compas_data(train_file, test_file):
    """
    将compas进行划分
    :return:
    """
    raw_data = pandas.read_csv("../dataset/NLP/compas/data/data.csv").values
    # numpy.random.shuffle(raw_data)
    # tran_data = raw_data[:int(raw_data.shape[0] * 0.8), ]
    # test_data = raw_data[int(raw_data.shape[0] * 0.8):, ]
    # pandas.DataFrame(tran_data).to_csv(train_file, index=False)
    # pandas.DataFrame(test_data).to_csv(test_file, index=False)

    words = set()
    for h in range(raw_data.shape[0]):
        item = raw_data[h]
        for hh in [0, 1, 2, 5, 6, 8, 9, 11, 12, 14, 16]:  # category feature 位置
            for hhh in str(item[hh]).replace("/", " ").split(" "):  # 特征值item[i]是否在字典feature_dictionary[i] 中
                # 特征值不在，为字典feature_dictionary[i]增加key值item[i]，value值初始化为len(vocab_dic[i])+1
                words.add(hhh)
    with open("../dataset/NLP/bank/data/word_list.txt", 'w') as f:
        f.write(",".join(f'"{d}"' for d in words) + "\n")
    f.close()


def compas_text_data_augmentation_race(data_file, save_tabular):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = ['Other', 'African-American', 'Caucasian', 'Hispanic', 'Native American', 'Asian']
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            aug_data = data[i].tolist()
            if race != aug_data[2].strip():
                aug_data[2] = race
                data_list.append(aug_data)
        aug.append(data_list)
    numpy.save(save_tabular, aug)


def compas_text_data_augmentation_gender(data_file, save_tabular):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    genders = ["Male", "Female"]
    for i in range(data.shape[0]):
        data_list = []
        for gender in genders:
            aug_data = data[i].tolist()
            if gender != aug_data[0].strip():
                aug_data[0] = gender
                data_list.append(aug_data)
        aug.append(data_list)
    numpy.save(save_tabular, aug)


def compas_text_data_augmentation_multiple(data_file, save_tabular):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本。
    保护属性定义为种族与性别
    :return:
    """
    data = pandas.read_csv(data_file).values
    aug = []
    races = ['Other', 'African-American', 'Caucasian', 'Hispanic', 'Native American', 'Asian']
    genders = ["Male", "Female"]
    for i in range(data.shape[0]):
        data_list = []
        for race in races:
            for gender in genders:
                aug_data = data[i].tolist()
                if race != aug_data[2].strip() or gender != aug_data[0].strip():
                    aug_data[2] = race
                    aug_data[0] = gender
                    data_list.append(aug_data)
        aug.append(data_list)
    numpy.save(save_tabular, aug)


def select_c_charge_degree(c_charge_degree):
    """

    :return:
    """
    if c_charge_degree == "F":
        return "felony"
    elif c_charge_degree == "M":
        return "misdemeanor"


def select_did_did_not(Tag):
    """

    :return:
    """
    if Tag == 1:
        return "did"
    else:
        return "did not"


def compas_tabular_to_text(template, item):
    """
    将compas数据集的表格数据转化为文本数据
    :return:
    """
    compas_text = template.format(item[1], item[2].replace("-", ""), item[0], item[4], select_c_charge_degree(item[5]),
                                  item[6], item[3], item[14], select_did_did_not(item[7]), item[8], item[9],
                                  select_did_did_not(item[10]), item[11], item[12], item[15], item[16])

    if compas_text.find("She/He") > -1:
        if compas_text.find("Female") > 0:
            compas_text = compas_text.replace("She/He", "She").replace("Her/His", "Her")
        else:
            compas_text = compas_text.replace("She/He", "He").replace("Her/His", "His")

    return compas_text


def generate_compas_text_data(data_file, race_file, gender_file, aug_file,
                              text_data_file, text_label_file, text_race_file, text_gender_file, text_aug_file):
    """
    使用template模板将compas表格数据转换为文本text数据
    :return:
    """
    # 定义模板
    templates = [
        "The individual is a {} years old {} {} . They have {} prior offense(s) and are charged with a {} degree ({}) ."
        "Their COMPAS general recidivism risk score is {} ( {} risk ) . Recidivism records : They {} recidivate , "
        "with a {} charge degree for {} . Violence records : They {} commit violent recidivism , with a {} charge for "
        "{} . Their violent recidivism risk score is {} ({} risk) .",
        "I am a {} years old {} {} . I have {} prior offense(s) and am charged with a {} degree ({}) ."
        "My COMPAS general recidivism risk score is {} ( {} risk ) . Recidivism records : I {} recidivate , "
        "with a {} charge degree for {} . Violence records : I {} commit violent recidivism , with a {} charge for "
        "{} . My violent recidivism risk score is {} ({} risk) .",
        "You are a {} years old {} {} . You have {} prior offense(s) and are charged with a {} degree ({}) ."
        "Your COMPAS general recidivism risk score is {} ( {} risk ) . Recidivism records : You {} recidivate , "
        "with a {} charge degree for {} . Violence records : You {} commit violent recidivism , with a {} charge for "
        "{} . Your violent recidivism risk score is {} ({} risk) .",
        "She/He is a {} years old {} {} . She/He has {} prior offense(s) and is charged with a {} degree ({}) ."
        "Her/His COMPAS general recidivism risk score is {} ( {} risk ) . Recidivism records : She/He {} recidivate , "
        "with a {} charge degree for {} . Violence records : She/He {} commit violent recidivism , with a {} charge "
        "for {} . Her/His violent recidivism risk score is {} ({} risk) .",
    ]
    # 定义同义词词典
    synonyms = {
        "individual": ["person", "subject", "resident", "defendant"],
        "is a": ["is an", "presents as a", "identifies as a"],
        "prior": ["previous", "earlier", "past"],
        "offense(s)": ["conviction(s)", "crime(s)", "charge(s)", "criminal_act(s)"],
        "charge": ["accuse", "face", "prosecute", "indict", "offense"],
        "charged": ["accused", "facing", "prosecuted", "indicted", "offended"],
        "degree": ["category", "level", "class", "severity"],
        "recidivism_risk": ["risk_assessment", "recidivism_prediction", "risk_classification"],
        "risk": ["probability", "likelihood", "propensity"],
        "records": ["history", "background", "documentation"],
        "recidivate": ["reoffend", "commit_another_crime", "relapse_into_crime"],
        "recidivism": ["reoffending", "crimes_again"],
        "commit": ["engage", "perpetrate"],
    }
    tabular_data = pandas.read_csv(data_file).values
    tabular_race = numpy.load(race_file)
    tabular_gender = numpy.load(gender_file)
    tabular_aug = numpy.load(aug_file)

    text_compas = []
    text_race = []
    text_gender = []
    text_aug = []
    text_label = tabular_data[:, 18]
    for i in range(tabular_data.shape[0]):
        # 随机选择一个模板，替换同义词
        selected_template = random.choice(templates)
        result_template = replace_synonyms(selected_template, synonyms)
        # 对表格数据进行转换
        text_compas.append(compas_tabular_to_text(result_template, tabular_data[i]))
        # 对race数据进行转换
        sim_race = tabular_race[i]
        race_result = []
        for j in range(len(sim_race)):
            race_result.append(compas_tabular_to_text(result_template, sim_race[j]))
        text_race.append(race_result)
        # 对gender数据进行转换
        sim_gender = tabular_gender[i]
        gender_result = []
        for j in range(len(sim_gender)):
            gender_result.append(compas_tabular_to_text(result_template, sim_gender[j]))
        text_gender.append(gender_result)
        # 对aug数据进行转换
        sim_aug = tabular_aug[i]
        aug_result = []
        for j in range(len(sim_aug)):
            aug_result.append(compas_tabular_to_text(result_template, sim_aug[j]))
        text_aug.append(aug_result)

    numpy.save(text_data_file, text_compas)
    numpy.save(text_label_file, text_label)
    numpy.save(text_race_file, text_race)
    numpy.save(text_gender_file, text_gender)
    numpy.save(text_aug_file, text_aug)


def generate_compas_adv_replace_synonyms():
    """
    生成compas数据集中所有文字的同义词替换，包括模板中的单词同义词，以及所有属性取值的同义词
    :return:
    """
    adv_synonyms_dic = {}
    # # 定义模板
    # templates = [
    #     "The individual is a {} years old {} {} . They have {} prior offense(s) and are charged with a {} degree ({}) ."
    #     "Their COMPAS general recidivism risk score is {} ( {} risk ) . Recidivism records : They {} recidivate , "
    #     "with a {} charge degree for {} . Violence records : They {} commit violent recidivism , with a {} charge for "
    #     "{} . Their violent recidivism risk score is {} ({} risk) .",
    #     "I am a {} years old {} {} . I have {} prior offense(s) and am charged with a {} degree ({}) ."
    #     "My COMPAS general recidivism risk score is {} ( {} risk ) . Recidivism records : I {} recidivate , "
    #     "with a {} charge degree for {} . Violence records : I {} commit violent recidivism , with a {} charge for "
    #     "{} . My violent recidivism risk score is {} ({} risk) .",
    #     "You are a {} years old {} {} . You have {} prior offense(s) and are charged with a {} degree ({}) ."
    #     "Your COMPAS general recidivism risk score is {} ( {} risk ) . Recidivism records : You {} recidivate , "
    #     "with a {} charge degree for {} . Violence records : You {} commit violent recidivism , with a {} charge for "
    #     "{} . Your violent recidivism risk score is {} ({} risk) .",
    #     "She/He is a {} years old {} {} . She/He has {} prior offense(s) and is charged with a {} degree ({}) ."
    #     "Her/His COMPAS general recidivism risk score is {} ( {} risk ) . Recidivism records : She/He {} recidivate , "
    #     "with a {} charge degree for {} . Violence records : She/He {} commit violent recidivism , with a {} charge "
    #     "for {} . Her/His violent recidivism risk score is {} ({} risk) .",
    # ]
    # for t in templates:
    #     t = re.sub(r'\{.*?\}', '', t)
    #     t = t.replace('-', ' ')
    #     t = re.sub(r'[^a-zA-Z\s]', '', t)
    #     words = t.split()
    #     for w in words:
    #         w_synonyms = get_replace_synonyms(w)
    #         if len(w_synonyms) > 0:
    #             if w not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
    #                 adv_synonyms_dic[w] = w_synonyms
    #
    # # 定义同义词词典
    # synonyms = {
    #     "individual": ["person", "subject", "resident", "defendant"],
    #     "is a": ["is an", "presents as a", "identifies as a"],
    #     "prior": ["previous", "earlier", "past"],
    #     "offense(s)": ["conviction(s)", "crime(s)", "charge(s)", "criminal_act(s)"],
    #     "charge": ["accuse", "face", "prosecute", "indict", "offense"],
    #     "charged": ["accused", "facing", "prosecuted", "indicted", "offended"],
    #     "degree": ["category", "level", "class", "severity"],
    #     "recidivism_risk": ["risk_assessment", "recidivism_prediction", "risk_classification"],
    #     "risk": ["probability", "likelihood", "propensity"],
    #     "records": ["history", "background", "documentation"],
    #     "recidivate": ["reoffend", "commit_another_crime", "relapse_into_crime"],
    #     "recidivism": ["reoffending", "crimes_again"],
    #     "commit": ["engage", "perpetrate"],
    # }
    # for k in synonyms:
    #     k_s = synonyms[k]
    #     for s_k in k_s:
    #         w_synonyms = get_replace_synonyms(s_k)
    #         if len(w_synonyms) > 0:
    #             if s_k not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
    #                 adv_synonyms_dic[s_k] = w_synonyms
    #
    # # 所有属性的取值

    vocab_dic = ["Chur", "Above", "Fabricating", "Attach", "Motor", "Dwelling", "Reg", "Parking", "2ND", "Temporary",
                 "Firearm", "Ped", "Reg", "Bank", "Canceled", "Resp", "Fraud", "More", "Attnd", "The", "Resisting",
                 "Threat", "Establishm", "Committing", "Renting", "dead", "MDMA", "Crlty", "is", "Batt", "Corrupt",
                 "Elec", "Scene", "Vict", "Struc", "Burg", "Lewd", "Threatening", "Card", "Arm", "Sub", "Alch", "Jail",
                 "DOC", "Paying", "DNA", "P", "Bills", "Paraphernalia", "Voyeurism", "Disqul", "Veh", "Harm", "Int",
                 "Oxycodone", "Extinquisher", "Issue", "First", "Issuing", "Food", "Fire", "Strong", "Aggrav",
                 "Molestation", "Domestic", "Minor", "Felo", "Pawn", "Opert", "Supply", "Degree", "Convey", "Tag",
                 "struct", "Faml", "Fact", "Possession", "Vi", "Alt", "Contribute", "Of", "Defendants", "Info",
                 "Unsafe", "Forge", "Exploit", "Manslaughter", "Performance", "Destroy", "Exhib", "unocc", "Device",
                 "Poss3,4", "Adult", "Dom", "Park", "Conduct", "Obey", "Abuse", "Torture", "Punish", "Trafficking",
                 "Accd", "Licenc", "Countrfeit", "Prostitu", "Breath", "Ring", "Strang", "Legal", "Emergency", "Schd",
                 "Battery", "Revk", "Fel", "Support", "Grounds", "Structure", "Dealing", "Assembly", "Consume", "Agg",
                 "Equip", "Than", "Dwell", "Heroin", "Petit", "Occp", "Drug", "Property", "Animals", "Hydrocodone",
                 "Improper", "Possess", "Restraining", "Child", "Perm", "Burglary", "Drinking", "Transactions", "Words",
                 "Protect", "deliver", "Possession", "Revoke", "Phentermine", "Cause", "FIELD", "Dead", "Trespassing",
                 "Residence", "Intent", "Methylethcathinone", "Another", "Person", "to", "Assign", "pur", "Crim",
                 "Impersonating", "Beverages", "Accessory", "Alprazolam", "Leave", "Suspended", "Evidence", "Suspend",
                 "Firearm", "Court", "Disabled", "Cust", "Speed", "Tresspass", "Site", "Enhanced", "Leaving", "Police",
                 "Trespass", "Obstruct", "Verif", "Church", "Substance", "Misuse", "Yrs", "Comply", "Item", "Harass",
                 "Leo", "Butylone", "Store", "Private", "Busn", "Months", "Assault", "Criminal", "Premises",
                 "Information", "Attempted", "Manage", "Leas", "Robbery", "Dwel", "Soliciting", "Load", "Agree",
                 "Return", "Plant", "Promis", "Invalid", "For", "Subst", "Contempt", "Fail", "Non", "Methado",
                 "Steroid", "Ride", "Native", "Scanning", "Id", "(Aggravated)", "Dev", "Bribery", "Financial",
                 "Assignation", "5-Fluoro", "Cancel", "Plates", "Uncov", "Retaliate", "Traf", "Interfere", "Horses",
                 "Tetrahydrocannabinols", "Dom", "Inst", "Und", "Anabolic", "Counterfeit", "Ecstasy", "level", "Depriv",
                 "School", "Methylenediox", "Delinquency", "Offer", "Crimin", "Use", "Witness", "Permit", "Yr", "Order",
                 "From", "Sample", "Unlaw", "near", "Offense", "Sound", "Arson", "sell", "Giving", "Servant",
                 "Ownership", "Danger", "of", "Fac", "Statement", "High", "Drv", "Lewdness", "Lasciv", "Wit",
                 "Restrictions", "Serious", "at", "African-American", "Conspire", "Pub", "PL", "than", "Wireless",
                 "Revoked", "Athletic", "Contract", "Solicitation", "Oth", "Officer", "Obtain", "Loitering", "Instui",
                 "Fighting", "Obstuct", "Bod", "Open", "Aid", "Purchasing", "Metal", "Sleeping", "Failure", "Damage",
                 "Intrf", "Girlfriend", "Utter", "Influence", "Cab", "Eng", "Instrument", "Discharge", "Violation-FL",
                 "Homicide", "Man", "Disable", "Aggr", "Low", "Sch", "Material", "Posses", "Deliver", "ID",
                 "Contraband", "Frd", "Contests", "3rd", "Bomb", "Destruct", "Delinq", "Great", "Tools", "Damage",
                 "Uttering", "Drivg", "Acc", "Test", "Anti-Shoplifting", "Pyrrolidinovalerophenone", "Accident",
                 "Hispanic", "Abuse-Causes", "Beach", "Personating", "Benzylpiperazine", "Boater", "Culpable", "Money",
                 "Carisoprodol", "Prowling", "Violation", "Submit", "Bev", "Habit", "Damg", "Fish", "Aggress", "With",
                 "Unocc", "Deprive", "Buprenorphine", "Aggravated", "Injunction", "Grams", "Lease", "Check", "Physical",
                 "Att", "battery", "Conviction", "Ethylone", "Tri-Rail", "Way", "Custody", "Copper", "Bylaw", "with",
                 "Vehicle", "Conveyance", "Para", "Etc", "Injury", "Unemployment", "Scen", "NO", "Address",
                 "Structuring", "Education", "Equipment", "Present", "RR", "During", "Reckless", "Twrd", "Introduce",
                 "Credit", "Deg", "Grt", "Second", "Intoxicating", "City", "Motorcycle", "3", "Struct", "Priv",
                 "Vehicle", "Scho", "or", "nan", "Cash", "Container", "Arrest", "Anoth", "Duties", "Mandatory",
                 "Building", "Function", "Elderlly", "Interference", "Male", "Fuel", "Lascivious", "Product", "Blood",
                 "Malic", "Grow", "Purchase", "1st", ">16", "18+", "ID", "Solict", "XLR11", "Against", "$300", "Abuse",
                 "Carrying", "Sounds>1000", "3,4Methylenediox", "Actual", "Charge", "Requirements", "Mot", "MARK", "a",
                 "Principal", "Throw", "Carry", "2-Methox", "Dang", "Sol", "Less", "Unauthorized", "Display", "Removed",
                 "Extradition", "arrest", "Caucasian", "Wholesale", "vict", "Licensed", "Estab", "Diazepam",
                 "Contractor", "Exhibition", "Construction", "F", "Sale", "Dols", "License", "Purpose", "Mischief",
                 "Vin", "Compensatn", "Codeine", "Unlawful", "Tampering", "Valid", "Exposes", "int", "Prot",
                 "Pyrrolidinobutiophenone", "CI", "Amphetamine", "Change", "Fighter", "w", "Grand", "65", "Disrupting",
                 "Prop", "Threaten", "D", "Taxi", "Intoxicated", "Phone", "Attmp", "3)", "Retail", "Snatch", "Oper",
                 "Emplyee", "Over", "Career", "DWLS", "Forging", "Disply", "Family", "Telemarketing", "Dome", "On",
                 "Theft", "Pornography", "Interf", "Unl", "Commission", "Earnings", "Deadly", "Diox", "I", "Com",
                 "Indecent", "Launder", "2nd", "Release", "Mask", "ID#", "Conve", "Shp", "Contr", "Driving", "Motor",
                 "Inj", "Where", "Stolen", "grams", "Bat", "Kidnapping", "After", "Dating", "Meth", "Intoxication",
                 "Offens", "Innkeeper", "Act", "Struct", "Dw", "Toward", "Victim", "Merchant", "Control", "LicTag",
                 "Upon", "ProstitutionViolation", "No", "on", "Cntrft", "Disobey", "Conspiracy", "Lic", "Neglect",
                 "Cocaine", "3,4", "3Rd", "Tamper", "DL", "Greater", "Unauth", "Posted", "Methylenedioxymethcath",
                 "Take", "Utility", "Enter", "Convict", "Key", "Commit", "Viol", "Abuse-Agg", "Drivers", "PB-22",
                 "Subs", "Enforcement", "Amp", "Rcpt", "Call", "Prostitution", "Speci", "Medium", "(F1)", "Asian",
                 "+$150", "0.20", "Simulation", "Pur", "LEO-Agg", "another", "Hired", "Deliv", "the", "Hrm-Deadly",
                 "by", "Manufacture", "Procure", "Escape", "Persnl", "Lost", "At", "Disorderly", "and", "Prior",
                 "Eluding", "Bus", "Invasion", "Resist", "Secure", "Driver", "Draft", "Poss", "Tobacco", "Offend",
                 "Pers", "Hrs", "Trifluoromethylphenylpipe", "Prescript", "Abet", "(MDMA)", "Landing",
                 "Methamphetamine", "Insur", "Lve", "Fireman", "(Facilitate", "Del", "Fleeing", "Public", "Contrft",
                 "Unlicensed", "(F7)", "LSD", "Live", "Register", "Cards", "Payment", "Cred", "charge", "Fraudulent",
                 "-", "Loiter", "Redeliver", "Sell", "Forged", "Trans", "Home", "Wearing", "Elder", "Pregnant",
                 "1a,1b,1d,2a,2b", "Felony)", "Robbery-Strong", "Victm", "200-400", "Fare", "Persn", "REGISTERED",
                 "$1000+", "Bodily", "(F3)", "Consideration", "Refuse", "1,4-Butanediol", "Operating", "Substa",
                 "Audio", "Care", "Informnt", "Warning", "Enforc", "Identity", "12+", "Shoot", "RX", "Panhandle",
                 "Sticker", "Detainee", "Carjacking", "Into", "Attempt", "Stalking", "Process", "Drink", "Vehicular",
                 "Resident", "Littering", "Concealed", "BOX", "A", "2", "in", "Goods", "Off", "Ma", "Prohibited",
                 "Traff", "Flee", "Attend", "Note", "Morphine", "Sel", "Fentanyl", "Traffic", "Report", "Unnatural",
                 "While", "Spouse", "Weapon", "DWI", "Solicit", "Intellectual", "case", "<16yr", "Pos", "Sexual",
                 "Delivery", "Falsely", "Mfg", "Hire", "+", "14g><28g", "Presence", "In", "Molest", "Reports",
                 "Habitual", "II", "Unauthorizd", "Cyberstalking", "Other", "Solic", "Expired", "Racing", "Mfr",
                 "Transport", "Worthless", "Repeat", "Baiting", "Alcoholic", "Conterfeit", "28><200", "Cruelty",
                 "Prescription", "WITH", "Safety", "Prson", "Armed", "Injunctn", "JWH-250", "Exposure", "(F2)", "Harm)",
                 "Hydromorphone", "Others", "Proof", "Wep", "Res", "Depnd", "DUI", "Articles", "Older", "Defrauding",
                 "O", "Computer", "Defraud", "Compulsory", "Conv", "American", "Crime", "Lodging", "Sign", "Altered",
                 "Amm", "Personal", "Felon", "Prostitute", "Elude", "Cannabis", "Kil", "Pretrial", "Contradict",
                 "Burgl", "Alter", "o", "Aide", "Cont", "<16", "(motor", "Sudd", "Offender", "(Vehicle)", "IC",
                 "Negligence", "21", "Suspd", "LEO", "Occupy", "Controlled", "Or", "Gamb", "Drugs", "B", "4-14", "Sex",
                 "Vessel", "Badges", "Farm", "Alcohol", "Imperson", "Invest", "Consp", "Wear", "Elderly", "Disturb",
                 "Traffick", "Occup", "firearm", "no", "Monitor", "911", "1000FT", "Disguise", "while", "Deceased",
                 "Similitude", "Handcuff", "Strike", "Beg", "Amobarbital", "Beverage", "Can", "16-", "Insurance",
                 "DUI-", "Drive", "Gambling", "Weap", "Convic", "Female", "Level", "Deft", "Missile", "Aiding", "for",
                 "Clonazepam", "Pay", "Under", "To", "Juvenile", "Railroad", "Lorazepam", "Utilizing", "Driv", "Injunc",
                 "Occupied", "#", "Felony", "Unoccupied", "Murder", "Methadone", "Enfor", "Highway", "Urge", "Unoccup",
                 "Injunct", "Injury", "Law", "Without", "Communic", "POSSESS", "mask", "Imprisonment", "Redeliv",
                 "Hiring", "Near", "Violence", "Unattended", "Lasc", "(Firearm)", "Dealer", "Illegal", "System",
                 "Wildlife", "Attendance", "Provide", "Offn", "Susp", "Video", "1-Pentyl", "Min", "Engage", "False",
                 "Name"
                 ]
    for v in vocab_dic:
        for vv in v.strip().split("-"):

            w_synonyms = get_replace_synonyms(vv)
            if len(w_synonyms) > 0:
                if vv not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
                    adv_synonyms_dic[vv] = w_synonyms

    print()
    with open("../dataset/NLP/compas/data/word_list.txt", 'w') as f:
        for key, value in adv_synonyms_dic.items():
            f.write("'{}':[".format(key) + ",".join(f'"{d}"' for d in value) + "],\n")
    f.close()

    # 对生成的同义词字典进行人工选择后，选取部分恰当的单词

    adv_synonyms_dic = {
        "individual": ["person", "subject", "resident", "defendant"],
        "is": ["presents", "identifies"],
        "a": ["an", "a", "A"],
        "prior": ["previous", "earlier", "past"],
        "offense(s)": ["conviction(s)", "crime(s)", "charge(s)", "criminal_act(s)"],
        "charge": ["accuse", "face", "prosecute", "indict", "offense"],
        "charged": ["accused", "facing", "prosecuted", "indicted", "offended"],
        "degree": ["category", "level", "class", "severity"],
        "recidivism_risk": ["risk_assessment", "recidivism_prediction", "risk_classification"],
        "risk": ["probability", "likelihood", "propensity"],
        "records": ["history", "background", "documentation"],
        "recidivate": ["reoffend", "commit_another_crime", "relapse_into_crime"],
        "recidivism": ["reoffending", "crimes_again"],
        "commit": ["engage", "perpetrate"],
        'Above': ["higher_up", "in_a_higher_place", "above"],
        'Fabricating': ["fabricate", "manufacture", "invent"],
        'Attach': ["bind", "attach", "tie", "bond"],
        'Motor': ["drive", "motor"],
        'Dwelling': ["habitation", "home", "dwelling_house", "domicile", "dwell", "dwelling", "live", "lie_in"],
        'Parking': ["parking", "park"],
        '2ND': ["2d", "second", "2nd"],
        'Temporary': ["irregular", "impermanent", "temporary", "temp"],
        'Bank': ["depository_financial_institution", "bank", "deposit", "banking_concern", "bank_building",
                 "banking_company", "coin_bank", "savings_bank"],
        'Canceled': ["cancel", "offset", "delete"],
        'Fraud': ["fake", "fraud", "pseud", "dupery"],
        'More': ["more_than", "more", "More", "Thomas_More"],
        'Resisting': ["reject", "jib", "protest", "defy", "dissent", "resist"],
        'Threat': ["menace", "threat", "terror"],
        'Renting': ["rent", "rental", "renting"],
        'dead': ["deadened", "dead"],
        'Corrupt': ["defile", "sully", "demoralise", "bribe", "deprave", "corrupted", "tainted", "corrupt"],
        'Scene': ["scene", "scenery"],
        'Threatening': ["imperil", "menacing", "menace", "threaten", "threatening", "endanger"],
        'Arm': ["weapon_system", "arm", "weapon"],
        'Jail': ["jailhouse", "jail", "incarcerate", "immure", "gaol"],
        'Paying': ["pay_up", "ante_up", "paying", "pay", "give", "paid", "pay_off"],
        'Harm': ["injury", "harm", "hurt", "damage"],
        'Issue': ["matter", "issuance" "event", "Issuing", "issue", "consequence"],
        'First': ["1st", "start", "number_one", "firstly", "first", "beginning", "number_1"],
        'Fire': ["fire", "burn", "firing", "fuel", "burn_down"],
        'Strong': ["hard", "strong", "firm", "solid"],
        'Minor': ["small_scale", "small", "minor"],
        'Supply': ["provide", "supplying", "supply"],
        'Degree': ["grade", "degree", "level"],
        'Tag': ["tag", "label", "mark"],
        'Possession': ["possession", "self_possession"],
        'Contribute': ["put_up", "contribute", "add"],
        'Info': ["info", "information"],
        'Unsafe': ["unsafe", "dangerous", "insecure"],
        'Destroy': ["ruin", "destruct", "destroy", "put_down", "demolish"],
        'Park': ["parking_area", "car_park", "park", "Park", "parkland"],
        'Abuse': ["ill-usage", "misuse", "abuse", "ill-use"],
        'Ring': ["telephone", "ringing", "ring"],
        'Battery': ["electric_battery", "battery"],
        'Assembly': ["forum", "assembly", "assemblage", "gathering", "meeting_place", "fabrication"],
        'Consume': ["take", "consume", "go_through", "take_in", "ingest", "run_through", "ware", "eat", "squander",
                    "exhaust", "deplete", "have", "devour", "wipe_out", "use_up", "waste", "down", "eat_up"],
        'Equip': ["equip", "fit_out", "outfit", "fit"],
        'Dwell': ["lie", "brood", "populate", "consist", "dwell", "harp", "inhabit", "live", "lie_in"],
        'Heroin': ["diacetylmorphine", "heroin"],
        'Drug': ["do_drugs", "drug", "dose"],
        'Property': ["belongings", "prop", "property", "attribute", "place", "holding", "dimension"],
        'Animals': ["animate_being", "animal", "brute", "creature", "fauna", "beast"],
        'Improper': ["improper", "wrong", "unlawful", "unconventional"],
        'Possess': ["have", "possess", "own"],
        'Restraining': ["restrict", "limit", "bound", "cumber", "constrain", "encumber", "confine", "keep", "restrain",
                        "trammel", "throttle", "hold", "keep_back", "intimidate", "hold_back"],
        'Child': ["child", "tiddler", "kid", "nestling", "small_fry", "shaver", "baby", "minor", "youngster", "tike",
                  "fry",
                  "nipper", "tyke"],
        'Perm': ["permanent", "Molotov", "Perm", "permanent_wave", "perm"],
        'Burglary': ["burglary"],
        'Drinking': ["wassail", "crapulence", "salute", "boozing", "drink_in", "toast", "booze", "imbibition", "drink",
                     "drunkenness", "imbibe", "fuddle", "imbibing", "pledge", "tope", "drinking"],
        'Transactions': ["dealings", "transaction", "dealing", "transactions", "proceedings", "minutes"],
        'Words': ["give_voice", "actor's_line", "give-and-take", "quarrel", "Word", "word_of_honor", "lyric", "tidings",
                  "Christian_Bible", "watchword", "Good_Book", "Holy_Scripture", "Bible", "phrase", "wrangle",
                  "discussion",
                  "speech", "row", "articulate", "intelligence", "Son", "Logos", "password", "dustup", "news",
                  "language",
                  "formulate", "countersign", "run-in", "word", "words", "parole", "Word_of_God", "Scripture",
                  "Holy_Writ",
                  "Book"],
        'Protect': ["protect"],
        'deliver': ["bear", "cede", "extradite", "rescue", "save", "drive_home", "turn_in", "give_birth", "deliver",
                    "surrender", "fork_over", "have", "redeem", "hand_over", "render", "give_up", "deport", "fork_up",
                    "birth", "fork_out", "pitch", "return", "present"],
        'Revoke': ["annul", "countermand", "reverse", "overturn", "vacate", "repeal", "rescind", "lift", "revoke",
                   "renege"],
        'Cause': ["cause", "stimulate", "causal_agency", "causal_agent", "crusade", "suit", "effort", "movement",
                  "have",
                  "do", "make", "reason", "campaign", "induce", "case", "drive", "lawsuit", "grounds", "causa", "get"],
        'FIELD': ["theatre", "field_of_operation", "field_of_battle", "airfield", "sphere", "theater_of_operations",
                  "subject", "flying_field", "domain", "athletic_field", "force_field", "orbit", "field",
                  "theatre_of_operations", "plain", "theater", "subject_field", "field_of_study", "field_of_view",
                  "landing_field", "playing_field", "field_of_operations", "battleground", "subject_area",
                  "battlefield",
                  "field_of_force", "field_of_honor", "arena", "area", "bailiwick", "line_of_business", "champaign",
                  "discipline", "study", "playing_area"],
        'Dead': ["beat", "utterly", "deadened", "bushed", "numb", "stagnant", "abruptly", "short", "drained", "dead",
                 "perfectly", "absolutely", "idle", "utter", "suddenly", "all_in"],
        'Trespassing': ["intrude", "overstep", "invasive", "sin", "encroaching", "trespassing", "transgress",
                        "trespass",
                        "take_advantage"],
        'Residence': ["hall", "manse", "abidance", "abode", "residence", "mansion_house", "residency", "mansion"],
        'Intent': ["intent", "absorbed", "captive", "enwrapped", "design", "spirit", "purport", "engrossed",
                   "intention",
                   "aim", "wrapped", "purpose"],
        'Another': ["another", "some_other"],
        'Person': ["someone", "person", "individual", "soul", "somebody", "mortal"],
        'Assign': ["set_apart", "specify", "allot", "attribute", "designate", "delegate", "assign", "impute", "depute",
                   "portion", "put", "arrogate", "ascribe"],
        'Impersonating': ["personate", "portray", "pose", "impersonate"],
        'Beverages': ["drink", "beverage", "drinkable", "potable"],
        'Accessory': ["accessary", "appurtenance", "auxiliary", "supplement", "appurtenant", "accoutrement", "add-on",
                      "adjunct", "accessory", "ancillary", "adjuvant", "accouterment"],
        'Alprazolam': ["Xanax", "alprazolam"],
        'Leave': ["will", "provide", "parting", "go_away", "exit", "leave-taking", "impart", "pull_up_stakes",
                  "leave_behind", "leave_alone", "pass_on", "go_forth", "leave_of_absence", "leave", "get_out", "lead",
                  "give", "allow", "entrust", "allow_for", "depart", "forget", "go_out", "farewell", "result",
                  "bequeath"],
        'Suspended': ["set_aside", "suspended", "freeze", "suspend", "debar"],
        'Evidence': ["bear_witness", "grounds", "evidence", "show", "manifest", "tell", "testify", "prove", "certify",
                     "attest", "demonstrate"],
        'Suspend': ["suspend", "debar", "set_aside", "freeze"],
        'Court': ["romance", "lawcourt", "court", "woo", "royal_court", "motor_hotel", "court_of_law", "motor_lodge",
                  "homage", "judicature", "Margaret_Court", "motor_inn", "solicit", "courtroom", "Court",
                  "tourist_court",
                  "tribunal", "courtyard", "court_of_justice"],
        'Disabled': ["invalid", "handicapped", "disenable", "disable", "disabled", "incapacitate", "handicap"],
        'Speed': ["belt_along", "travel_rapidly", "speed_up", "quicken", "focal_ratio", "speed", "rush", "rush_along",
                  "f_number", "pelt_along", "pep_pill", "hotfoot", "cannonball_along", "zip", "speeding", "hurrying",
                  "step_on_it", "amphetamine", "upper", "race", "swiftness", "hurry", "velocity", "hie", "stop_number",
                  "accelerate", "hasten", "bucket_along", "fastness"],
        'Site': ["locate", "land_site", "web_site", "website", "internet_site", "situation", "place", "site"],
        'Enhanced': ["raise", "enhanced", "heighten", "enhance"],
        'Leaving': ["will", "provide", "go_away", "exit", "departure", "impart", "pull_up_stakes", "leave_behind",
                    "leave_alone", "pass_on", "going", "going_away", "go_forth", "leave", "get_out", "lead", "give",
                    "allow", "entrust", "allow_for", "depart", "leaving", "forget", "go_out", "result", "bequeath"],
        'Police': ["police_force", "constabulary", "police", "patrol", "law"],
        'Trespass': ["intrude", "trespass", "intrusion", "overstep", "sin", "violation", "transgress", "usurpation",
                     "take_advantage", "encroachment"],
        'Obstruct': ["block", "stymy", "embarrass", "hinder", "jam", "blockade", "obstruct", "obturate", "occlude",
                     "impede", "stymie", "close_up"],
        'Church': ["church", "church_building", "church_service", "Christian_church"],
        'Substance': ["subject_matter", "essence", "sum", "meaning", "kernel", "nub", "marrow", "gist", "heart", "pith",
                      "centre", "heart_and_soul", "center", "meat", "inwardness", "content", "means", "nitty-gritty",
                      "message", "substance", "core"],
        'Misuse': ["misapply", "abuse", "misuse", "pervert"],
        'Yrs': ["year", "yr", "twelvemonth"],
        'Comply': ["abide_by", "follow", "comply"],
        'Item': ["particular", "point", "item", "detail", "token"],
        'Harass': ["harass", "chivvy", "chevvy", "beset", "plague", "chevy", "molest", "hassle", "provoke", "harry",
                   "chivy"],
        'Leo': ["Leo_the_Lion", "Lion", "Leo"],
        'Store': ["fund", "salt_away", "put_in", "storage", "memory_board", "storehouse", "stack_away", "entrepot",
                  "stash_away", "hive_away", "lay_in", "depot", "memory", "shop", "stock", "computer_memory",
                  "computer_storage", "store"],
        'Private': ["common_soldier", "private", "individual", "buck_private", "secret"],
        'Months': ["calendar_month", "month"],
        'Assault': ["snipe", "violate", "set_on", "Assault", "assault", "rape", "attack", "dishonour", "violation",
                    "assail", "round", "lash_out", "ravish", "dishonor", "ravishment", "outrage"],
        'Criminal': ["felon", "felonious", "deplorable", "criminal", "outlaw", "vicious", "reprehensible",
                     "condemnable",
                     "malefactor", "crook"],
        'Premises': ["assumption", "premise", "preface", "precede", "introduce", "premiss", "premises"],
        'Information': ["info", "data", "selective_information", "entropy", "information"],
        'Attempted': ["try", "attempted", "set_about", "assay", "undertake", "essay", "attempt", "seek"],
        'Manage': ["cope", "handle", "oversee", "superintend", "bring_off", "wield", "care", "pull_off", "contend",
                   "do",
                   "grapple", "carry_off", "get_by", "manage", "make_out", "wangle", "finagle", "supervise",
                   "negociate",
                   "deal", "make_do"],
        'Leas': ["pastureland", "grazing_land", "lea", "pasture", "ley"],
        'Robbery': ["looting", "robbery"],
        'Soliciting': ["romance", "hook", "court", "solicit", "woo", "accost", "beg", "tap"],
        'Load': ["lade", "load", "loading", "shipment", "payload", "charge", "consignment", "burden", "onus",
                 "encumbrance",
                 "laden", "stretch", "warhead", "load_up", "freight", "incumbrance", "dilute", "lading", "debase",
                 "lode",
                 "adulterate", "cargo"],
        'Agree': ["match", "tally", "accord", "jibe", "check", "harmonize", "fit_in", "concord", "gibe", "harmonise",
                  "hold", "concur", "fit", "correspond", "consort", "agree"],
        'Return': ["repay", "take_back", "give_back", "rejoinder", "takings", "regress", "regaining", "reelect",
                   "counter",
                   "refund", "homecoming", "getting_even", "retort", "retrovert", "fall", "deliver", "recurrence",
                   "generate", "paying_back", "turn_back", "render", "riposte", "take", "comeback", "devolve",
                   "tax_return",
                   "restoration", "income_tax_return", "recall", "revert", "give", "payoff", "reappearance",
                   "restitution",
                   "yield", "hark_back", "return_key", "come_back", "replication", "issue", "bring_back", "pass",
                   "return",
                   "rejoin", "proceeds", "coming_back"],
        'Plant': ["imbed", "flora", "found", "works", "establish", "constitute", "set", "engraft", "plant_life",
                  "embed",
                  "plant", "institute", "industrial_plant", "implant"],
        'Invalid': ["invalid", "shut-in", "disable", "incapacitate", "handicap"],
        'Contempt': ["contempt", "disdain", "scorn", "despite", "disrespect"],
        'Fail': ["flunk", "flush_it", "give_way", "go_wrong", "break", "go", "miscarry", "break_down", "conk_out",
                 "bomb",
                 "fail", "die", "run_out", "go_bad", "give_out", "betray", "neglect"],
        'Non': ["non", "not"],
        'Steroid': ["steroid_hormone", "sex_hormone", "steroid"],
        'Ride': ["turn_on", "depend_upon", "depend_on", "sit", "tease", "cod", "tantalize", "mount", "hinge_upon",
                 "ride",
                 "rally", "drive", "devolve_on", "rag", "bait", "twit", "hinge_on", "razz", "taunt", "tantalise"],
        'Native': ["indigen", "indigene", "aborigine", "aboriginal", "native"],
        'Scanning': ["scanning", "skim", "scan", "read", "rake", "glance_over", "run_down"],
        'Id': ["I.D.", "Idaho", "Gem_State", "id", "ID"],
        'Bribery': ["bribery", "graft"],
        'Financial': ["fiscal", "financial"],
        'Assignation': ["apportionment", "tryst", "assignation", "parcelling", "apportioning", "allotment",
                        "allocation",
                        "parceling"],
        '5': ["fin", "pentad", "5", "quint", "cinque", "five", "quintuplet", "quintet", "fivesome", "Phoebe",
              "Little_Phoebe", "V", "v"],
        'Cancel': ["cancel", "scrub", "invalidate", "strike_down", "natural", "set_off", "scratch", "offset",
                   "call_off",
                   "delete"],
        'Plates': ["dental_plate", "home", "plate", "scale", "shell", "plateful", "plat", "crustal_plate", "home_plate",
                   "home_base", "denture", "plot", "collection_plate", "photographic_plate"],
        'Retaliate': ["retaliate", "strike_back", "revenge", "avenge"],
        'Interfere': ["step_in", "interfere", "interpose", "intervene"],
        'Horses': ["buck", "Equus_caballus", "horse_cavalry", "knight", "gymnastic_horse", "cavalry", "horse",
                   "sawbuck",
                   "sawhorse"],
        'Tetrahydrocannabinols': ["THC", "tetrahydrocannabinol"],
        'Inst': ["inst", "instant"],
        'Anabolic': ["anabolic"],
        'Counterfeit': ["imitative", "fake", "forge", "forgery", "counterfeit"],
        'Ecstasy': ["disco_biscuit", "ecstasy", "X", "rapture", "cristal", "XTC", "exaltation", "transport", "raptus",
                    "hug_drug", "Adam", "go"],
        'level': ["story", "even", "degree", "rase", "stratum", "plane", "tied", "tear_down", "floor", "take_down",
                  "level_off", "point", "stage", "grade", "spirit_level", "tier", "charge", "storey", "flat",
                  "even_out",
                  "flush", "unwavering", "pull_down", "level", "dismantle", "horizontal_surface", "layer", "raze"],
        'School': ["school", "schoolhouse", "shoal", "schooltime", "schooling", "educate", "civilize", "school_day",
                   "train", "civilise", "cultivate"],
        'Delinquency': ["willful_neglect", "dereliction", "delinquency", "juvenile_delinquency"],
        'Offer': ["put_up", "extend", "provide", "propose", "crack", "tender", "whirl", "declare_oneself", "offer",
                  "pop_the_question", "offer_up", "bid", "pass", "volunteer", "offering", "fling", "proffer", "go"],
        'Use': ["exercise", "enjoyment", "employment", "habituate", "manipulation", "usance", "habit", "consumption",
                "apply", "utilise", "practice", "employ", "role", "utilisation", "use", "utilize", "utilization",
                "purpose",
                "use_of_goods_and_services", "economic_consumption", "usage", "expend", "function"],
        'Witness': ["informant", "see", "attestant", "spectator", "witness", "attestor", "find", "attestator",
                    "watcher",
                    "witnesser", "viewer", "looker"],
        'Permit': ["let", "permit", "permission", "licence", "countenance", "tolerate", "allow", "license",
                   "Trachinotus_falcatus"],
        'Yr': ["year", "yr", "twelvemonth"],
        'Order': ["edict", "order", "range", "parliamentary_procedure", "place", "put", "dictate", "gild", "Order",
                  "enjoin", "arrange", "lodge", "ordinate", "grade", "social_club", "govern", "rules_of_order",
                  "monastic_order", "club", "rate", "regulate", "parliamentary_law", "order_of_magnitude", "guild",
                  "purchase_order", "fiat", "decree", "tell", "orderliness", "prescribe", "rescript", "ordain",
                  "society",
                  "Holy_Order", "regularize", "ordering", "set_up", "consecrate", "regularise", "rank", "say",
                  "ordination"],
        'Sample': ["sampling", "taste", "sample", "try_out", "sample_distribution", "try"],
        'near': ["cheeseparing", "most", "draw_near", "come_on", "about", "nearly", "go_up", "skinny", "approximate",
                 "penny-pinching", "virtually", "close", "almost", "dear", "draw_close", "nigh", "well-nigh",
                 "come_near",
                 "near", "good", "approach"],
        'Offense': ["offense", "umbrage", "criminal_offence", "criminal_offense", "law-breaking", "offensive",
                    "offensive_activity", "crime", "offence", "discourtesy"],
        'Sound': ["levelheaded", "profound", "speech_sound", "effectual", "vocalise", "strait", "fathom", "reasoned",
                  "well-grounded", "audio", "legal", "phone", "healthy", "level-headed", "go", "sound", "voice",
                  "auditory_sensation", "good", "heavy", "vocalize", "intelligent", "wakeless"],
        'Arson': ["fire-raising", "arson", "incendiarism"],
        'sell': ["sell", "deal", "trade", "betray"],
        'Giving': ["openhanded", "break", "chip_in", "bighearted", "impart", "grant", "hold", "handsome", "afford",
                   "dedicate", "bounteous", "contribute", "pass_on", "sacrifice", "have", "generate", "freehanded",
                   "giving", "reach", "apply", "move_over", "turn_over", "establish", "render", "feed", "cave_in",
                   "make",
                   "give_way", "kick_in", "bountiful", "gift", "leave", "big", "give", "founder", "fall_in", "ease_up",
                   "devote", "hand", "commit", "open", "yield", "liberal", "collapse", "consecrate", "throw", "pay",
                   "pass",
                   "return", "present"],
        'Servant': ["handmaid", "retainer", "handmaiden", "servant"],
        'Ownership': ["possession", "ownership"],
        'Danger': ["danger", "peril", "risk"],
        'Statement': ["affirmation", "financial_statement", "program_line", "command", "instruction", "assertion",
                      "argument", "statement"],
        'High': ["highschool", "high_gear", "mellow", "richly", "high", "eminent", "heights", "in_high_spirits",
                 "gamey",
                 "senior_high_school", "high_school", "senior_high", "high_up", "high-pitched", "gamy", "luxuriously"],
        'Lewdness': ["bawdiness", "obscenity", "salaciousness", "salacity", "lewdness"],
        'Wit': ["witticism", "learning_ability", "mentality", "brain", "mental_capacity", "brainpower", "wag", "wit",
                "card", "humor", "humour", "wittiness"],
        'Restrictions': ["confinement", "restriction", "limitation"],
        'Serious': ["dangerous", "severe", "grievous", "life-threatening", "good", "grave", "serious", "sober",
                    "unplayful"],
        'at': ["At", "atomic_number_85", "at", "astatine"],
        'African': ["African"],
        'American': ["American", "American_language", "American_English"],
        'Conspire': ["cabal", "collude", "conjure", "conspire", "complot", "machinate"],
        'Pub': ["saloon", "taphouse", "pothouse", "pub", "public_house", "gin_mill"],
        'Wireless': ["radio_set", "receiving_set", "radio", "radiocommunication", "wireless", "tuner",
                     "radio_receiver"],
        'Revoked': ["reverse", "annul", "countermand", "overturn", "vacate", "repeal", "rescind", "lift", "revoke"],
        'Athletic': ["acrobatic", "gymnastic", "athletic"],
        'Contract': ["compact", "narrow", "cut", "sign_on", "constrict", "concentrate", "contract_bridge", "condense",
                     "take", "contract", "reduce", "declaration", "sign", "shrink", "undertake", "sign_up", "press",
                     "squeeze", "shorten", "abbreviate", "foreshorten", "get", "abridge", "compress"],
        'Solicitation': ["allurement", "solicitation", "appeal", "collection", "ingathering"],
        'Officer': ["police_officer", "policeman", "ship's_officer", "officer", "military_officer", "officeholder"],
        'Obtain': ["receive", "get", "incur", "find", "prevail", "hold", "obtain"],
        'Loitering': ["mess_about", "linger", "mill_around", "lollygag", "loaf", "footle", "loiter", "lurk",
                      "hang_around",
                      "lounge", "lallygag", "mill_about", "tarry"],
        'Fighting': ["combat-ready", "scrap", "fight", "defend", "campaign", "agitate", "active", "fighting", "press",
                     "contend", "struggle", "oppose", "fight_back", "fight_down", "combat", "crusade", "push"],
        'Bod': ["shape", "soma", "bod", "form", "material_body", "figure", "chassis", "physical_body", "physique",
                "build",
                "human_body", "frame", "flesh", "anatomy"],
        'Open': ["undecided", "surface", "open_up", "unresolved", "clear", "candid", "unfold", "loose", "assailable",
                 "afford", "spread", "subject", "out-of-doors", "open_air", "outdoors", "capable", "undetermined",
                 "give",
                 "exposed", "spread_out", "open", "opened", "undefended", "undefendable", "receptive", "heart-to-heart",
                 "unfastened", "overt"],
        'Aid': ["care", "assistance", "help", "financial_aid", "aid", "assist", "attention", "tending", "economic_aid"],
        'Purchasing': ["buy", "buying", "purchasing", "purchase"],
        'Metal': ["alloy", "metal", "metallic", "metallic_element"],
        'Sleeping': ["log_Z's", "dormant", "kip", "catch_some_Z's", "quiescency", "sleep", "dormancy", "sleeping",
                     "quiescence", "slumber"],
        'Failure': ["nonstarter", "unsuccessful_person", "loser", "bankruptcy", "failure"],
        'Damage': ["wrong", "price", "terms", "harm", "legal_injury", "equipment_casualty", "hurt", "impairment",
                   "scathe",
                   "damage"],
        'Girlfriend': ["girl", "lady_friend", "girlfriend"],
        'Utter': ["complete", "verbalize", "double-dyed", "give_tongue_to", "speak", "arrant", "everlasting", "stark",
                  "let_out", "perfect", "sodding", "verbalise", "dead", "express", "gross", "consummate", "mouth",
                  "thoroughgoing", "emit", "pure", "unadulterated", "staring", "let_loose", "utter", "talk"],
        'Influence': ["shape", "work", "determine", "charm", "mold", "influence", "act_upon", "regulate", "tempt"],
        'Cab': ["cab", "cabriolet", "hack", "taxicab", "taxi"],
        'Instrument': ["pawn", "instrument", "official_document", "musical_instrument", "cat's-paw", "legal_document",
                       "tool", "instrumentate", "legal_instrument", "instrumental_role"],
        'Discharge': ["complete", "venting", "fire", "exonerate", "firing_off", "drop_off", "sack", "liberation",
                      "exhaust",
                      "expelling", "clear", "put_down", "dismission", "release", "assoil", "acquit", "free",
                      "exculpate",
                      "muster_out", "unload", "discharge", "electric_arc", "electric_discharge", "outpouring", "firing",
                      "arc", "dismissal", "dispatch", "spark", "go_off", "expel", "emission", "waiver", "eject",
                      "empty",
                      "run", "sacking", "drop", "set_down"],
        'Violation': ["infringement", "infraction", "intrusion", "ravishment", "assault", "rape", "violation",
                      "irreverence", "misdemeanor", "trespass", "encroachment", "misdemeanour", "usurpation"],
        'FL': ["FL", "Florida", "Sunshine_State", "Everglade_State"],
        'Homicide': ["homicide"],
        'Man': ["human", "serviceman", "gentleman's_gentleman", "homo", "adult_male", "military_personnel", "gentleman",
                "valet_de_chambre", "man", "Isle_of_Man", "humanity", "humans", "valet", "human_race", "piece",
                "human_beings", "military_man", "humankind", "Man", "human_being", "mankind", "world"],
        'Disable': ["invalid", "disenable", "disable", "incapacitate", "handicap"],
        'Low': ["grim", "low-down", "depression", "low-spirited", "Low", "abject", "miserable", "Sir_David_Low",
                "humiliated", "scurvy", "crushed", "down_in_the_mouth", "first_gear", "low-pitched", "down", "blue",
                "depleted", "moo", "downhearted", "low_gear", "modest", "lowly", "broken", "dispirited", "low-toned",
                "small", "first", "depressed", "gloomy", "David_Low", "low", "Sir_David_Alexander_Cecil_Low", "scummy",
                "downcast", "humbled", "humble"],
        'Material': ["textile", "substantial", "material", "fabric", "stuff", "cloth", "real", "corporeal"],
        'Posses': ["posse", "posse_comitatus"],
        'Deliver': ["bear", "cede", "extradite", "rescue", "save", "drive_home", "turn_in", "give_birth", "deliver",
                    "surrender", "fork_over", "have", "redeem", "hand_over", "render", "give_up", "deport", "fork_up",
                    "birth", "fork_out", "pitch", "return", "present"],
        'ID': ["I.D.", "Idaho", "Gem_State", "id", "ID"],
        'Contraband': ["black-market", "bootleg", "black", "contraband", "smuggled"],
        'Contests': ["repugn", "competition", "contest", "contend"],
        '3rd': ["third", "3rd", "tertiary"],
        'Bomb': ["flunk", "flush_it", "dud", "bomb_calorimeter", "turkey", "bombard", "fail", "bomb"],
        'Destruct': ["destroy", "destruct"],
        'Great': ["expectant", "great", "capital", "smashing", "enceinte", "bully", "dandy", "nifty", "corking",
                  "gravid",
                  "large", "swell", "with_child", "big", "cracking", "groovy", "outstanding", "keen", "heavy",
                  "slap-up",
                  "peachy", "bang-up", "majuscule", "not_bad", "neat"],
        'Tools': ["shaft", "putz", "tool_around", "pecker", "prick", "instrument", "joyride", "dick", "tool", "cock",
                  "creature", "puppet", "peter"],
        'Uttering': ["verbalize", "give_tongue_to", "let_out", "speak", "verbalise", "let_loose", "express", "mouth",
                     "utter", "talk", "emit"],
        'Acc': ["ACC", "Air_Combat_Command"],
        'Test': ["examination", "test", "exam", "mental_test", "psychometric_test", "trial", "screen", "run", "try_out",
                 "mental_testing", "prove", "quiz", "essay", "examine", "try", "trial_run", "tryout"],
        'Anti': ["anti"],
        'Shoplifting': ["shrinkage", "shoplift", "shoplifting"],
        'Accident': ["chance_event", "stroke", "fortuity", "accident"],
        'Hispanic': ["Latino", "Hispanic_American", "Spanish_American", "Hispanic"],
        'Causes': ["cause", "stimulate", "causal_agency", "causal_agent", "crusade", "suit", "effort", "movement",
                   "have",
                   "do", "make", "reason", "campaign", "induce", "case", "drive", "lawsuit", "grounds", "causa", "get"],
        'Beach': ["beach"],
        'Personating': ["personate", "personify", "pose", "impersonate"],
        'Boater': ["sailor", "straw_hat", "skimmer", "Panama_hat", "Panama", "leghorn", "boater", "boatman",
                   "waterman"],
        'Culpable': ["blameful", "culpable", "blameable", "blameworthy", "censurable", "blamable"],
        'Money': ["money"],
        'Prowling': ["prowl", "lurch"],
        'Submit': ["state", "take", "defer", "put_in", "posit", "subject", "pass_on", "render", "put_forward", "bow",
                   "reconcile", "relegate", "submit", "present", "accede", "resign", "give_in"],
        'Habit': ["substance_abuse", "drug_abuse", "wont", "habit", "riding_habit", "use"],
        'Fish': ["Fish", "Pisces", "angle", "Pisces_the_Fishes", "fish"],
        'Aggress': ["aggress", "attack"],
        'Deprive': ["impoverish", "divest", "strip", "deprive"],
        'Aggravated': ["exasperate", "aggravate", "provoked", "exacerbate", "worsen", "aggravated"],
        'Injunction': ["enjoining", "cease_and_desist_order", "injunction", "enjoinment"],
        'Grams': ["g", "gramme", "gm", "Gram", "Hans_C._J._Gram", "gram"],
        'Lease': ["charter", "take", "engage", "let", "hire", "term_of_a_contract", "rent", "letting", "rental",
                  "lease"],
        'Check': ["break", "check_up_on", "train", "hold", "tick_off", "fit", "bridle", "baulk", "ascertain",
                  "find_out",
                  "see_to_it", "crack", "control", "insure", "checkout", "hitch", "check_out", "bank_check", "arrest",
                  "correspond", "curb", "assure", "impediment", "hold_back", "turn_back", "confirmation", "handicap",
                  "see",
                  "tally", "discipline", "check-out_procedure", "check_off", "tick", "stoppage", "stop", "condition",
                  "mark", "delay", "hindrance", "verification", "contain", "determine", "moderate", "tab", "chip",
                  "ensure",
                  "check_into", "mark_off", "halt", "look_into", "hold_in", "retard", "check_over", "match", "cheque",
                  "jibe", "deterrent", "check", "substantiation", "go_over", "chequer", "learn", "hinderance",
                  "suss_out",
                  "assay", "gibe", "watch", "stay", "balk", "chink", "agree", "chit", "check_mark", "checker"],
        'Physical': ["physical", "strong-arm", "forcible"],
        'battery': ["shelling", "electric_battery", "barrage_fire", "battery", "bombardment", "stamp_battery",
                    "assault_and_battery", "barrage"],
        'Conviction': ["strong_belief", "conviction", "condemnation", "sentence", "article_of_faith",
                       "judgment_of_conviction"],
        'Rail': ["track", "rail_off", "railing", "inveigh", "vilify", "train", "fulminate", "vituperate", "rail",
                 "rails",
                 "revile", "rail_in", "runway"],
        'Way': ["style", "way_of_life", "way", "manner", "right_smart", "fashion", "mode", "direction", "agency",
                "means",
                "room", "elbow_room", "path"],
        'Custody': ["custody", "detainment", "hold", "hands", "detention"],
        'Copper': ["atomic_number_29", "pig", "copper", "Cu", "fuzz", "cop", "bull", "copper_color"],
        'Bylaw': ["bylaw"],
        'Vehicle': ["fomite", "vehicle"],
        'Conveyance': ["conveyance", "impartation", "transportation", "conveyance_of_title", "conveyancing",
                       "conveying",
                       "transferral", "transport", "imparting", "transfer"],
        'Para': ["Para", "Santa_Maria_de_Belem", "paratrooper", "parity", "Feliz_Lusitania", "St._Mary_of_Bethlehem",
                 "Para_River", "para", "Belem"],
        'Injury': ["trauma", "injury", "harm", "accidental_injury", "hurt", "combat_injury", "wound"],
        'Unemployment': ["unemployment"],
        'NO': ["No", "atomic_number_102", "no_more", "no", "nobelium"],
        'Address': ["name_and_address", "handle", "reference", "address", "computer_address", "savoir-faire", "cover",
                    "speak", "come_up_to", "direct", "plow", "speech", "turn_to", "destination", "call", "deal",
                    "treat",
                    "accost"],
        'Structuring': ["structure"],
        'Education': ["pedagogy", "training", "didactics", "educational_activity", "Department_of_Education",
                      "Education_Department", "education", "Education", "instruction", "teaching", "breeding"],
        'Equipment': ["equipment"],
        'Present': ["confront", "present", "submit", "demonstrate", "face", "stage", "show", "award", "deliver", "demo",
                    "portray", "represent", "salute", "lay_out", "nowadays", "gift", "give", "introduce", "acquaint",
                    "present_tense", "exhibit", "pose"],
        'Reckless': ["foolhardy", "reckless", "heedless", "heady", "rash"],
        'Introduce': ["inclose", "put_in", "acquaint", "enclose", "innovate", "preface", "premise", "inaugurate",
                      "infix",
                      "stick_in", "bring_out", "precede", "bring_in", "usher_in", "introduce", "insert", "present",
                      "enter"],
        'Credit': ["recognition", "reference", "acknowledgment", "credit", "course_credit", "citation", "cite",
                   "accredit",
                   "credit_entry", "quotation", "credit_rating", "mention", "deferred_payment"],
        'Second': ["arcsecond", "secondment", "sec", "indorsement", "mo", "minute", "back", "2d", "endorse", "indorse",
                   "second_base", "second", "bit", "instant", "moment", "endorsement", "s", "second_gear", "irregular",
                   "secondly", "2nd"],
        'Intoxicating': ["lift_up", "inebriate", "intoxicating", "intoxicate", "heady", "pick_up", "elate", "soak",
                         "intoxicant", "uplift"],
        'City': ["urban_center", "metropolis", "city"],
        'Motorcycle': ["bike", "cycle", "motorbike", "motorcycle"],
        '3': ["terzetto", "trine", "III", "threesome", "iii", "tercet", "triplet", "ternion", "trey", "triad",
              "deuce-ace",
              "ternary", "three", "trinity", "trio", "3", "tierce", "troika", "leash"],
        'or': ["operating_theater", "OR", "surgery", "Beaver_State", "operating_room", "Oregon", "operating_theatre"],
        'nan': ["naan", "Nan_River", "granny", "grandma", "grannie", "nanna", "Nan", "grandmother", "nan", "gran"],
        'Cash': ["Johnny_Cash", "cash", "hard_currency", "immediate_payment", "cash_in", "hard_cash", "Cash",
                 "John_Cash"],
        'Container': ["container"],
        'Arrest': ["nail", "hold", "hitch", "cop", "arrest", "hold_back", "turn_back", "pinch", "nab", "collar",
                   "apprehension", "stoppage", "stop", "catch", "apprehend", "contain", "pick_up",
                   "taking_into_custody",
                   "halt", "check", "get", "stay"],
        'Duties': ["obligation", "duty", "responsibility", "tariff"],
        'Mandatory': ["required", "mandatory", "mandatary", "mandate", "compulsory"],
        'Building': ["construction", "building", "edifice", "progress", "make", "work_up", "build", "ramp_up",
                     "construct",
                     "establish", "build_up"],
        'Function': ["work", "affair", "serve", "social_occasion", "officiate", "routine", "occasion",
                     "single-valued_function", "procedure", "go", "role", "subroutine", "operate", "map", "use",
                     "mapping",
                     "part", "purpose", "social_function", "office", "run", "mathematical_function", "subprogram",
                     "function"],
        'Interference': ["interference", "incumbrance", "encumbrance", "hitch", "disturbance", "hinderance", "noise",
                         "preventive", "preventative", "hindrance", "intervention"],
        'Male': ["male", "manlike", "male_person", "manful", "Male", "virile", "manly"],
        'Fuel': ["fuel", "fire"],
        'Lascivious': ["lewd", "libidinous", "lascivious", "lustful"],
        'Product': ["production", "merchandise", "ware", "mathematical_product", "intersection", "product",
                    "Cartesian_product"],
        'Blood': ["stemma", "line", "lineage", "parentage", "pedigree", "line_of_descent", "rakehell", "ancestry",
                  "roue",
                  "rake", "blood", "rip", "profligate", "stock", "bloodline", "blood_line", "origin", "descent"],
        'Grow': ["maturate", "raise", "acquire", "mature", "turn", "arise", "rise", "develop", "produce", "farm",
                 "spring_up", "get", "originate", "uprise", "grow"],
        'Purchase': ["purchase", "leverage", "buy"],
        '1st': ["first", "1st"],
        'Carrying': ["bear", "post", "behave", "pack", "impart", "hold", "expect", "transport", "channel", "extend",
                     "gestate", "acquit", "carry", "express", "take", "comport", "contain", "stock", "transmit",
                     "deport",
                     "convey", "stockpile", "dribble", "run", "conduct", "persuade", "sway", "have_a_bun_in_the_oven"],
        'Actual': ["actual", "existent", "factual", "real", "literal", "genuine"],
        'Charge': ["send", "turn_on", "load", "commission", "kick", "commove", "bill", "consign", "boot",
                   "electric_charge",
                   "rush", "blame", "excite", "care", "buck", "appoint", "mission", "lodge", "charge_up", "charge",
                   "point",
                   "heraldic_bearing", "file", "saddle", "burden", "thrill", "institutionalise", "armorial_bearing",
                   "tear",
                   "institutionalize", "complaint", "flush", "accuse", "agitate", "bang", "rouse", "bearing", "level",
                   "explosive_charge", "bear_down", "shoot", "bursting_charge", "commit", "tutelage", "accusation",
                   "guardianship", "shoot_down", "billing", "direction", "burster", "cathexis"],
        'Requirements': ["demand", "requisite", "prerequisite", "necessary", "essential", "necessity", "requirement"],
        'Mot': ["MOT_test", "bon_mot", "MOT", "mot", "Ministry_of_Transportation_test"],
        'MARK': ["marker", "score", "Mark", "tag", "pit", "sucker", "note", "Gospel_According_to_Mark", "commemorate",
                 "gull", "mug", "tick_off", "punctuate", "St._Mark", "distinguish", "grade", "cross", "differentiate",
                 "set", "print", "brand", "fool", "chump", "stigma", "crisscross", "cross_out", "stigmatise", "scar",
                 "scrape", "patsy", "Deutsche_Mark", "scratch", "strike_out", "check_off", "Saint_Mark", "tick",
                 "bull's_eye", "pock", "mark", "soft_touch", "sign", "home_run", "label", "nock", "German_mark",
                 "strike_off", "fall_guy", "denounce", "mark_off", "marking", "target", "stigmatize", "notice", "check",
                 "Deutschmark", "cross_off", "bell_ringer", "stain"],
        'a': ["angstrom_unit", "angstrom", "vitamin_A", "amp", "group_A", "adenine", "deoxyadenosine_monophosphate",
              "a",
              "axerophthol", "ampere", "type_A", "antiophthalmic_factor", "A"],
        'Principal': ["school_principal", "principal_sum", "chief", "principal", "main", "star", "lead", "head",
                      "primary",
                      "dealer", "head_teacher", "master", "corpus"],
        'Throw': ["contrive", "throw_off", "shake_off", "cast", "switch", "hold", "cam_stroke", "throw_away", "fox",
                  "bewilder", "discombobulate", "confound", "fuddle", "project", "have", "thrust", "make", "shed",
                  "befuddle", "hurl", "confuse", "give", "cast_off", "flip", "bemuse", "stroke", "bedevil", "throw",
                  "drop"],
        'Carry': ["bear", "post", "behave", "pack", "impart", "hold", "expect", "transport", "channel", "extend",
                  "gestate",
                  "acquit", "carry", "express", "take", "comport", "contain", "stock", "transmit", "deport", "convey",
                  "stockpile", "dribble", "run", "conduct", "persuade", "sway", "have_a_bun_in_the_oven"],
        '2': ["two", "ii", "2", "deuce", "II"],
        'Sol': ["colloidal_suspension", "sol", "soh", "colloidal_solution", "Sol", "so"],
        'Less': ["to_a_lesser_extent", "less"],
        'Unauthorized': ["unauthorised", "unauthorized", "wildcat"],
        'Display': ["video_display", "display", "show", "presentation", "exhibit", "expose", "showing"],
        'Removed': ["move_out", "murder", "off", "absent", "slay", "transfer", "distant", "take_out", "remote", "take",
                    "bump_off", "dispatch", "take_away", "hit", "remove", "withdraw", "removed", "get_rid_of",
                    "polish_off"],
        'Extradition': ["extradition"],
        'arrest': ["nail", "hold", "hitch", "cop", "arrest", "hold_back", "turn_back", "pinch", "nab", "collar",
                   "apprehension", "stoppage", "stop", "catch", "apprehend", "contain", "pick_up",
                   "taking_into_custody",
                   "halt", "check", "get", "stay"],
        'Caucasian': ["White", "Caucasian_language", "Caucasoid", "Caucasic", "Caucasian", "White_person"],
        'Wholesale': ["wholesale", "sweeping", "in_large_quantities"],
        'Licensed': ["commissioned", "licenced", "accredited", "licence", "certify", "license", "licensed"],
        'Diazepam': ["diazepam", "Valium"],
        'Contractor': ["contractor", "contractile_organ", "declarer"],
        'Exhibition': ["exhibition", "exposition", "expo"],
        'Construction': ["construction", "building", "structure", "grammatical_construction", "twist", "expression",
                         "mental_synthesis"],
        'F': ["degree_Fahrenheit", "F", "fluorine", "f", "atomic_number_9", "farad"],
        'Sale': ["cut-rate_sale", "sales_event", "sale", "sales_agreement"],
        'Dols': ["DoL", "Department_of_Labor", "Labor", "dol", "Labor_Department"],
        'License': ["permit", "permission", "licence", "certify", "license"],
        'Purpose': ["intent", "determination", "role", "design", "function", "use", "purport", "intention", "aim",
                    "propose", "resolve", "purpose"],
        'Mischief': ["mischief", "balefulness", "devilry", "roguishness", "deviltry", "rascality", "roguery",
                     "maleficence",
                     "mischievousness", "shenanigan", "devilment", "mischief-making"],
        'Codeine': ["codeine"],
        'Unlawful': ["outlaw", "illicit", "outlawed", "improper", "unconventional", "illegitimate", "wrongful",
                     "unlawful"],
        'Tampering': ["meddle", "tamper", "tampering", "meddling", "monkey", "fiddle"],
        'Valid': ["valid"],
        'Exposes': ["unwrap", "break", "display", "peril", "queer", "endanger", "disclose", "let_out", "reveal",
                    "scupper",
                    "bring_out", "let_on", "debunk", "uncover", "give_away", "expose", "exhibit", "unmasking",
                    "divulge",
                    "discover"],
        'CI': ["one_hundred_one", "ci", "hundred_and_one", "101", "Ci", "curie"],
        'Amphetamine': ["speed", "amphetamine", "upper", "pep_pill"],
        'Change': ["transfer", "vary", "modify", "shift", "convert", "deepen", "modification", "interchange", "change",
                   "switch", "exchange", "alter", "alteration", "variety", "commute"],
        'Fighter': ["fighter", "champion", "attack_aircraft", "hero", "scrapper", "belligerent", "combatant", "battler",
                    "fighter_aircraft", "paladin"],
        'w': ["w", "double-u", "due_west", "tungsten", "watt", "westward", "west", "atomic_number_74", "W", "wolfram"],
        'Grand': ["K", "M", "grand", "exalted", "august", "lordly", "one_thousand", "idealistic", "gilded", "wonderful",
                  "noble-minded", "G", "grand_piano", "marvelous", "marvellous", "opulent", "tremendous", "1000",
                  "heroic",
                  "thou", "rattling", "high-flown", "sumptuous", "chiliad", "wondrous", "rarified", "howling",
                  "luxurious",
                  "distinguished", "rarefied", "magisterial", "sublime", "terrific", "elevated", "yard", "high-minded",
                  "expansive", "deluxe", "princely", "lofty", "thousand", "fantastic", "imposing"],
        '65': ["lxv", "sixty-five", "65"],
        'Disrupting': ["interrupt", "break_up", "disrupt", "cut_off"],
        'Prop': ["prop", "shore", "property", "airplane_propeller", "prop_up", "airscrew", "shore_up"],
        'Threaten': ["imperil", "peril", "menace", "jeopardize", "threaten", "jeopardise", "endanger"],
        'D': ["calciferol", "cholecalciferol", "vitamin_D", "500", "viosterol", "five_hundred", "ergocalciferol", "D",
              "d"],
        'Taxi': ["cab", "hack", "taxicab", "taxi"],
        'Intoxicated': ["inebriated", "lift_up", "intoxicated", "inebriate", "drunk", "intoxicate", "pick_up", "elate",
                        "soak", "uplift"],
        'Phone': ["telephone", "headphone", "earphone", "sound", "speech_sound", "call_up", "phone", "ring", "call",
                  "earpiece", "telephone_set"],
        'Retail': ["retail"],
        'Snatch': ["slit", "cunt", "kidnap", "snatch_up", "bit", "abduct", "nobble", "catch", "pussy", "kidnapping",
                   "puss",
                   "grab", "snatch", "twat", "snap"],
        'Over': ["complete", "over", "concluded", "terminated", "all_over", "ended", "o'er"],
        'Career': ["vocation", "career", "life_history", "calling"],
        'Forging': ["shape", "work", "contrive", "form", "mold", "fake", "spirt", "mould", "invent", "hammer", "forge",
                    "fashion", "forging", "excogitate", "formulate", "spurt", "devise", "counterfeit"],
        'Family': ["family", "household", "family_unit", "family_line", "kinsperson", "class", "mob", "category",
                   "syndicate", "sept", "kin", "fellowship", "kinfolk", "kinsfolk", "phratry", "home", "house", "folk",
                   "crime_syndicate", "menage"],
        'Telemarketing': ["teleselling", "telecommerce", "telemarketing"],
        'Dome': ["bean", "noodle", "covered_stadium", "attic", "noggin", "domed_stadium", "bonce", "dome"],
        'On': ["on", "along"],
        'Theft': ["stealing", "thievery", "larceny", "theft", "thieving"],
        'Pornography': ["porn", "porno", "smut", "pornography", "erotica"],
        'Commission': ["delegacy", "mission", "committal", "commission", "charge", "commissioning", "deputation",
                       "perpetration", "direction", "delegation", "committee", "military_commission"],
        'Earnings': ["remuneration", "profit", "net_income", "lucre", "salary", "earnings", "wage", "pay", "net",
                     "profits",
                     "net_profit"],
        'Deadly': ["virulent", "pernicious", "venomous", "baneful", "lethal", "deathly", "devilishly", "madly",
                   "insanely",
                   "deucedly", "mortal", "deadly", "lifelessly", "pestilent"],
        'I': ["ace", "atomic_number_53", "1", "single", "unity", "i", "iodine", "ane", "iodin", "one", "I"],
        'Indecent': ["indecorous", "indecent", "untoward", "uncomely", "unbecoming", "unseemly"],
        'Launder': ["wash", "launder"],
        '2nd': ["2d", "second", "2nd"],
        'Release': ["expiration", "let_go", "spill", "publish", "exit", "sack", "liberation", "departure", "button",
                    "exhaust", "vent", "outlet", "handout", "dismission", "release", "loose", "press_release",
                    "freeing",
                    "free", "unloose", "secrete", "going", "bring_out", "discharge", "spillage", "firing", "unblock",
                    "dismissal", "passing", "turn", "unfreeze", "tone_ending", "give_up", "let_go_of", "resign",
                    "unloosen",
                    "liberate", "expel", "waiver", "loss", "relinquish", "eject", "issue", "sacking", "acquittance",
                    "put_out"],
        'Mask': ["cloak", "disguise", "masquerade", "mask", "dissemble", "masque", "block_out", "masquerade_party"],
        'Driving': ["ram", "repulse", "beat_back", "repel", "aim", "push_back", "push", "driving", "force_back", "take",
                    "impulsive", "ride", "labor", "tug", "drive", "labour", "force", "motor", "get"],
        'Stolen': ["slip", "steal"],
        'grams': ["g", "gramme", "gm", "Gram", "Hans_C._J._Gram", "gram"],
        'Bat': ["drub", "bat", "squash_racket", "cricket_bat", "chiropteran", "cream", "thrash", "squash_racquet",
                "flutter", "at-bat", "lick", "clobber"],
        'Kidnapping': ["kidnap", "abduct", "nobble", "kidnapping", "snatch"],
        'After': ["subsequently", "later_on", "later", "afterwards", "afterward", "after"],
        'Dating': ["date_stamp", "dating", "see", "geological_dating", "go_steady", "date", "go_out"],
        'Meth': ["ice", "methamphetamine_hydrochloride", "chicken_feed", "methamphetamine", "trash", "deoxyephedrine",
                 "Methedrine", "shabu", "crank", "glass", "meth", "chalk"],
        'Intoxication': ["drunkenness", "tipsiness", "intoxication", "inebriation", "toxic_condition", "poisoning",
                         "inebriety", "insobriety"],
        'Innkeeper': ["innkeeper", "host", "boniface"],
        'Act': ["work", "act_as", "behave", "routine", "move", "pretend", "act", "human_activity", "represent", "do",
                "playact", "play", "turn", "bit", "enactment", "number", "deed", "human_action", "dissemble",
                "roleplay"],
        'Victim': ["dupe", "victim"],
        'Merchant': ["merchandiser", "merchant"],
        'Control': ["restraint", "keep_in_line", "hold", "master", "mastery", "control_condition", "ascendence",
                    "ascendancy", "ascertain", "see_to_it", "dominance", "control", "insure", "curb", "assure", "see",
                    "operate", "ascendance", "manipulate", "contain", "verify", "moderate", "controller", "ensure",
                    "hold_in", "check", "command", "ascendency"],
        'No': ["No", "atomic_number_102", "no_more", "no", "nobelium"],
        'on': ["on", "along"],
        'Disobey': ["disobey"],
        'Conspiracy': ["confederacy", "cabal", "conspiracy"],
        'Neglect': ["omit", "disuse", "disregard", "overlook", "carelessness", "miss", "leave_out", "ignore", "fail",
                    "nonperformance", "neglectfulness", "pretermit", "overleap", "drop", "negligence", "neglect"],
        'Cocaine': ["cocain", "cocaine"],
        '3Rd': ["third", "3rd", "tertiary"],
        'Tamper': ["meddle", "tamp", "tamper", "tamping_bar", "monkey", "fiddle"],
        'DL': ["dl", "decilitre", "deciliter"],
        'Greater': ["expectant", "great", "capital", "smashing", "greater", "enceinte", "bully", "dandy", "nifty",
                    "corking", "gravid", "large", "swell", "with_child", "big", "cracking", "groovy", "outstanding",
                    "keen",
                    "heavy", "slap-up", "peachy", "bang-up", "majuscule", "not_bad", "neat"],
        'Posted': ["put_up", "send", "mail", "post", "station", "place", "carry", "brand", "posted", "stake"],
        'Take': ["charter", "consume", "subscribe", "admit", "direct", "takings", "study", "pick_out", "strike",
                 "train",
                 "pack", "use_up", "ask", "hold", "submit", "exact", "lease", "take_up", "demand", "take_in", "film",
                 "require", "get_hold_of", "have", "subscribe_to", "carry", "hire", "choose", "need", "assume", "fill",
                 "necessitate", "call_for", "rent", "aim", "occupy", "take", "look_at", "make", "ingest", "contract",
                 "take_aim", "lead", "bring", "take_away", "drive", "accept", "contain", "postulate", "payoff", "claim",
                 "take_on", "shoot", "select", "adopt", "engage", "yield", "convey", "acquire", "consider", "read",
                 "involve", "issue", "remove", "learn", "get", "conduct", "withdraw", "return", "deal", "proceeds",
                 "guide"],
        'Utility': ["utility-grade", "utility", "public-service_corporation", "utility_program", "public_utility",
                    "public_utility_company", "service_program", "substitute", "usefulness"],
        'Enter': ["go_in", "enroll", "record", "insert", "move_into", "put_down", "accede", "embark", "enter",
                  "recruit",
                  "come_in", "get_into", "inscribe", "participate", "introduce", "get_in", "enrol", "figure", "infix",
                  "go_into"],
        'Convict': ["yard_bird", "con", "convict", "inmate", "yardbird"],
        'Key': ["key_fruit", "central", "primal", "key", "cay", "tonality", "cardinal", "identify", "headstone",
                "keystone",
                "key_out", "winder", "distinguish", "fundamental", "Francis_Scott_Key", "Key", "Florida_key", "paint",
                "name", "samara", "describe", "discover"],
        'Commit': ["send", "place", "put", "dedicate", "pull", "charge", "institutionalise", "practice", "trust",
                   "institutionalize", "give", "entrust", "confide", "devote", "commit", "consecrate", "invest",
                   "perpetrate", "intrust"],
        'Viol': ["viol"],
        'Drivers': ["device_driver", "number_one_wood", "driver"],
        'PB': ["pebibyte", "Pbit", "PB", "lead", "petabit", "PiB", "Pb", "petabyte", "atomic_number_82"],
        '22': ["XXII", "22", "twenty-two", "xxii"],
        'Subs': ["Italian_sandwich", "fill_in", "U-boat", "bomber", "hero", "hoagy", "poor_boy", "Cuban_sandwich",
                 "hoagie",
                 "grinder", "sub", "submarine", "stand_in", "substitute", "hero_sandwich", "zep", "wedge",
                 "submarine_sandwich", "torpedo", "pigboat"],
        'Enforcement': ["enforcement"],
        'Amp': ["amp", "adenylic_acid", "AMP", "ampere", "A", "adenosine_monophosphate"],
        'Call': ["squall", "foretell", "scream", "visit", "song", "address", "vociferation", "telephone", "hollo",
                 "birdsong", "call_up", "promise", "call_in", "outcry", "shout_out", "send_for", "margin_call", "phone",
                 "yell", "holler", "call_off", "call", "phone_call", "anticipate", "shout", "birdcall", "prognosticate",
                 "telephone_call", "bid", "ring", "claim", "Call", "call_option", "name", "predict", "forebode", "cry"],
        'Prostitution': ["prostitution", "whoredom", "harlotry"],
        'Medium': ["metier", "culture_medium", "intermediate", "mass_medium", "average", "medium", "spiritualist",
                   "sensitive"],
        'Asian': ["Asian", "Asiatic"],
        'Simulation': ["feigning", "simulation", "pretense", "model", "computer_simulation", "pretending", "pretence"],
        'LEO': ["Leo_the_Lion", "Lion", "Leo"],
        'another': ["another", "some_other"],
        'Hired': ["charter", "take", "leased", "engage", "hire", "hired", "rent", "lease", "chartered", "employ"],
        'by': ["by", "away", "aside", "past"],
        'Manufacture': ["fabricate", "industry", "manufacture", "make_up", "invent", "manufacturing", "cook_up",
                        "construct", "fabrication"],
        'Procure': ["secure", "pander", "procure", "pimp"],
        'Escape': ["escape_cock", "get_away", "outflow", "escape_valve", "scarper", "hightail_it", "get_off",
                   "escapism",
                   "take_to_the_woods", "miss", "leak", "bunk", "fly_the_coop", "dodging", "evasion", "lam", "get_by",
                   "get_out", "head_for_the_hills", "break_loose", "scat", "run_away", "break_away", "leakage",
                   "relief_valve", "escape", "run", "safety_valve", "turn_tail", "elude", "flight"],
        'Lost': ["helpless", "mazed", "drop_off", "recede", "confused", "deep_in_thought", "mixed-up", "confounded",
                 "bemused", "bewildered", "befuddled", "fall_back", "miss", "fall_behind", "lost", "doomed", "lose",
                 "suffer", "turn_a_loss", "preoccupied", "missed", "disoriented", "misplace", "at_sea", "baffled",
                 "mislay"],
        'At': ["At", "atomic_number_85", "at", "astatine"],
        'Disorderly': ["jumbled", "disorderly", "higgledy-piggledy", "chaotic", "hugger-mugger", "topsy-turvy"],
        'Prior': ["anterior", "prior"],
        'Eluding': ["put_off", "elusion", "circumvent", "duck", "parry", "slip", "skirt", "sidestep", "eluding",
                    "dodge",
                    "escape", "fudge", "evade", "bilk", "hedge", "elude"],
        'Bus': ["double-decker", "motorcoach", "busbar", "heap", "omnibus", "bus_topology", "charabanc", "jitney",
                "bus",
                "motorbus", "coach", "autobus", "passenger_vehicle", "jalopy"],
        'Invasion': ["encroachment", "intrusion", "invasion"],
        'Resist': ["refuse", "fend", "hold_out", "stand_firm", "stand", "reject", "jib", "protest", "defy", "dissent",
                   "resist", "balk", "baulk", "withstand"],
        'Secure': ["safe", "dependable", "batten_down", "guarantee", "unattackable", "procure", "plug", "strong",
                   "fasten",
                   "impregnable", "batten", "stop_up", "insure", "inviolable", "assure", "fix", "untroubled",
                   "unafraid",
                   "unassailable", "ensure", "good", "secure"],
        'Driver': ["device_driver", "number_one_wood", "driver"],
        'Draft': ["draft", "rough_drawing", "draught", "enlist", "muster_in", "order_of_payment", "draft_copy",
                  "blueprint",
                  "tipple", "conscription", "muster", "selective_service", "potation", "gulp", "swig", "outline",
                  "bill_of_exchange", "drawing"],
        'Tobacco': ["baccy", "tobacco_plant", "tobacco"],
        'Offend': ["spite", "shock", "break", "injure", "transgress", "violate", "scandalise", "hurt", "appall",
                   "outrage",
                   "appal", "scandalize", "pique", "offend", "wound", "go_against", "infract", "breach", "bruise"],
        'Hrs': ["60_minutes", "hr", "hour"],
        'Prescript': ["prescript", "rule"],
        'Abet': ["abet"],
        'Landing': ["shore", "land", "landing_place", "landing", "bring", "set_ashore", "shoot_down", "bring_down",
                    "down",
                    "put_down", "set_down"],
        'Methamphetamine': ["ice", "methamphetamine_hydrochloride", "chicken_feed", "methamphetamine", "trash",
                            "deoxyephedrine", "Methedrine", "shabu", "crank", "glass", "meth", "chalk"],
        'Fireman': ["firefighter", "reliever", "stoker", "fire_fighter", "fire-eater", "relief_pitcher", "fireman"],
        'Fleeing': ["take_flight", "flee", "fly"],
        'Public': ["public", "populace", "world"],
        'Unlicensed': ["unaccredited", "unlicensed", "unlicenced"],
        'LSD': ["lysergic_acid_diethylamide", "LSD"],
        'Live': ["unrecorded", "experience", "endure", "survive", "bouncy", "alive", "resilient", "hot", "last",
                 "exist",
                 "dwell", "lively", "know", "go", "be", "live_on", "live", "subsist", "hold_up", "populate", "hold_out",
                 "springy", "inhabit"],
        'Register': ["record", "cash_register", "read", "registry", "show", "cross-file", "file", "register"],
        'Cards': ["carte_du_jour", "card_game", "bill", "wag", "wit", "posting", "tease", "batting_order", "cards",
                  "identity_card", "carte", "visiting_card", "calling_card", "circuit_board", "bill_of_fare",
                  "scorecard",
                  "menu", "add-in", "card", "circuit_card", "notice", "placard", "lineup", "plug-in", "board",
                  "poster"],
        'Payment': ["defrayment", "requital", "defrayal", "payment"],
        'Cred': ["street_credibility", "street_cred", "cred"],
        'charge': ["send", "turn_on", "load", "commission", "kick", "commove", "bill", "consign", "boot",
                   "electric_charge",
                   "rush", "blame", "excite", "care", "buck", "appoint", "mission", "lodge", "charge_up", "charge",
                   "point",
                   "heraldic_bearing", "file", "saddle", "burden", "thrill", "institutionalise", "armorial_bearing",
                   "tear",
                   "institutionalize", "complaint", "flush", "accuse", "agitate", "bang", "rouse", "bearing", "level",
                   "explosive_charge", "bear_down", "shoot", "bursting_charge", "commit", "tutelage", "accusation",
                   "guardianship", "shoot_down", "billing", "direction", "burster", "cathexis"],
        'Fraudulent': ["fraudulent", "deceitful", "fallacious"],
        'Loiter': ["mess_about", "linger", "mill_around", "lollygag", "loaf", "footle", "loiter", "lurk", "hang_around",
                   "lounge", "lallygag", "mill_about", "tarry"],
        'Sell': ["sell", "deal", "trade", "betray"],
        'Forged': ["work", "contrive", "fake", "spirt", "hammer", "mold", "mould", "forged", "spurt", "shape", "bad",
                   "forge", "fashion", "excogitate", "formulate", "counterfeit", "form", "invent", "devise"],
        'Home': ["family", "base", "abode", "place", "interior", "household", "nursing_home", "rest_home", "internal",
                 "habitation", "plate", "national", "dwelling_house", "domicile", "home_plate", "home_base", "dwelling",
                 "home", "house", "menage"],
        'Wearing': ["bear", "put_on", "break", "weary", "outwear", "endure", "bust", "jade", "wear_thin", "tire",
                    "assume",
                    "get_into", "wear_down", "wearying", "eroding", "exhausting", "wear_upon", "fag_out", "erosion",
                    "have_on", "wear_off", "hold_out", "eating_away", "wear", "fall_apart", "fag", "tire_out",
                    "fatigue",
                    "wearing", "wear_out", "don", "tiring", "wearing_away"],
        'Elder': ["elder", "elderberry_bush", "sr.", "senior", "older"],
        'Pregnant': ["fraught", "pregnant", "significant", "meaning"],
        '200': ["two_hundred", "cc", "200"],
        '400': ["cd", "four_hundred", "400"],
        'Fare': ["get_along", "menu", "make_out", "transportation", "fare", "come", "do"],
        'REGISTERED': ["record", "show", "read", "cross-file", "file", "registered", "register"],
        'Bodily': ["corporal", "corporeal", "bodily", "somatic"],
        'Consideration': ["consideration", "condition", "circumstance", "thoughtfulness", "retainer",
                          "considerateness"],
        'Refuse': ["refuse", "turn_away", "scraps", "pass_up", "reject", "turn_down", "defy", "garbage", "resist",
                   "food_waste", "deny", "decline"],
        'Operating': ["work", "maneuver", "operate", "engage", "control", "operate_on", "lock", "manoeuver",
                      "manoeuvre",
                      "in_operation", "run", "operational", "mesh", "operating", "function", "go"],
        'Audio': ["sound_recording", "audio_recording", "audio", "sound", "audio_frequency"],
        'Care': ["give_care", "like", "handle", "precaution", "wish", "care", "charge", "attention", "concern",
                 "tending",
                 "caution", "worry", "fear", "upkeep", "manage", "tutelage", "aid", "maintenance", "guardianship",
                 "forethought", "deal"],
        'Warning': ["admonish", "warning", "monition", "monitory", "exemplary", "warn", "admonitory", "cautionary",
                    "admonition", "monish", "word_of_advice", "discourage"],
        'Identity': ["personal_identity", "indistinguishability", "identity_operator", "identicalness", "identity",
                     "individuality", "identity_element"],
        'Shoot': ["photograph", "snap", "scud", "buck", "charge", "film", "blast", "dash", "dart", "dissipate", "fool",
                  "pullulate", "tear", "take", "inject", "burgeon_forth", "flash", "spud", "pip", "shoot", "hit",
                  "bourgeon", "frivol_away", "fritter", "sprout", "shoot_down", "germinate", "fritter_away", "scoot",
                  "fool_away"],
        'Panhandle': ["panhandle"],
        'Sticker': ["stumper", "gummed_label", "paster", "poser", "toughie", "spine", "sticker", "spikelet", "dagger",
                    "prickle", "thorn", "pricker"],
        'Detainee': ["political_detainee", "detainee"],
        'Carjacking': ["carjacking", "carjack"],
        'Attempt': ["attack", "effort", "set_about", "assay", "seek", "undertake", "endeavour", "essay", "attempt",
                    "try",
                    "endeavor"],
        'Stalking': ["stalk", "still_hunt", "haunt", "stalking"],
        'Process': ["work", "serve", "cognitive_process", "action", "treat", "procedure", "work_on",
                    "unconscious_process",
                    "march", "operation", "litigate", "cognitive_operation", "outgrowth", "appendage",
                    "physical_process",
                    "summons", "mental_process", "swear_out", "process", "sue"],
        'Drink': ["boozing", "drink_in", "booze", "drink", "imbibe", "pledge", "potable", "tope", "deglutition",
                  "wassail",
                  "drunkenness", "fuddle", "drinking", "swallow", "salute", "toast", "drinkable", "beverage",
                  "crapulence"],
        'Vehicular': ["vehicular"],
        'Resident': ["house_physician", "resident", "occupier", "occupant", "resident_physician", "nonmigratory"],
        'Littering': ["litter"],
        'Concealed': ["hide", "hold_in", "conceal", "hidden", "hold_back", "concealed", "out_of_sight"],
        'BOX': ["corner", "box_seat", "box", "package", "loge", "boxwood", "boxful"],
        'A': ["angstrom_unit", "angstrom", "vitamin_A", "amp", "group_A", "adenine", "deoxyadenosine_monophosphate",
              "a",
              "axerophthol", "ampere", "type_A", "antiophthalmic_factor", "A"],
        'in': ["Hoosier_State", "in", "inward", "indium", "atomic_number_49", "IN", "Indiana", "inch", "inwards", "In"],
        'Goods': ["trade_good", "good", "goodness", "commodity"],
        'Off': ["bump_off", "hit", "away", "turned", "murder", "off", "remove", "dispatch", "cancelled", "slay",
                "forth",
                "sour", "polish_off"],
        'Ma': ["momma", "milliampere", "mom", "ma", "Massachusetts", "mammy", "MA", "Old_Colony", "Master_of_Arts",
               "mum",
               "mommy", "mamma", "mummy", "mama", "AM", "mA", "Bay_State", "Artium_Magister"],
        'Prohibited': ["interdict", "banned", "disallow", "proscribed", "taboo", "out", "prohibited", "tabu",
                       "verboten",
                       "prohibit", "forbid", "proscribe", "forbidden", "nix", "veto"],
        'Flee': ["take_flight", "flee", "fly"],
        'Attend': ["take_care", "advert", "see", "wait_on", "pay_heed", "attend_to", "serve", "hang", "go_to", "assist",
                   "give_ear", "look", "attend"],
        'Note': ["short_letter", "line", "observe", "note", "government_note", "bill", "billet", "Federal_Reserve_note",
                 "banker's_bill", "bank_note", "mention", "take_down", "notation", "tone", "banknote", "take_note",
                 "distinction", "promissory_note", "preeminence", "note_of_hand", "musical_note", "annotation",
                 "eminence",
                 "greenback", "mark", "bank_bill", "remark", "notice"],
        'Morphine': ["morphia", "morphine"],
        'Fentanyl': ["Sublimaze", "Fentanyl"],
        'Traffic': ["dealings", "traffic"],
        'Report': ["report", "story", "paper", "theme", "cover", "account", "written_report", "news_report", "write_up",
                   "report_card", "describe", "reputation", "composition", "study"],
        'Unnatural': ["abnormal", "unnatural", "affected"],
        'While': ["patch", "piece", "while", "spell"],
        'Spouse': ["partner", "spouse", "married_person", "mate", "better_half"],
        'Weapon': ["artillery", "arm", "weapon_system", "weapon"],
        'Solicit': ["romance", "hook", "court", "solicit", "woo", "accost", "beg", "tap"],
        'Intellectual': ["rational", "intellectual", "cerebral", "intellect", "noetic"],
        'case': ["cause", "vitrine", "display_case", "fount", "eccentric", "instance", "pillow_slip", "face", "type",
                 "subject", "character", "suit", "font", "caseful", "encase", "typesetter's_case", "slip", "showcase",
                 "guinea_pig", "incase", "grammatical_case", "case", "compositor's_case", "lawsuit", "event", "casing",
                 "sheath", "causa", "shell", "example", "typeface", "pillowcase"],
        'Pos': ["United_States_Post_Office", "polonium", "P.O.", "Post_Office", "petty_officer", "Po_River", "PO",
                "atomic_number_84", "Po", "US_Post_Office"],
        'Sexual': ["sexual", "intimate"],
        'Delivery': ["obstetrical_delivery", "rescue", "bringing", "deliverance", "manner_of_speaking", "livery",
                     "saving",
                     "legal_transfer", "pitch", "speech", "delivery"],
        'Falsely': ["falsely", "incorrectly"],
        'Hire': ["charter", "take", "engage", "hire", "rent", "lease", "employ"],
        'Presence': ["front", "presence", "mien", "bearing", "comportment"],
        'In': ["Hoosier_State", "in", "inward", "indium", "atomic_number_49", "IN", "Indiana", "inch", "inwards", "In"],
        'Molest': ["harass", "chivvy", "chevvy", "beset", "plague", "chevy", "molest", "hassle", "provoke", "harry",
                   "chivy"],
        'Reports': ["report", "story", "paper", "theme", "cover", "account", "written_report", "news_report",
                    "write_up",
                    "report_card", "describe", "reputation", "composition", "study"],
        'Habitual': ["habitual", "wonted", "accustomed", "customary"],
        'II': ["two", "ii", "2", "deuce", "II"],
        'Other': ["other", "early", "former"],
        'Expired': ["give-up_the_ghost", "snuff_it", "conk", "exit", "expire", "expired", "croak", "choke",
                    "kick_the_bucket", "drop_dead", "cash_in_one's_chips", "perish", "go", "decease", "exhale",
                    "run_out",
                    "breathe_out", "buy_the_farm", "pop_off", "pass_away", "die", "pass"],
        'Racing': ["rush_along", "race", "pelt_along", "belt_along", "hie", "hotfoot", "run", "step_on_it", "rush",
                   "speed",
                   "cannonball_along", "hasten", "racing", "bucket_along"],
        'Transport': ["send", "shipping", "ecstasy", "conveyance", "channelize", "enthrall", "exaltation", "enthral",
                      "transport", "raptus", "transfer", "channel", "tape_drive", "delight", "rapture", "carry",
                      "enrapture", "transportation", "ravish", "ship", "enchant", "transmit", "tape_transport",
                      "channelise", "transferral"],
        'Worthless': ["vile", "unworthy", "wretched", "slimy", "despicable", "ugly", "worthless"],
        'Repeat': ["reprise", "echo", "take_over", "repetition", "reprize", "reduplicate", "recur", "ingeminate",
                   "repeat",
                   "iterate", "retell", "double", "duplicate", "replicate", "reiterate", "recapitulate", "restate"],
        'Baiting': ["cod", "rag", "bait", "twit", "tantalize", "ride", "razz", "taunt", "rally", "tantalise", "tease",
                    "baiting"],
        'Alcoholic': ["soaker", "boozer", "dipsomaniac", "alky", "alcohol-dependent", "lush", "alcoholic", "souse"],
        'Cruelty': ["ruthlessness", "mercilessness", "cruelness", "harshness", "inhuman_treatment", "cruelty",
                    "pitilessness"],
        'Prescription': ["prescription_medicine", "ethical_drug", "prescription_drug", "prescription"],
        'Safety': ["safe", "rubber", "safety", "guard", "prophylactic", "refuge", "safety_device", "base_hit",
                   "condom"],
        'Armed': ["armed", "arm", "gird", "fortify", "build_up"],
        'Exposure': ["pic", "photograph", "photo", "vulnerability", "picture", "exposure"],
        'Hydromorphone': ["hydromorphone", "Dilaudid", "hydromorphone_hydrochloride"],
        'Proof': ["proof", "test_copy", "validation", "substantiation", "proofread", "trial_impression",
                  "cogent_evidence"],
        'Res': ["rhenium", "Re", "RES", "atomic_number_75", "reticuloendothelial_system", "re", "Ra", "ray"],
        'DUI': ["yoke", "couple", "duo", "twosome", "duad", "duet", "dyad", "twain", "duette", "brace", "span", "pair",
                "distich", "couplet"],
        'Articles': ["clause", "article"],
        'Older': ["honest-to-god", "sometime", "sure-enough", "former", "elderly", "elder", "old", "quondam", "Old",
                  "previous", "sr.", "aged", "honest-to-goodness", "senior", "one-time", "onetime", "erstwhile",
                  "older"],
        'Defrauding': ["diddle", "mulct", "scam", "victimize", "bunco", "con", "swindle", "rook", "nobble", "defraud",
                       "gyp", "short-change", "goldbrick", "hornswoggle", "gip"],
        'O': ["group_O", "type_O", "O", "o", "oxygen", "atomic_number_8"],
        'Computer': ["data_processor", "reckoner", "computing_device", "computer", "calculator",
                     "information_processing_system", "figurer", "computing_machine", "electronic_computer",
                     "estimator"],
        'Defraud': ["diddle", "mulct", "scam", "victimize", "bunco", "con", "swindle", "rook", "nobble", "defraud",
                    "gyp",
                    "short-change", "goldbrick", "hornswoggle", "gip"],
        'Compulsory': ["mandatory", "required", "compulsory"],
        'Crime': ["offense", "criminal_offence", "criminal_offense", "law-breaking", "crime", "offence"],
        'Lodging': ["lodging", "housing", "lodge", "charge", "accommodate", "lodgment", "deposit", "wedge",
                    "living_accommodations", "file", "stick", "lodgement"],
        'Sign': ["signboard", "subscribe", "signalize", "signalise", "planetary_house", "signal", "sign_on", "gestural",
                 "augury", "sign_of_the_zodiac", "preindication", "signed", "ratify", "contract", "signaling",
                 "polarity",
                 "bless", "sign-language", "mark", "sign", "sign_up", "star_sign", "house", "foretoken", "mansion"],
        'Altered': ["altered", "vary", "modify", "spay", "change", "castrate", "neutered", "interpolate", "adapted",
                    "alter", "neuter", "falsify"],
        'Personal': ["personal"],
        'Felon': ["felon", "criminal", "outlaw", "whitlow", "malefactor", "crook"],
        'Prostitute': ["working_girl", "sporting_lady", "cocotte", "cyprian", "bawd", "lady_of_pleasure", "tart",
                       "whore",
                       "prostitute", "woman_of_the_street", "fancy_woman", "harlot"],
        'Elude': ["put_off", "circumvent", "duck", "parry", "skirt", "sidestep", "dodge", "escape", "fudge", "elude",
                  "evade", "hedge", "bilk"],
        'Cannabis': ["marihuana", "marijuana", "cannabis", "hemp", "ganja"],
        'Pretrial': ["pretrial_conference", "pretrial"],
        'Contradict': ["contradict", "belie", "negate", "oppose", "controvert", "contravene"],
        'Alter': ["vary", "modify", "spay", "change", "castrate", "interpolate", "alter", "neuter", "falsify"],
        'o': ["group_O", "type_O", "O", "o", "oxygen", "atomic_number_8"],
        'Aide': ["aide", "auxiliary", "aide-de-camp", "adjutant"],
        'Offender': ["wrongdoer", "offender"],
        'IC': ["ic", "ninety-nine", "United_States_Intelligence_Community", "IC", "Intelligence_Community",
               "National_Intelligence_Community", "99"],
        'Negligence': ["carelessness", "nonperformance", "neglectfulness", "negligence", "neglect"],
        '21': ["twenty-one", "xxi", "XXI", "21"],
        'Occupy': ["occupy", "take", "engage", "engross", "worry", "reside", "concern", "invade", "use_up", "absorb",
                   "busy", "interest", "fill", "lodge_in"],
        'Controlled': ["keep_in_line", "hold", "master", "ascertain", "see_to_it", "control", "insure", "assure",
                       "curb",
                       "see", "operate", "manipulate", "contain", "verify", "moderate", "ensure", "hold_in",
                       "controlled",
                       "check", "command"],
        'Or': ["operating_theater", "OR", "surgery", "Beaver_State", "operating_room", "Oregon", "operating_theatre"],
        'Drugs': ["do_drugs", "drug", "dose"],
        'B': ["b", "Bel", "barn", "B_vitamin", "B-complex_vitamin", "B_complex", "vitamin_B_complex", "atomic_number_5",
              "bacillus", "type_B", "vitamin_B", "boron", "B", "group_B"],
        '4': ["Little_Joe", "tetrad", "foursome", "quadruplet", "quaternary", "4", "quatern", "quaternion", "iv",
              "quartet",
              "quaternity", "four", "IV"],
        '14': ["14", "fourteen", "XIV", "xiv"],
        'Sex': ["excite", "sexual_urge", "turn_on", "sex", "sexual_practice", "sexual_activity", "sex_activity",
                "arouse",
                "wind_up", "sexuality", "gender"],
        'Vessel': ["vas", "vessel", "watercraft"],
        'Badges': ["badge"],
        'Farm': ["raise", "produce", "farm", "grow"],
        'Alcohol': ["inebriant", "alcoholic_drink", "alcohol", "alcoholic_beverage", "intoxicant"],
        'Invest': ["commit", "induct", "seat", "adorn", "enthrone", "endow", "gift", "indue", "place", "invest",
                   "empower",
                   "endue", "put", "vest", "clothe"],
        'Wear': ["bear", "put_on", "break", "weary", "outwear", "endure", "vesture", "bust", "jade", "wear_thin",
                 "tire",
                 "assume", "get_into", "habiliment", "don", "wear_down", "clothing", "wear_upon", "fag_out", "have_on",
                 "wear_off", "wearable", "hold_out", "fall_apart", "wear", "fag", "tire_out", "fatigue", "wearing",
                 "wear_out", "article_of_clothing"],
        'Elderly': ["elderly", "senior", "aged", "older"],
        'Disturb': ["stir_up", "upset", "raise_up", "agitate", "disturb", "interrupt", "commove", "vex", "trouble",
                    "shake_up", "touch"],
        'firearm': ["firearm", "piece", "small-arm"],
        'no': ["No", "atomic_number_102", "no_more", "no", "nobelium"],
        'Monitor': ["proctor", "monitor", "Monitor", "monitoring_device", "reminder", "admonisher", "monitor_lizard",
                    "varan", "supervise"],
        'Disguise': ["camouflage", "disguise", "mask"],
        'while': ["patch", "piece", "while", "spell"],
        'Deceased': ["give-up_the_ghost", "dead_person", "snuff_it", "conk", "exit", "expire", "dead_soul", "croak",
                     "choke", "kick_the_bucket", "drop_dead", "cash_in_one's_chips", "perish", "go", "decease",
                     "at_peace",
                     "asleep", "deceased", "deceased_person", "decedent", "buy_the_farm", "pop_off", "pass_away",
                     "at_rest",
                     "die", "departed", "pass", "gone"],
        'Similitude': ["likeness", "counterpart", "alikeness", "similitude", "twin"],
        'Handcuff': ["handcuff", "manacle", "handlock", "cuff"],
        'Strike': ["smasher", "run_into", "chance_upon", "expunge", "fall_upon", "strike", "move", "take_up",
                   "chance_on",
                   "fall", "strickle", "light_upon", "collide_with", "assume", "mint", "tap", "take", "scratch", "bang",
                   "rap", "come_to", "come_upon", "come_across", "excise", "coin", "impinge_on", "happen_upon",
                   "walk_out",
                   "shine", "impress", "hit", "work_stoppage", "ten-strike", "affect", "attain", "discover", "smash"],
        'Beg': ["pray", "solicit", "implore", "beg", "tap"],
        'Amobarbital': ["amobarbital"],
        'Beverage': ["drink", "beverage", "drinkable", "potable"],
        'Can': ["give_notice", "fire", "lavatory", "bathroom", "sack", "displace", "hindquarters", "give_the_axe",
                "toilet",
                "dismiss", "can", "privy", "stern", "seat", "rear_end", "rump", "john", "posterior", "arse", "buns",
                "pot",
                "lav", "nates", "tail", "give_the_sack", "stool", "rear", "tail_end", "butt", "terminate", "hind_end",
                "prat", "derriere", "behind", "backside", "put_up", "throne", "bum", "ass", "bottom", "potty",
                "fundament",
                "buttocks", "crapper", "force_out", "commode", "fanny", "tin_can", "send_away", "tin", "keister",
                "tush",
                "can_buoy", "tooshie", "canful"],
        '16': ["sixteen", "xvi", "XVI", "16"],
        'Insurance': ["insurance_policy", "indemnity", "policy", "insurance"],
        'Drive': ["cause", "ram", "repulse", "crusade", "beat_back", "effort", "movement", "driving_force", "repel",
                  "aim",
                  "push_back", "push", "driving", "force_back", "take", "thrust", "campaign", "parkway", "ride",
                  "driveway",
                  "labor", "drive", "tug", "labour", "force", "motor", "get", "private_road"],
        'Gambling': ["take_a_chance", "play", "chance", "adventure", "hazard", "run_a_risk", "gambling", "gamble",
                     "take_chances", "gaming", "risk"],
        'Female': ["female_person", "female", "distaff"],
        'Level': ["story", "even", "degree", "rase", "stratum", "plane", "tied", "tear_down", "floor", "take_down",
                  "level_off", "point", "stage", "grade", "spirit_level", "tier", "charge", "storey", "flat",
                  "even_out",
                  "flush", "unwavering", "pull_down", "level", "dismantle", "horizontal_surface", "layer", "raze"],
        'Deft': ["dextrous", "dexterous", "deft"],
        'Missile': ["missile", "projectile"],
        'Aiding': ["help", "aid", "assist"],
        'Pay': ["remuneration", "bear", "devote", "compensate", "pay_up", "yield", "make_up", "ante_up", "salary",
                "pay",
                "wage", "earnings", "give", "pay_off"],
        'Under': ["nether", "under", "below"],
        'Juvenile': ["juvenile", "juvenile_person", "adolescent", "puerile", "jejune"],
        'Railroad': ["railroad_line", "railway_system", "railway", "railroad", "railroad_track", "dragoon", "sandbag",
                     "railway_line"],
        'Lorazepam': ["lorazepam", "Ativan"],
        'Utilizing': ["use", "apply", "utilise", "utilize", "employ"],
        'Occupied': ["occupy", "take", "tenanted", "engaged", "engage", "engross", "worry", "reside", "concern",
                     "invade",
                     "use_up", "absorb", "busy", "occupied", "interest", "fill", "lodge_in"],
        'Felony': ["felony"],
        'Unoccupied': ["unoccupied", "untenanted"],
        'Murder': ["bump_off", "hit", "execution", "murder", "off", "remove", "dispatch", "mangle", "mutilate",
                   "slaying",
                   "slay", "polish_off"],
        'Methadone': ["methadone_hydrochloride", "synthetic_heroin", "dolophine_hydrochloride", "fixer", "methadon",
                      "methadone"],
        'Highway': ["main_road", "highway"],
        'Urge': ["inspire", "itch", "urge_on", "advocate", "root_on", "impulse", "exhort", "pep_up", "recommend",
                 "cheer",
                 "press", "barrack", "urge"],
        'Law': ["jurisprudence", "law_of_nature", "legal_philosophy", "natural_law", "police", "police_force",
                "constabulary", "law", "practice_of_law"],
        'POSSESS': ["have", "possess", "own"],
        'mask': ["cloak", "disguise", "masquerade", "mask", "dissemble", "masque", "block_out", "masquerade_party"],
        'Imprisonment': ["captivity", "incarceration", "imprisonment", "immurement", "internment"],
        'Hiring': ["charter", "take", "engage", "hire", "rent", "lease", "employ"],
        'Near': ["cheeseparing", "most", "draw_near", "come_on", "about", "nearly", "go_up", "skinny", "approximate",
                 "penny-pinching", "virtually", "close", "almost", "dear", "draw_close", "nigh", "well-nigh",
                 "come_near",
                 "near", "good", "approach"],
        'Violence': ["force", "vehemence", "fierceness", "ferocity", "wildness", "fury", "furiousness", "violence"],
        'Unattended': ["unattended", "neglected"],
        'Dealer': ["trader", "principal", "dealer", "bargainer", "monger"],
        'Illegal': ["illegal"],
        'System': ["system", "organisation", "system_of_rules", "scheme", "organization", "arrangement"],
        'Wildlife': ["wildlife"],
        'Attendance': ["attendance", "attending"],
        'Provide': ["put_up", "provide", "furnish", "offer", "leave", "bring_home_the_bacon", "cater", "allow_for",
                    "allow",
                    "supply", "ply", "render"],
        'Video': ["video_recording", "TV", "television", "telecasting", "picture", "video"],
        '1': ["ace", "1", "single", "unity", "i", "ane", "one", "I"],
        'Min': ["Min_dialect", "Min", "Fukien", "Fukkianese", "Amoy", "minute", "min", "Taiwanese", "Hokkianese"],
        'Engage': ["charter", "prosecute", "lease", "plight", "engross", "affiance", "hire", "rent", "employ", "occupy",
                   "take", "enlist", "operate", "absorb", "wage", "engage", "lock", "betroth", "mesh", "pursue"],
        'False': ["put_on", "treacherously", "faithlessly", "fake", "imitation", "false", "fictive", "off-key",
                  "traitorously", "pretended", "sham", "untrue", "simulated", "delusive", "faux", "assumed", "mistaken",
                  "treasonably", "sour", "fictitious"],
        'Name': ["key", "constitute", "identify", "mention", "distinguish", "key_out", "nominate", "refer", "call",
                 "advert", "public_figure", "make", "epithet", "gens", "diagnose", "list", "name", "cite", "figure",
                 "describe", "bring_up", "discover", "appoint"],
    }
    with open("../dataset/NLP/compas/data/adv_synonyms_dic.pkl", 'wb') as f:
        pickle.dump(adv_synonyms_dic, f)
    adv_synonyms_text = []
    for k in adv_synonyms_dic:
        # if k not in adv_synonyms_text:
        #     adv_synonyms_text.append(k)
        for v in adv_synonyms_dic[k]:
            if v not in adv_synonyms_text:
                adv_synonyms_text.append(v)
    return [" ".join(adv_synonyms_text)]


# credit dataset


# ACSEmployment dataset
def change_employment_tabular_to_text(tabular_data, vocab_dic, fea_dim):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    text_data = []
    for i in range(16):
        if i in [0]:
            text_data.append(str(tabular_data[i]))
        elif i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            for k in vocab_dic[i].keys():
                if vocab_dic[i][k][0] == tabular_data[i]:
                    text_data.append(k)
    if len(text_data) == 16:
        return text_data
    else:
        return None


def ACSEmployment_tabular_to_text(template, item):
    """
    将employment数据集的表格数据转化为文本数据
    :return:
    """
    employment_text = template.format(item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8],
                                      item[9], item[10], item[11], item[12], item[13], item[14], item[15])

    return employment_text


def generate_ACSEmployment_text_data(data_file, race_file, gender_file, aug_file,
                                     text_data_file, text_label_file, text_race_file, text_gender_file, text_aug_file):
    """
    使用template模板将employment表格数据转换为文本text数据
    :return:
    """
    # 定义模板
    templates = [
        "The following individual context corresponds to a survey respondent. The survey was conducted among US "
        "residents in 2018. Please answer the question what was the person's total income in the past year.based on "
        "the information provided. "
        "Individual context:"
        "The age is: {}."
        "The educational attainment is: {}."
        "The marital status is: {}."
        "The relationship is: {}."
        "The disability recode is: {}."
        "The employment status of parents are: {}."
        "The citizenship status is: {}."
        "The mobility status is: {}."
        "The military service is: {}."
        "The ancestry recode is: {}."
        "The nativity is: {}."
        "The hearing difficulty is: {}."
        "The Vision difficulty is: {}."
        "The cognitive difficulty is: {}."
        "The gender is: {}."
        "The race is: {}."
    ]
    # 定义同义词词典
    synonyms = {}
    tabular_data = pandas.read_csv(data_file).values
    tabular_race = numpy.load(race_file)
    tabular_gender = numpy.load(gender_file)
    tabular_aug = numpy.load(aug_file)

    text_employment = []
    text_race = []
    text_gender = []
    text_aug = []
    text_label = tabular_data[:, 16]

    fea_dim = numpy.load("../dataset/ACS/employment/data/fea_dim.npy").tolist()
    with open("../dataset/ACS/employment/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)

    for i in range(tabular_data.shape[0]):
        # 随机选择一个模板，替换同义词
        selected_template = random.choice(templates)
        result_template = replace_synonyms(selected_template, synonyms)
        # 对表格数据进行转换
        item = change_employment_tabular_to_text(tabular_data[i], vocab_dic, fea_dim)
        text_employment.append(ACSEmployment_tabular_to_text(result_template, item))
        # 对race数据进行转换
        sim_race = tabular_race[i]
        race_result = []
        for j in range(len(sim_race)):
            race_result.append(ACSEmployment_tabular_to_text(result_template, sim_race[j]))
        text_race.append(race_result)
        # 对gender数据进行转换
        sim_gender = tabular_gender[i]
        gender_result = []
        for j in range(len(sim_gender)):
            gender_result.append(ACSEmployment_tabular_to_text(result_template, sim_gender[j]))
        text_gender.append(gender_result)
        # 对aug数据进行转换
        sim_aug = tabular_aug[i]
        aug_result = []
        for j in range(len(sim_aug)):
            aug_result.append(ACSEmployment_tabular_to_text(result_template, sim_aug[j]))
        text_aug.append(aug_result)

    numpy.save(text_data_file, text_employment)
    numpy.save(text_label_file, text_label)
    numpy.save(text_race_file, text_race)
    numpy.save(text_gender_file, text_gender)
    numpy.save(text_aug_file, text_aug)


def generate_ACSEmployment_adv_replace_synonyms():
    """
    生成employment数据集中所有文字的同义词替换，包括模板中的单词同义词，以及所有属性取值的同义词
    :return:
    """
    adv_synonyms_dic = {}
    # 定义模板
    templates = [
        "The following individual context corresponds to a survey respondent. The survey was conducted among US "
        "residents in 2018. Please answer the question what was the person's total income in the past year.based on "
        "the information provided. "
        "Individual context:"
        "The age is: {}."
        "The educational attainment is: {}."
        "The marital status is: {}."
        "The relationship is: {}."
        "The disability recode is: {}."
        "The employment status of parents are: {}."
        "The citizenship status is: {}."
        "The mobility status is: {}."
        "The military service is: {}."
        "The ancestry recode is: {}."
        "The nativity is: {}."
        "The hearing difficulty is: {}."
        "The Vision difficulty is: {}."
        "The cognitive difficulty is: {}."
        "The gender is: {}."
        "The race is: {}."
    ]

    for t in templates:
        t = re.sub(r'\{.*?\}', '', t)
        t = t.replace('-', ' ')
        t = re.sub(r'[^a-zA-Z\s]', '', t)
        words = t.split()
        for w in words:
            w_synonyms = get_replace_synonyms(w)
            if len(w_synonyms) > 0:
                if w not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
                    adv_synonyms_dic[w] = w_synonyms

    # 所有属性的取值
    vocab_dic = ["less than 3 years old", "No schooling completed", "Nursery school/preschool", "Kindergarten",
                 "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6", "Grade 7", "Grade 8", "Grade 9",
                 "Grade 10", "Grade 11", "12th Grade - no diploma", "Regular high school diploma",
                 "GED or alternative credential", "Some college but less than 1 year",
                 "1 or more years of college credit but no degree", "Associate's degree", "Bachelor's degree",
                 "Master's degree", "Professional degree beyond a bachelor's degree", "Doctorate degree", "Married",
                 "Widowed", "Divorced", "Separated", "Never married or under 15 years old", "Reference person",
                 "Husband/wife", "Biological son or daughter", "Adopted son or daughter", "Stepson or stepdaughter",
                 "Brother or sister", "Father or mother", "Grandchild", "Parent-in-law",
                 "Son-in-law or daughter-in-law", "Other relative", "Roomer or boarder", "Housemate or roommate",
                 "Unmarried partner", "Foster child", "Other non relative",
                 "Institutionalized group quarters population", "Non institutionalized group quarters population",
                 "With a disability", "Without a disability",
                 "not own child of householder, and not child in subfamily",
                 "Living with two parents: both parents in labor force",
                 "Living with two parents: Father only in labor force",
                 "Living with two parents: Mother only in labor force",
                 "Living with two parents: Neither parent in labor force",
                 "Living with father: Father in the labor force", "Living with father: Father not in labor force",
                 "Living with mother: Mother in the labor force", "Living with mother: Mother not in labor force",
                 "Born in the U.S.", "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas",
                 "Born abroad of American parent(s)", "U.S. citizen by naturalization", "Not a citizen of the U.S.",
                 "less than 1 year old", "Yes, same house ( non movers )", "No, outside US and Puerto Rico",
                 "No, different house in US or Puerto Rico", "less than 17 years old", "Now on active duty",
                 "On active duty in the past, but not now",
                 "Only on active duty for training in Reserves/National Guard", "Never served in the military",
                 "Single", "Multiple", "Unclassified", "Not reported", "Suppressed for data year 2018 for select PUMAs",
                 "Native", "Foreign born", "Yes", "No", "Male", "Female", "White alone",
                 "Black or African American alone", "American Indian alone", "Alaska Native alone",
                 "American Indian and Alaska Native tribes specified, or American Indian or Alaska Native, not specified and no other races",
                 "Asian alone", "Native Hawaiian and Other Pacific Islander alone", "Some Other Race alone",
                 "Two or More Races"]

    for v in vocab_dic:
        for vv in v.strip().split(" "):

            w_synonyms = get_replace_synonyms(vv)
            if len(w_synonyms) > 0:
                if vv not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
                    adv_synonyms_dic[vv] = w_synonyms

    with open("../dataset/ACS/NLP/employment/data/adv_synonyms_dic.pkl", 'wb') as f:
        pickle.dump(adv_synonyms_dic, f)
    adv_synonyms_text = []
    for k in adv_synonyms_dic:
        # if k not in adv_synonyms_text:
        #     adv_synonyms_text.append(k)
        for v in adv_synonyms_dic[k]:
            if v not in adv_synonyms_text:
                adv_synonyms_text.append(v)
    return [" ".join(adv_synonyms_text)]


# ACSIncome dataset
def change_income_tabular_to_text(tabular_data, vocab_dic, fea_dim):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    text_data = []
    for i in range(10):
        if i in [0, 4, 5, 7]:
            text_data.append(str(tabular_data[i]))
        elif i in [1, 2, 3, 6, 8, 9]:
            for k in vocab_dic[i].keys():
                if vocab_dic[i][k][0] == tabular_data[i]:
                    text_data.append(k)
    if len(text_data) == 10:
        return text_data
    else:
        return None


def ACSIncome_tabular_to_text(template, item):
    """
    将income数据集的表格数据转化为文本数据
    :return:
    """
    income_text = template.format(item[0], item[1], item[2], item[3], item[4],
                                  item[5], item[6], item[7], item[8], item[9])

    return income_text


def generate_ACSIncome_text_data(data_file, race_file, gender_file, aug_file,
                                 text_data_file, text_label_file, text_race_file, text_gender_file, text_aug_file):
    """
    使用template模板将income表格数据转换为文本text数据
    :return:
    """
    # 定义模板
    templates = [
        "The following individual context corresponds to a survey respondent. The survey was conducted among US "
        "residents in 2018. Please answer the question what was the person's total income in the past year.based on "
        "the information provided. "
        "Individual context:"
        "The age is: {}."
        "The class of worker is: {}."
        "The educational attainment is: {}."
        "The marital status is: {}."
        "The occupation is: {}."
        "The place of birth is: {}."
        "The relationship is: {}."
        "The usual hours worked per week past 12 months is: {}."
        "The gender is: {}."
        "The race is: {}."
    ]
    # 定义同义词词典
    synonyms = {}
    tabular_data = pandas.read_csv(data_file).values
    tabular_race = numpy.load(race_file)
    tabular_gender = numpy.load(gender_file)
    tabular_aug = numpy.load(aug_file)

    text_income = []
    text_race = []
    text_gender = []
    text_aug = []
    text_label = tabular_data[:, 10]

    fea_dim = numpy.load("../dataset/ACS/income/data/fea_dim.npy").tolist()
    with open("../dataset/ACS/income/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)

    for i in range(tabular_data.shape[0]):
        # 随机选择一个模板，替换同义词
        selected_template = random.choice(templates)
        result_template = replace_synonyms(selected_template, synonyms)
        # 对表格数据进行转换
        item = change_income_tabular_to_text(tabular_data[i], vocab_dic, fea_dim)
        text_income.append(ACSIncome_tabular_to_text(result_template, item))
        # 对race数据进行转换
        sim_race = tabular_race[i]
        race_result = []
        for j in range(len(sim_race)):
            race_result.append(ACSIncome_tabular_to_text(result_template, sim_race[j]))
        text_race.append(race_result)
        # 对gender数据进行转换
        sim_gender = tabular_gender[i]
        gender_result = []
        for j in range(len(sim_gender)):
            gender_result.append(ACSIncome_tabular_to_text(result_template, sim_gender[j]))
        text_gender.append(gender_result)
        # 对aug数据进行转换
        sim_aug = tabular_aug[i]
        aug_result = []
        for j in range(len(sim_aug)):
            aug_result.append(ACSIncome_tabular_to_text(result_template, sim_aug[j]))
        text_aug.append(aug_result)

    numpy.save(text_data_file, text_income)
    numpy.save(text_label_file, text_label)
    numpy.save(text_race_file, text_race)
    numpy.save(text_gender_file, text_gender)
    numpy.save(text_aug_file, text_aug)


def generate_ACSIncome_adv_replace_synonyms():
    """
    生成income数据集中所有文字的同义词替换，包括模板中的单词同义词，以及所有属性取值的同义词
    :return:
    """
    adv_synonyms_dic = {}
    # 定义模板
    templates = [
        "The following individual context corresponds to a survey respondent. The survey was conducted among US "
        "residents in 2018. Please answer the question what was the person's total income in the past year.based on "
        "the information provided. "
        "Individual context:"
        "The age is: {}."
        "The class of worker is: {}."
        "The educational attainment is: {}."
        "The marital status is: {}."
        "The Occupation is: {}."
        "The place of birth is: {}."
        "The relationship is: {}."
        "The usual hours worked per week past 12 months is: {}."
        "The gender is: {}."
        "The race is: {}."
    ]

    for t in templates:
        t = re.sub(r'\{.*?\}', '', t)
        t = t.replace('-', ' ')
        t = re.sub(r'[^a-zA-Z\s]', '', t)
        words = t.split()
        for w in words:
            w_synonyms = get_replace_synonyms(w)
            if len(w_synonyms) > 0:
                if w not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
                    adv_synonyms_dic[w] = w_synonyms

    # 所有属性的取值
    vocab_dic = ["not in universe",
                 "Employee of a private for-profit company or business, or of an individual, for wages, salary, "
                 "or commissions", "Employee of a private not-for-profit, tax-exempt, or charitable organization",
                 "Local government employee (city, county, etc.)", "State government employee",
                 "Federal government employee",
                 "Self-employed in own not incorporated business, professional practice, or farm",
                 "Self-employed in own incorporated business, professional practice or farm",
                 "Working without pay in family business or farm",
                 "Unemployed and last worked 5 years ago or earlier or never worked",
                 "less than 3 years old", "No schooling completed", "Nursery school/preschool", "Kindergarten",
                 "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6", "Grade 7", "Grade 8", "Grade 9",
                 "Grade 10", "Grade 11", "12th Grade - no diploma", "Regular high school diploma",
                 "GED or alternative credential", "Some college but less than 1 year",
                 "1 or more years of college credit but no degree", "Associate's degree", "Bachelor's degree",
                 "Master's degree", "Professional degree beyond a bachelor's degree", "Doctorate degree", "Married",
                 "Widowed", "Divorced", "Separated", "Never married or under 15 years old", "Reference person",
                 "Husband/wife", "Biological son or daughter", "Adopted son or daughter", "Stepson or stepdaughter",
                 "Brother or sister", "Father or mother", "Grandchild", "Parent-in-law",
                 "Son-in-law or daughter-in-law", "Other relative", "Roomer or boarder", "Housemate or roommate",
                 "Unmarried partner", "Foster child", "Other non relative",
                 "Institutionalized group quarters population", "Non institutionalized group quarters population",
                 "Male", "Female", "White alone",
                 "Black or African American alone", "American Indian alone", "Alaska Native alone",
                 "American Indian and Alaska Native tribes specified, or American Indian or Alaska Native, "
                 "not specified and no other races", "Asian alone", "Native Hawaiian and Other Pacific Islander alone",
                 "Some Other Race alone", "Two or More Races"]

    for v in vocab_dic:
        for vv in v.strip().split(" "):
            w_synonyms = get_replace_synonyms(vv)
            if len(w_synonyms) > 0:
                if vv not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
                    adv_synonyms_dic[vv] = w_synonyms

    with open("../dataset/ACS/NLP/income/data/adv_synonyms_dic.pkl", 'wb') as f:
        pickle.dump(adv_synonyms_dic, f)
    adv_synonyms_text = []
    for k in adv_synonyms_dic:
        for v in adv_synonyms_dic[k]:
            if v not in adv_synonyms_text:
                adv_synonyms_text.append(v)
    return [" ".join(adv_synonyms_text)]


# ACSCoverage dataset
def change_coverage_tabular_to_text(tabular_data, vocab_dic, fea_dim):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    text_data = []
    for i in range(19):
        if i in [0, 14, 16]:
            text_data.append(str(tabular_data[i]))
        elif i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]:
            for k in vocab_dic[i].keys():
                if vocab_dic[i][k][0] == tabular_data[i]:
                    text_data.append(k)
    if len(text_data) == 19:
        return text_data
    else:
        return None


def ACSCoverage_tabular_to_text(template, item):
    """
    将Coverage数据集的表格数据转化为文本数据
    :return:
    """
    coverage_text = template.format(item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8],
                                    item[9], item[10], item[11], item[12], item[13], item[14], item[15], item[16],
                                    item[17], item[18])

    return coverage_text


def generate_ACSCoverage_text_data(data_file, race_file, gender_file, aug_file,
                                   text_data_file, text_label_file, text_race_file, text_gender_file, text_aug_file):
    """
    使用template模板将Coverage表格数据转换为文本text数据
    :return:
    """
    # 定义模板
    templates = [
        "The following individual context corresponds to a survey respondent. The survey was conducted among US "
        "residents in 2018. Please answer the question what was the person's total income in the past year.based on "
        "the information provided. "
        "Individual context:"
        "The age is: {}."
        "The educational attainment is: {}."
        "The marital status is: {}."
        "The gender is: {}."
        "The disability recode is: {}."
        "The employment status of parents are: {}."
        "The citizenship status is: {}."
        "The mobility status is: {}."
        "The military service is: {}."
        "The ancestry recode is: {}."
        "The nativity is: {}."
        "The hearing difficulty is: {}."
        "The Vision difficulty is: {}."
        "The cognitive difficulty is: {}."
        "The total person’s income is: {}."
        "The employment status recode is: {}."
        "The state code is: {}."
        "The gave birth to child within the past 12 months is: {}."
        "The race is: {}."
    ]

    # 定义同义词词典
    synonyms = {}
    tabular_data = pandas.read_csv(data_file).values
    tabular_race = numpy.load(race_file)
    tabular_gender = numpy.load(gender_file)
    tabular_aug = numpy.load(aug_file)

    text_coverage = []
    text_race = []
    text_gender = []
    text_aug = []
    text_label = tabular_data[:, 19]

    fea_dim = numpy.load("../dataset/ACS/coverage/data/fea_dim.npy").tolist()
    with open("../dataset/ACS/coverage/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)

    for i in range(tabular_data.shape[0]):
        # 随机选择一个模板，替换同义词
        selected_template = random.choice(templates)
        result_template = replace_synonyms(selected_template, synonyms)
        # 对表格数据进行转换
        item = change_coverage_tabular_to_text(tabular_data[i], vocab_dic, fea_dim)
        text_coverage.append(ACSCoverage_tabular_to_text(result_template, item))
        # 对race数据进行转换
        sim_race = tabular_race[i]
        race_result = []
        for j in range(len(sim_race)):
            race_result.append(ACSCoverage_tabular_to_text(result_template, sim_race[j]))
        text_race.append(race_result)
        # 对gender数据进行转换
        sim_gender = tabular_gender[i]
        gender_result = []
        for j in range(len(sim_gender)):
            gender_result.append(ACSCoverage_tabular_to_text(result_template, sim_gender[j]))
        text_gender.append(gender_result)
        # 对aug数据进行转换
        sim_aug = tabular_aug[i]
        aug_result = []
        for j in range(len(sim_aug)):
            aug_result.append(ACSCoverage_tabular_to_text(result_template, sim_aug[j]))
        text_aug.append(aug_result)

    numpy.save(text_data_file, text_coverage)
    numpy.save(text_label_file, text_label)
    numpy.save(text_race_file, text_race)
    numpy.save(text_gender_file, text_gender)
    numpy.save(text_aug_file, text_aug)


def generate_ACSCoverage_adv_replace_synonyms():
    """
    生成Coverage数据集中所有文字的同义词替换，包括模板中的单词同义词，以及所有属性取值的同义词
    :return:
    """
    adv_synonyms_dic = {}
    # 定义模板
    templates = [
        "The following individual context corresponds to a survey respondent. The survey was conducted among US "
        "residents in 2018. Please answer the question what was the person's total income in the past year.based on "
        "the information provided. "
        "Individual context:"
        "The age is: {}."
        "The educational attainment is: {}."
        "The marital status is: {}."
        "The gender is: {}."
        "The disability recode is: {}."
        "The employment status of parents are: {}."
        "The citizenship status is: {}."
        "The mobility status is: {}."
        "The military service is: {}."
        "The ancestry recode is: {}."
        "The nativity is: {}."
        "The hearing difficulty is: {}."
        "The Vision difficulty is: {}."
        "The cognitive difficulty is: {}."
        "The total person’s income is: {}."
        "The employment status recode is: {}."
        "The state code is: {}."
        "The gave birth to child within the past 12 months is: {}."
        "The race is: {}."
    ]

    for t in templates:
        t = re.sub(r'\{.*?\}', '', t)
        t = t.replace('-', ' ')
        t = re.sub(r'[^a-zA-Z\s]', '', t)
        words = t.split()
        for w in words:
            w_synonyms = get_replace_synonyms(w)
            if len(w_synonyms) > 0:
                if w not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
                    adv_synonyms_dic[w] = w_synonyms

    # 所有属性的取值
    vocab_dic = ["less than 3 years old", "No schooling completed", "Nursery school/preschool", "Kindergarten",
                 "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6", "Grade 7", "Grade 8", "Grade 9",
                 "Grade 10", "Grade 11", "12th Grade - no diploma", "Regular high school diploma",
                 "GED or alternative credential", "Some college but less than 1 year",
                 "1 or more years of college credit but no degree", "Associate's degree", "Bachelor's degree",
                 "Master's degree", "Professional degree beyond a bachelor's degree", "Doctorate degree", "Married",
                 "Widowed", "Divorced", "Separated", "Never married or under 15 years old", "Reference person",
                 "Husband/wife", "Biological son or daughter", "Adopted son or daughter", "Stepson or stepdaughter",
                 "Brother or sister", "Father or mother", "Grandchild", "Parent-in-law",
                 "Son-in-law or daughter-in-law", "Other relative", "Roomer or boarder", "Housemate or roommate",
                 "Unmarried partner", "Foster child", "Other non relative",
                 "Institutionalized group quarters population", "Non institutionalized group quarters population",
                 "With a disability", "Without a disability",
                 "not own child of householder, and not child in subfamily",
                 "Living with two parents: both parents in labor force",
                 "Living with two parents: Father only in labor force",
                 "Living with two parents: Mother only in labor force",
                 "Living with two parents: Neither parent in labor force",
                 "Living with father: Father in the labor force", "Living with father: Father not in labor force",
                 "Living with mother: Mother in the labor force", "Living with mother: Mother not in labor force",
                 "Born in the U.S.", "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas",
                 "Born abroad of American parent(s)", "U.S. citizen by naturalization", "Not a citizen of the U.S.",
                 "less than 1 year old", "Yes, same house ( non movers )", "No, outside US and Puerto Rico",
                 "No, different house in US or Puerto Rico", "less than 17 years old", "Now on active duty",
                 "On active duty in the past, but not now",
                 "Only on active duty for training in Reserves/National Guard", "Never served in the military",
                 "Single", "Multiple", "Unclassified", "Not reported", "Suppressed for data year 2018 for select PUMAs",
                 "Native", "Foreign born", "Yes", "No", "Male", "Female", "White alone",
                 "Black or African American alone", "American Indian alone", "Alaska Native alone",
                 "American Indian and Alaska Native tribes specified, or American Indian or Alaska Native, not specified and no other races",
                 "Asian alone", "Native Hawaiian and Other Pacific Islander alone", "Some Other Race alone",
                 "Two or More Races", "Civilian employed, at work", "less than 16 years old",
                 "Civilian employed, with a job but not at work", "Unemployed",
                 "Armed forces, at work", "Armed forces, with a job but not at work", "Not in labor force"]

    for v in vocab_dic:
        for vv in v.strip().split(" "):

            w_synonyms = get_replace_synonyms(vv)
            if len(w_synonyms) > 0:
                if vv not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
                    adv_synonyms_dic[vv] = w_synonyms

    with open("../dataset/ACS/NLP/coverage/data/adv_synonyms_dic.pkl", 'wb') as f:
        pickle.dump(adv_synonyms_dic, f)
    adv_synonyms_text = []
    for k in adv_synonyms_dic:
        # if k not in adv_synonyms_text:
        #     adv_synonyms_text.append(k)
        for v in adv_synonyms_dic[k]:
            if v not in adv_synonyms_text:
                adv_synonyms_text.append(v)
    return [" ".join(adv_synonyms_text)]


# ACSTravel dataset
def change_travel_tabular_to_text(tabular_data, vocab_dic, fea_dim):
    """
    将adult数据集的编码结果回复为NLP
    :return:
    """
    text_data = []
    for i in range(16):
        if i in [0, 15]:
            text_data.append(str(tabular_data[i]))
        elif i in [1, 2, 3, 4, 5, 6, 7, 8, 11, 13]:
            for k in vocab_dic[i].keys():
                if vocab_dic[i][k][0] == tabular_data[i]:
                    text_data.append(k)
        elif i in [9, 10, 12, 14]:
            text_data.append(vocab_dic[i][tabular_data[i]][0])

    if len(text_data) == 16:
        return text_data
    else:
        return None


def ACSTravel_tabular_to_text(template, item):
    """
    将travel数据集的表格数据转化为文本数据
    :return:
    """
    travel_text = template.format(item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8],
                                  item[9], item[10], item[11], item[12], item[13], item[14], item[15], item[16])

    return travel_text


def generate_ACSTravel_text_data(data_file, race_file, gender_file, aug_file,
                                 text_data_file, text_label_file, text_race_file, text_gender_file, text_aug_file):
    """
    使用template模板将travel表格数据转换为文本text数据
    :return:
    """
    # 定义模板
    templates = [
        "The following individual context corresponds to a survey respondent. The survey was conducted among US "
        "residents in 2018. Please answer the question what was the person's total income in the past year.based on "
        "the information provided. "
        "Individual context:"
        "The age is: {}."
        "The educational attainment is: {}."
        "The marital status is: {}."
        "The gender is: {}."
        "The disability recode is: {}."
        "The employment status of parents are: {}."
        "The mobility status (lived here 1 year ago) is: {}."
        "The relationship is: {}."
        "The race is: {}."
        "The public use microdata area code (PUMA) is: {}."
        "The state code is: {}."
        "The citizenship status is: {}."
        "The occupation is: {}."
        "the means of transportation to work is: {}."
        "The place of work PUMA based on 2010 Census definitions is: {}."
        "The income-to-poverty ratio recode is: {}."
    ]

    # 定义同义词词典
    synonyms = {}
    tabular_data = pandas.read_csv(data_file).values
    tabular_race = numpy.load(race_file)
    tabular_gender = numpy.load(gender_file)
    tabular_aug = numpy.load(aug_file)

    text_travel = []
    text_race = []
    text_gender = []
    text_aug = []
    text_label = tabular_data[:, 16]

    fea_dim = numpy.load("../dataset/ACS/travel/data/fea_dim.npy").tolist()
    with open("../dataset/ACS/travel/data/vocab_dic.pkl", 'rb') as f:
        vocab_dic = pickle.load(f)

    for i in range(tabular_data.shape[0]):
        # 随机选择一个模板，替换同义词
        selected_template = random.choice(templates)
        result_template = replace_synonyms(selected_template, synonyms)
        # 对表格数据进行转换
        item = change_ACS_tabular_to_text(tabular_data[i], vocab_dic, fea_dim)
        text_travel.append(ACSTravel_tabular_to_text(result_template, item))
        # 对race数据进行转换
        sim_race = tabular_race[i]
        race_result = []
        for j in range(len(sim_race)):
            race_result.append(ACSTravel_tabular_to_text(result_template, sim_race[j]))
        text_race.append(race_result)
        # 对gender数据进行转换
        sim_gender = tabular_gender[i]
        gender_result = []
        for j in range(len(sim_gender)):
            gender_result.append(ACSTravel_tabular_to_text(result_template, sim_gender[j]))
        text_gender.append(gender_result)
        # 对aug数据进行转换
        sim_aug = tabular_aug[i]
        aug_result = []
        for j in range(len(sim_aug)):
            aug_result.append(ACSTravel_tabular_to_text(result_template, sim_aug[j]))
        text_aug.append(aug_result)

    numpy.save(text_data_file, text_travel)
    numpy.save(text_label_file, text_label)
    numpy.save(text_race_file, text_race)
    numpy.save(text_gender_file, text_gender)
    numpy.save(text_aug_file, text_aug)


def generate_ACSTravel_adv_replace_synonyms():
    """
    生成travel数据集中所有文字的同义词替换，包括模板中的单词同义词，以及所有属性取值的同义词
    :return:
    """
    adv_synonyms_dic = {}
    # 定义模板
    templates = [
        "The following individual context corresponds to a survey respondent. The survey was conducted among US "
        "residents in 2018. Please answer the question what was the person's total income in the past year.based on "
        "the information provided. "
        "Individual context:"
        "The age is: {}."
        "The educational attainment is: {}."
        "The marital status is: {}."
        "The gender is: {}."
        "The disability recode is: {}."
        "The employment status of parents are: {}."
        "The mobility status (lived here 1 year ago) is: {}."
        "The relationship is: {}."
        "The race is: {}."
        "The public use microdata area code (PUMA) is: {}."
        "The state code is: {}."
        "The citizenship status is: {}."
        "The occupation is: {}."
        "the means of transportation to work is: {}."
        "The place of work PUMA based on 2010 Census definitions is: {}."
        "The income-to-poverty ratio recode is: {}."
    ]

    for t in templates:
        t = re.sub(r'\{.*?\}', '', t)
        t = t.replace('-', ' ')
        t = re.sub(r'[^a-zA-Z\s]', '', t)
        words = t.split()
        for w in words:
            w_synonyms = get_replace_synonyms(w)
            if len(w_synonyms) > 0:
                if w not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
                    adv_synonyms_dic[w] = w_synonyms

    # 所有属性的取值
    vocab_dic = ["less than 3 years old", "No schooling completed", "Nursery school/preschool", "Kindergarten",
                 "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6", "Grade 7", "Grade 8", "Grade 9",
                 "Grade 10", "Grade 11", "12th Grade - no diploma", "Regular high school diploma",
                 "GED or alternative credential", "Some college but less than 1 year",
                 "1 or more years of college credit but no degree", "Associate's degree", "Bachelor's degree",
                 "Master's degree", "Professional degree beyond a bachelor's degree", "Doctorate degree", "Married",
                 "Widowed", "Divorced", "Separated", "Never married or under 15 years old", "Reference person",
                 "Husband/wife", "Biological son or daughter", "Adopted son or daughter", "Stepson or stepdaughter",
                 "Brother or sister", "Father or mother", "Grandchild", "Parent-in-law",
                 "Son-in-law or daughter-in-law", "Other relative", "Roomer or boarder", "Housemate or roommate",
                 "Unmarried partner", "Foster child", "Other non relative",
                 "Institutionalized group quarters population", "Non institutionalized group quarters population",
                 "With a disability", "Without a disability",
                 "not own child of householder, and not child in subfamily",
                 "Living with two parents: both parents in labor force",
                 "Living with two parents: Father only in labor force",
                 "Living with two parents: Mother only in labor force",
                 "Living with two parents: Neither parent in labor force",
                 "Living with father: Father in the labor force", "Living with father: Father not in labor force",
                 "Living with mother: Mother in the labor force", "Living with mother: Mother not in labor force",
                 "Born in the U.S.", "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas",
                 "Born abroad of American parent(s)", "U.S. citizen by naturalization", "Not a citizen of the U.S.",
                 "less than 1 year old", "Yes, same house ( non movers )", "No, outside US and Puerto Rico",
                 "No, different house in US or Puerto Rico", "less than 17 years old", "Now on active duty",
                 "On active duty in the past, but not now",
                 "Only on active duty for training in Reserves/National Guard", "Never served in the military",
                 "Single", "Multiple", "Unclassified", "Not reported", "Suppressed for data year 2018 for select PUMAs",
                 "Native", "Foreign born", "Yes", "No", "Male", "Female", "White alone",
                 "Black or African American alone", "American Indian alone", "Alaska Native alone",
                 "American Indian and Alaska Native tribes specified, or American Indian or Alaska Native, not specified and no other races",
                 "Asian alone", "Native Hawaiian and Other Pacific Islander alone", "Some Other Race alone",
                 "Two or More Races", "Civilian employed, at work", "less than 16 years old",
                 "Civilian employed, with a job but not at work", "Unemployed",
                 "Armed forces, at work", "Armed forces, with a job but not at work", "Not in labor force",
                 "Car, truck, or van", "Bus or trolley bus", "Streetcar or trolley car (carro publico in Puerto Rico)",
                 " Subway or elevated", "Railroad", "Ferryboat", " Taxicab", "Motorcycle", "Bicycle", "Walked;",
                 "Worked at home", "Other method"]
    for v in vocab_dic:
        for vv in v.strip().split(" "):

            w_synonyms = get_replace_synonyms(vv)
            if len(w_synonyms) > 0:
                if vv not in adv_synonyms_dic:  # 为每个单词构建同义词替换字典
                    adv_synonyms_dic[vv] = w_synonyms

    with open("../dataset/ACS/NLP/travel/data/adv_synonyms_dic.pkl", 'wb') as f:
        pickle.dump(adv_synonyms_dic, f)
    adv_synonyms_text = []
    for k in adv_synonyms_dic:
        # if k not in adv_synonyms_text:
        #     adv_synonyms_text.append(k)
        for v in adv_synonyms_dic[k]:
            if v not in adv_synonyms_text:
                adv_synonyms_text.append(v)
    return [" ".join(adv_synonyms_text)]


# 保存编码后的数据
def save_sequence_data(text_file, tokenizer, maxlen, save_file):
    """
    将文本数据使用Tokenizer转换为code sequences，之后填充数据到最大长度 maxlen， 之后进行保存
    :return:
    """
    text_data = numpy.load(text_file)
    sequences = tokenizer.texts_to_sequences(text_data)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    numpy.save(save_file, padded_sequences)


def save_augmentation_sequence_data(text_file, tokenizer, maxlen, save_file):
    """
    将文本数据使用Tokenizer转换为code sequences，之后填充数据到最大长度 maxlen， 之后进行保存
    :return:
    """
    result = []
    text_data = numpy.load(text_file)
    for i in range(text_data.shape[1]):
        sequences = tokenizer.texts_to_sequences(text_data[:, i])
        padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
        result.append(padded_sequences)
    numpy.save(save_file, result)
