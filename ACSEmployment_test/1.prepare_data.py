from utils.utils_prepare_data import get_ACSEmployment_voca_dic_and_fea_dim, split_ACSEmployment_data, \
    Numerical_Embedding_Code_ACSEmployment_data, ACSEmployment_data_augmentation_race, \
    ACSEmployment_data_augmentation_gender, \
    ACSEmployment_data_augmentation_multiple, Numerical_Embedding_Code_ACSEmployment_data_similar, \
    Numerical_OneHot_Code_ACSEmployment_data, Numerical_OneHot_Code_ACSEmployment_data_similar

if __name__ == "__main__":
    # get_ACSEmployment_voca_dic_and_fea_dim()
    # # 划分训练、验证、测试集
    train_file = "../dataset/ACS/employment/data/train_data.csv"
    vali_file = "../dataset/ACS/employment/data/vali_data.csv"
    test_file = "../dataset/ACS/employment/data/test_data.csv"
    split_ACSEmployment_data(train_file, vali_file, test_file)
    # 对validation数据就行数据增强
    ACSEmployment_data_augmentation_race(vali_file, "../dataset/ACS/employment/data/race_vali_data.npy")
    ACSEmployment_data_augmentation_gender(vali_file, "../dataset/ACS/employment/data/gender_vali_data.npy")
    ACSEmployment_data_augmentation_multiple(vali_file, "../dataset/ACS/employment/data/aug_vali_data.npy")
    # 对test数据就行数据增强
    ACSEmployment_data_augmentation_race(test_file, "../dataset/ACS/employment/data/race_test_data.npy")
    ACSEmployment_data_augmentation_gender(test_file, "../dataset/ACS/employment/data/gender_test_data.npy")
    ACSEmployment_data_augmentation_multiple(test_file, "../dataset/ACS/employment/data/aug_test_data.npy")

    # 对训练、验证、测试数据就行Numerical_Embedding编码
    Numerical_Embedding_Code_ACSEmployment_data(train_file, "train")
    Numerical_Embedding_Code_ACSEmployment_data(vali_file, "vali")
    Numerical_Embedding_Code_ACSEmployment_data(test_file, "test")
    # 对数据增强后的验证数据就行Numerical_Embedding编码
    Numerical_Embedding_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/race_vali_data.npy", "race_vali")
    Numerical_Embedding_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/gender_vali_data.npy", "gender_vali")
    Numerical_Embedding_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/aug_vali_data.npy", "aug_vali")
    # 对数据增强后的测试数据就行Numerical_Embedding编码
    Numerical_OneHot_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/race_test_data.npy", "race_test")
    Numerical_OneHot_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/gender_test_data.npy", "gender_test")
    Numerical_OneHot_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/aug_test_data.npy", "aug_test")

    # 对训练、验证、测试数据就行Numerical_OneHot编码
    Numerical_OneHot_Code_ACSEmployment_data(train_file, "train")
    Numerical_OneHot_Code_ACSEmployment_data(vali_file, "vali")
    Numerical_OneHot_Code_ACSEmployment_data(test_file, "test")
    # 对数据增强后的验证数据就行Numerical_OneHot编码
    Numerical_Embedding_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/race_vali_data.npy","race_vali")
    Numerical_Embedding_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/gender_vali_data.npy","gender_vali")
    Numerical_Embedding_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/aug_vali_data.npy", "aug_vali")
    # 对数据增强后的测试数据就行Numerical_OneHot编码
    Numerical_OneHot_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/race_test_data.npy","race_test")
    Numerical_OneHot_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/gender_test_data.npy","gender_test")
    Numerical_OneHot_Code_ACSEmployment_data_similar("../dataset/ACS/employment/data/aug_test_data.npy", "aug_test")

    # ACSEmployment_data_augmentation_race(train_file, "../dataset/ACS/employment/data/race_train_data.npy")
    # ACSEmployment_data_augmentation_gender(train_file, "../dataset/ACS/employment/data/gender_train_data.npy")
    # ACSEmployment_data_augmentation_multiple(train_file, "../dataset/ACS/employment/data/aug_train_data.npy")
    #
    # reCode_ACSEmployment_data_similar("../dataset/ACS/employment/data/race_train_data.npy", "race_train")
    # reCode_ACSEmployment_data_similar("../dataset/ACS/employment/data/gender_train_data.npy", "gender_train")
    # reCode_ACSEmployment_data_similar("../dataset/ACS/employment/data/aug_train_data.npy", "aug_train")
