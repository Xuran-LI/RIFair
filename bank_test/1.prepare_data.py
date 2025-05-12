from utils.utils_prepare_data import get_bank_voca_dic_and_fea_dim, split_bank_data, reCode_bank_data, \
    bank_data_augmentation_age, reCode_bank_data_similar

if __name__ == "__main__":
    get_bank_voca_dic_and_fea_dim()
    split_bank_data()

    train_file = "../dataset/bank/data/train_data.csv"
    test_file = "../dataset/bank/data/test_data.csv"

    reCode_bank_data(train_file, "train")
    reCode_bank_data(test_file, "test")

    bank_data_augmentation_age(test_file, "../dataset/bank/data/age_test_data.npy")
    reCode_bank_data_similar("../dataset/bank/data/age_test_data.npy", "age_test")

    bank_data_augmentation_age(train_file, "../dataset/bank/data/age_train_data.npy")
    reCode_bank_data_similar("../dataset/bank/data/age_train_data.npy", "age_train")
