import numpy
from tensorflow.python.keras import Input
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils.utils_Transform_AutoInt import TokenAndPositionEmbeddingBank, TransformerBlock
from utils.utils_draw import draw_adv_loss, draw_adv_pre
from utils.utils_evaluate import check_NLP_items_AF, get_NLP_items_perturbation_effect
from utils.utils_input_output import write_NLP, get_search_NLP_instances, get_search_labels, get_search_times

from utils.utils_result import restore_NLP_bank_nlp


def restore_adv_bank_data(items, labels, pres):
    """
    将对抗生成结果与相似样本恢复为NLP
    :return:
    """
    NLP = []
    for i in range(len(labels)):
        NLP.append(restore_NLP_bank_nlp(items[i], labels[i], numpy.argmax(pres[i])))
    return NLP


def get_adv_items_perturbation_effect_information(name2, name3):
    """
    检查对抗攻击过程中扰动后预测结果、损失函数、以及注意力权重的变化
    :return:
    """
    # 模型
    model_file = "../dataset/NLP/bank/model/{}.h5".format(name3)
    custom_layers = {'TokenAndPositionEmbeddingBank': TokenAndPositionEmbeddingBank,
                     'TransformerBlock': TransformerBlock}
    model = load_model(model_file, custom_objects=custom_layers)
    # embedding层模型
    embedding_layer = model.get_layer(index=1)
    embedding_inputs = Input(shape=(90,))
    embedding_output = embedding_layer(embedding_inputs)
    embedd_model = Model(inputs=embedding_inputs, outputs=embedding_output)
    # 注意力权重模型
    attention_layer = model.get_layer(index=2)
    attention_output = attention_layer.att(embedding_output, embedding_output, return_attention_scores=True)
    attention_model = Model(embedding_inputs, attention_output)

    # 获取当前保护属性下的对抗样本
    test_data = None
    test_sim = None
    test_label = None
    test_time = None
    for name1 in ["TB", "FF", "FB"]:
        # generated adversarial data, similar counterpart, label, search times
        generate_files = ["../dataset/NLP/bank/adv/{}_{}_{}_d.txt".format(name1, name2, name3),
                          "../dataset/NLP/bank/adv/{}_{}_{}_s.txt".format(name1, name2, name3),
                          "../dataset/NLP/bank/adv/{}_{}_{}_y.txt".format(name1, name2, name3),
                          "../dataset/NLP/bank/adv/{}_{}_{}_t.txt".format(name1, name2, name3)]

        adv_data = get_search_NLP_instances(generate_files[0])
        sim_data = get_search_NLP_instances(generate_files[1])
        adv_label = get_search_labels(generate_files[2]).reshape(-1, 1)
        search_times = get_search_times(generate_files[3]).reshape(-1, 1)
        if test_label is None:
            test_data = adv_data
            test_sim = sim_data
            test_label = adv_label
            test_time = search_times
        else:
            test_data = numpy.concatenate((test_data, adv_data), axis=0)
            test_sim = numpy.concatenate((test_sim, sim_data), axis=0)
            test_label = numpy.concatenate((test_label, adv_label), axis=0)
            test_time = numpy.concatenate((test_time, search_times), axis=0)

    # 是否搜索到TF，TB，FF，FB结果
    TF_tag = False
    # TB_tag = False
    # FF_tag = False
    # FB_tag = False
    TB_tag = True
    FF_tag = True
    FB_tag = True
    # 计算模型对生成样本及其相似样本的预测结果、每个样本的扰动次数
    pre1 = model.predict(test_data)
    pre2 = model.predict(test_sim)
    cate_label = to_categorical(test_label, num_classes=2)
    for i in range(1, len(test_time)):
        test_time[i] += test_time[i - 1]
    # 分析每轮扰动时样本的扰动信息
    for i in range(test_time.shape[0]):
        if i == 0:
            start_index = 0
            end_index = test_time[i][0]
        else:
            start_index = test_time[i - 1][0]
            end_index = test_time[i][0]
        # 获取本轮扰动结果：扰动后样本、相似样本、标签、扰动后预测结果、扰动后准确公平性检测结果
        search_data = test_data[start_index:end_index]
        search_sim = test_sim[start_index:end_index]
        search_data_pre = pre1[start_index:end_index]
        search_sim_pre = pre2[start_index:end_index]
        search_labels = test_label[start_index:end_index]
        search_cate = cate_label[start_index:end_index]
        search_AF = check_NLP_items_AF(search_data, search_sim, search_data_pre, search_sim_pre, search_labels)

        # 绘制扰动过程中loss，pre的变化情况，输出NLP结果
        if search_AF[0][0] and numpy.sum(search_AF[1]) > 0 and not TB_tag:
            # 原样本准确公平，搜索结果为true bias
            ones = numpy.ones_like(search_cate)
            loss1 = mean_squared_error(search_data_pre, search_cate)
            loss2 = mean_squared_error(search_sim_pre, ones - search_cate)
            NLP1 = restore_adv_bank_data(search_data, search_labels, search_data_pre)
            NLP2 = restore_adv_bank_data(search_sim, search_labels, search_sim_pre)
            effects1 = get_NLP_items_perturbation_effect(embedd_model, model, search_data)
            effects2 = get_NLP_items_perturbation_effect(embedd_model, model, search_sim)
            # 合并信息
            NLP = []
            for ii in range(len(NLP1)):
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result1 = NLP1[ii]
                NLP_result1.append("PII:{:.4f}".format(effects1[0][ii]))
                NLP_result1.append("input pert:{:.4f}".format(effects1[1][ii]))
                NLP_result1.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects1[3][ii]])))
                NLP_result1.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_data_pre[ii]])))
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result2 = NLP2[ii]
                NLP_result2.append("PII:{:.4f}".format(effects2[0][ii]))
                NLP_result2.append("input pert:{:.4f}".format(effects2[1][ii]))
                NLP_result2.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects2[3][ii]])))
                NLP_result2.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_sim_pre[ii]])))
                NLP.append([NLP_result1, NLP_result2])
            with open("../dataset/NLP/bank/result/{}_{}_{}_N.txt".format("TB", name2, name3), "w") as nlp_file:
                write_NLP(nlp_file, NLP)
            nlp_file.close()
            search_pre = [search_data_pre, search_sim_pre]
            draw_adv_pre(search_pre, search_cate, ["item", "counterpart"], "TB",
                         "../dataset/NLP/bank/result/{}_{}_{}_P.pdf".format("TB", name2, name3))
            loss = [loss1, loss2, loss1 + loss2]
            draw_adv_loss(loss, ["item", "counterpart", "optimization"], "iteration", "loss", "TB",
                          "../dataset/NLP/bank/result/{}_{}_{}_L.pdf".format("TB", name2, name3))
            TB_tag = True
        elif search_AF[0][0] and numpy.sum(search_AF[2]) > 0 and not FF_tag:
            # 原样本准确公平，搜索结果为false fair
            ones = numpy.ones_like(search_cate)
            loss1 = mean_squared_error(search_data_pre, ones - search_cate)
            loss2 = mean_squared_error(search_sim_pre, ones - search_cate)
            NLP1 = restore_adv_bank_data(search_data, search_labels, search_data_pre)
            NLP2 = restore_adv_bank_data(search_sim, search_labels, search_sim_pre)
            effects1 = get_NLP_items_perturbation_effect(embedd_model, model, search_data)
            effects2 = get_NLP_items_perturbation_effect(embedd_model, model, search_sim)
            # 合并信息
            NLP = []
            for ii in range(len(NLP1)):
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result1 = NLP1[ii]
                NLP_result1.append("PII:{:.4f}".format(effects1[0][ii]))
                NLP_result1.append("input pert:{:.4f}".format(effects1[1][ii]))
                NLP_result1.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects1[3][ii]])))
                NLP_result1.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_data_pre[ii]])))
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result2 = NLP2[ii]
                NLP_result2.append("PII:{:.4f}".format(effects2[0][ii]))
                NLP_result2.append("input pert:{:.4f}".format(effects2[1][ii]))
                NLP_result2.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects2[3][ii]])))
                NLP_result2.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_sim_pre[ii]])))
                NLP.append([NLP_result1, NLP_result2])
            with open("../dataset/NLP/bank/result/{}_{}_{}_N.txt".format("FF", name2, name3), "w") as nlp_file:
                write_NLP(nlp_file, NLP)
            nlp_file.close()
            search_pre = [search_data_pre, search_sim_pre]
            draw_adv_pre(search_pre, search_cate, ["item", "counterpart"], "FF",
                         "../dataset/NLP/bank/result/{}_{}_{}_P.pdf".format("FF", name2, name3))
            loss = [loss1, loss2, loss1 + loss2]
            draw_adv_loss(loss, ["item", "counterpart", "optimization"], "iteration", "loss", "FF",
                          "../dataset/NLP/bank/result/{}_{}_{}_L.pdf".format("FF", name2, name3))
            FF_tag = True
        elif search_AF[0][0] and numpy.sum(search_AF[3]) > 0 and not FB_tag:
            # 原样本准确公平，搜索结果为false bias
            ones = numpy.ones_like(search_cate)
            loss1 = mean_squared_error(search_data_pre, ones - search_cate)
            loss2 = mean_squared_error(search_sim_pre, search_cate)
            NLP1 = restore_adv_bank_data(search_data, search_labels, search_data_pre)
            NLP2 = restore_adv_bank_data(search_sim, search_labels, search_sim_pre)
            effects1 = get_NLP_items_perturbation_effect(embedd_model, model, search_data)
            effects2 = get_NLP_items_perturbation_effect(embedd_model, model, search_sim)
            # 合并信息
            NLP = []
            for ii in range(len(NLP1)):
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result1 = NLP1[ii]
                NLP_result1.append("PII:{:.4f}".format(effects1[0][ii]))
                NLP_result1.append("input pert:{:.4f}".format(effects1[1][ii]))
                NLP_result1.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects1[3][ii]])))
                NLP_result1.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_data_pre[ii]])))
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result2 = NLP2[ii]
                NLP_result2.append("PII:{:.4f}".format(effects2[0][ii]))
                NLP_result2.append("input pert:{:.4f}".format(effects2[1][ii]))
                NLP_result2.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects2[3][ii]])))
                NLP_result2.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_sim_pre[ii]])))
                NLP.append([NLP_result1, NLP_result2])
            with open("../dataset/NLP/bank/result/{}_{}_{}_N.txt".format("FB", name2, name3), "w") as nlp_file:
                write_NLP(nlp_file, NLP)
            nlp_file.close()
            search_pre = [search_data_pre, search_sim_pre]
            draw_adv_pre(search_pre, search_cate, ["item", "counterpart"], "FB",
                         "../dataset/NLP/bank/result/{}_{}_{}_P.pdf".format("FB", name2, name3))
            loss = [loss1, loss2, loss1 + loss2]
            draw_adv_loss(loss, ["item", "counterpart", "optimization"], "iteration", "loss", "FB",
                          "../dataset/NLP/bank/result/{}_{}_{}_L.pdf".format("FB", name2, name3))
            FB_tag = True
        elif numpy.sum(search_AF[0]) == len(search_AF[0]) and not TF_tag:
            # 原样本准确公平，搜索结果为true fair
            loss1 = mean_squared_error(search_data_pre, search_cate)
            loss2 = mean_squared_error(search_sim_pre, search_cate)
            NLP1 = restore_adv_bank_data(search_data, search_labels, search_data_pre)
            NLP2 = restore_adv_bank_data(search_sim, search_labels, search_sim_pre)
            effects1 = get_NLP_items_perturbation_effect(embedd_model, model, search_data)
            effects2 = get_NLP_items_perturbation_effect(embedd_model, model, search_sim)
            # 合并信息
            NLP = []
            for ii in range(len(NLP1)):
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result1 = NLP1[ii]
                NLP_result1.append("PII:{:.4f}".format(effects1[0][ii]))
                NLP_result1.append("input pert:{:.4f}".format(effects1[1][ii]))
                NLP_result1.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects1[3][ii]])))
                NLP_result1.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_data_pre[ii]])))
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result2 = NLP2[ii]
                NLP_result2.append("PII:{:.4f}".format(effects2[0][ii]))
                NLP_result2.append("input pert:{:.4f}".format(effects2[1][ii]))
                NLP_result2.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects2[3][ii]])))
                NLP_result2.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_sim_pre[ii]])))
                NLP.append([NLP_result1, NLP_result2])
            with open("../dataset/NLP/bank/result/{}_{}_{}_N.txt".format("TF", name2, name3), "w") as nlp_file:
                write_NLP(nlp_file, NLP)
            nlp_file.close()
            search_pre = [search_data_pre, search_sim_pre]
            draw_adv_pre(search_pre, search_cate, ["item", "counterpart"], "TF",
                         "../dataset/NLP/bank/result/{}_{}_{}_P.pdf".format("TF", name2, name3))
            loss = [loss1, loss2, loss1 + loss2]
            draw_adv_loss(loss, ["item", "counterpart", "optimization"], "iteration", "loss", "TF",
                          "../dataset/NLP/bank/result/{}_{}_{}_L.pdf".format("TF", name2, name3))
            TF_tag = True

        if TF_tag and TB_tag and FF_tag and FB_tag:
            return


if __name__ == "__main__":
    # 检查攻击成功率，以及扰动过程中预测结果，注意力权重的变化
    get_adv_items_perturbation_effect_information("age", "Transformer_BL0")
