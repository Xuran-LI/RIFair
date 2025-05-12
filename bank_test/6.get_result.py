import numpy
from tensorflow.python.keras import Input
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils.utils_Transform_AutoInt import BankEmbedding, AutoIntTransformerBlock
from utils.utils_draw import draw_heatmap, draw_adv_loss, draw_adv_pre
from utils.utils_evaluate import check_items_AF, get_items_perturbation_effect
from utils.utils_input_output import write_NLP, get_search_instances, get_search_labels, get_search_times
from utils.utils_result import restore_bank_nlp


def restore_adv_bank_data(items, labels, pres):
    """
    将对抗生成结果与相似样本恢复为NLP
    :return:
    """
    NLP = []
    for i in range(len(labels)):
        NLP.append(restore_bank_nlp(items[0][i], items[1][i], labels[i], numpy.argmax(pres[i])))
    return NLP


def get_adv_items_perturbation_effect_information(name2, name3):
    """
    检查对抗攻击过程中扰动后预测结果、损失函数、以及注意力权重的变化
    :return:
    """
    # 模型
    model_file = "../dataset/bank/model/{}.h5".format(name3)
    custom_layers = {'BankEmbedding': BankEmbedding, 'AutoIntTransformerBlock': AutoIntTransformerBlock}
    model = load_model(model_file, custom_objects=custom_layers)
    # embedding层模型
    embedding_layer = model.get_layer(index=2)
    index_inputs = Input(shape=(16,))
    value_inputs = Input(shape=(16,))
    embedding_output = embedding_layer(index_inputs, value_inputs)
    embedd_model = Model(inputs=[index_inputs, value_inputs], outputs=embedding_output)
    # 注意力权重模型
    attention_layer = model.get_layer(index=3)
    attention_output = attention_layer.att(embedding_output, embedding_output, return_attention_scores=True)
    attention_model = Model([index_inputs, value_inputs], attention_output)

    # 获取当前保护属性下的对抗样本
    index1 = None
    value1 = None
    index2 = None
    value2 = None
    labels = None
    times = None
    for name1 in ["TB", "FF", "FB"]:
        # generated adversarial data, similar counterpart, label, search times
        generate_files = ["../dataset/bank/adv/{}_{}_{}_i.txt".format(name1, name2, name3),
                          "../dataset/bank/adv/{}_{}_{}_s.txt".format(name1, name2, name3),
                          "../dataset/bank/adv/{}_{}_{}_y.txt".format(name1, name2, name3),
                          "../dataset/bank/adv/{}_{}_{}_t.txt".format(name1, name2, name3)]

        adv_label = get_search_labels(generate_files[2]).reshape(-1, 1)
        adv_index, adv_value = get_search_instances(generate_files[0])
        sim_index, sim_value = get_search_instances(generate_files[1])
        search_times = get_search_times(generate_files[3]).reshape(-1, 1)
        if labels is None:
            index1 = adv_index
            value1 = adv_value
            index2 = sim_index
            value2 = sim_value
            labels = adv_label
            times = search_times
        else:
            index1 = numpy.concatenate((index1, adv_index), axis=0)
            value1 = numpy.concatenate((value1, adv_value), axis=0)
            index2 = numpy.concatenate((index2, sim_index), axis=0)
            value2 = numpy.concatenate((value2, sim_value), axis=0)
            labels = numpy.concatenate((labels, adv_label), axis=0)
            times = numpy.concatenate((times, search_times), axis=0)

    # 是否搜索到TF，TB，FF，FB结果
    TF_tag = False
    TB_tag = False
    FF_tag = False
    FB_tag = False

    # 计算模型对生成样本及其相似样本的预测结果、每个样本的扰动次数
    pre1 = model.predict([index1, value1])
    pre2 = model.predict([index2, value2])
    cate_label = to_categorical(labels, num_classes=2)
    for i in range(1, len(times)):
        times[i] += times[i - 1]
    # 分析每轮扰动时样本的扰动信息
    for i in range(times.shape[0]):
        if i == 0:
            start_index = 0
            end_index = times[i][0]
        else:
            start_index = times[i - 1][0]
            end_index = times[i][0]
        # 获取本轮扰动结果：扰动后样本、相似样本、标签、扰动后预测结果、扰动后准确公平性检测结果
        search_pre1 = pre1[start_index:end_index]
        search_pre2 = pre2[start_index:end_index]
        search_items1 = [index1[start_index:end_index], value1[start_index:end_index]]
        search_items2 = [index2[start_index:end_index], value2[start_index:end_index]]
        search_labels = labels[start_index:end_index]
        search_cate = cate_label[start_index:end_index]
        search_AF = check_items_AF(search_items1, search_items2, search_pre1, search_pre2, search_labels)

        # 绘制扰动过程中loss，pre的变化情况，输出NLP结果
        if search_AF[0][0] and search_AF[1][-1] and not TB_tag:
            # 原样本准确公平，搜索结果为true bias
            ones = numpy.ones_like(search_cate)
            loss1 = mean_squared_error(search_pre1, search_cate)
            loss2 = mean_squared_error(search_pre2, ones - search_cate)
            NLP1 = restore_adv_bank_data(search_items1, search_labels, search_pre1)
            NLP2 = restore_adv_bank_data(search_items2, search_labels, search_pre2)
            effects1 = get_items_perturbation_effect(embedd_model, model, search_items1)
            effects2 = get_items_perturbation_effect(embedd_model, model, search_items2)
            # 合并信息
            NLP = []
            for ii in range(len(NLP1)):
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result1 = NLP1[ii]
                NLP_result1.append("PII:{:.4f}".format(effects1[0][ii]))
                NLP_result1.append("input pert:{:.4f}".format(effects1[1][ii]))
                NLP_result1.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects1[3][ii]])))
                NLP_result1.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_pre1[ii]])))
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result2 = NLP2[ii]
                NLP_result2.append("PII:{:.4f}".format(effects2[0][ii]))
                NLP_result2.append("input pert:{:.4f}".format(effects2[1][ii]))
                NLP_result2.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects2[3][ii]])))
                NLP_result2.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_pre2[ii]])))
                NLP.append([NLP_result1, NLP_result2])
            with open("../dataset/bank/result/{}_{}_{}_N.txt".format("TB", name2, name3), "w") as nlp_file:
                write_NLP(nlp_file, NLP)
            nlp_file.close()
            search_pre = [search_pre1, search_pre2]
            draw_adv_pre(search_pre, search_cate, ["item", "counterpart"], "TB",
                         "../dataset/bank/result/{}_{}_{}_P.pdf".format("TB", name2, name3))
            loss = [loss1, loss2, loss1 + loss2]
            draw_adv_loss(loss, ["item", "counterpart", "optimization"], "iteration", "loss", "FB",
                          "../dataset/bank/result/{}_{}_{}_L.pdf".format("TB", name2, name3))
            TB_tag = True
        elif search_AF[0][0] and search_AF[2][-1] and not FF_tag:
            # 原样本准确公平，搜索结果为false fair
            ones = numpy.ones_like(search_cate)
            loss1 = mean_squared_error(search_pre1, ones - search_cate)
            loss2 = mean_squared_error(search_pre2, ones - search_cate)
            NLP1 = restore_adv_bank_data(search_items1, search_labels, search_pre1)
            NLP2 = restore_adv_bank_data(search_items2, search_labels, search_pre2)
            effects1 = get_items_perturbation_effect(embedd_model, model, search_items1)
            effects2 = get_items_perturbation_effect(embedd_model, model, search_items2)
            # 合并信息
            NLP = []
            for ii in range(len(NLP1)):
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result1 = NLP1[ii]
                NLP_result1.append("PII:{:.4f}".format(effects1[0][ii]))
                NLP_result1.append("input pert:{:.4f}".format(effects1[1][ii]))
                NLP_result1.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects1[3][ii]])))
                NLP_result1.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_pre1[ii]])))
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result2 = NLP2[ii]
                NLP_result2.append("PII:{:.4f}".format(effects2[0][ii]))
                NLP_result2.append("input pert:{:.4f}".format(effects2[1][ii]))
                NLP_result2.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects2[3][ii]])))
                NLP_result2.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_pre2[ii]])))
                NLP.append([NLP_result1, NLP_result2])
            with open("../dataset/bank/result/{}_{}_{}_N.txt".format("FF", name2, name3), "w") as nlp_file:
                write_NLP(nlp_file, NLP)
            nlp_file.close()
            search_pre = [search_pre1, search_pre2]
            draw_adv_pre(search_pre, search_cate, ["item", "counterpart"], "FF",
                         "../dataset/bank/result/{}_{}_{}_P.pdf".format("FF", name2, name3))
            loss = [loss1, loss2, loss1 + loss2]
            draw_adv_loss(loss, ["item", "counterpart", "optimization"], "iteration", "loss", "FB",
                          "../dataset/bank/result/{}_{}_{}_L.pdf".format("FF", name2, name3))
            FF_tag = True
        elif search_AF[0][0] and search_AF[3][-1] and not FB_tag:
            # 原样本准确公平，搜索结果为false bias
            ones = numpy.ones_like(search_cate)
            loss1 = mean_squared_error(search_pre1, ones - search_cate)
            loss2 = mean_squared_error(search_pre2, search_cate)
            NLP1 = restore_adv_bank_data(search_items1, search_labels, search_pre1)
            NLP2 = restore_adv_bank_data(search_items2, search_labels, search_pre2)
            effects1 = get_items_perturbation_effect(embedd_model, model, search_items1)
            effects2 = get_items_perturbation_effect(embedd_model, model, search_items2)
            # 合并信息
            NLP = []
            for ii in range(len(NLP1)):
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result1 = NLP1[ii]
                NLP_result1.append("PII:{:.4f}".format(effects1[0][ii]))
                NLP_result1.append("input pert:{:.4f}".format(effects1[1][ii]))
                NLP_result1.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects1[3][ii]])))
                NLP_result1.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_pre1[ii]])))
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result2 = NLP2[ii]
                NLP_result2.append("PII:{:.4f}".format(effects2[0][ii]))
                NLP_result2.append("input pert:{:.4f}".format(effects2[1][ii]))
                NLP_result2.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects2[3][ii]])))
                NLP_result2.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_pre2[ii]])))
                NLP.append([NLP_result1, NLP_result2])
            with open("../dataset/bank/result/{}_{}_{}_N.txt".format("FB", name2, name3), "w") as nlp_file:
                write_NLP(nlp_file, NLP)
            nlp_file.close()
            search_pre = [search_pre1, search_pre2]
            draw_adv_pre(search_pre, search_cate, ["item", "counterpart"], "FB",
                         "../dataset/bank/result/{}_{}_{}_P.pdf".format("FB", name2, name3))
            loss = [loss1, loss2, loss1 + loss2]
            draw_adv_loss(loss, ["item", "counterpart", "optimization"], "iteration", "loss", "FB",
                          "../dataset/bank/result/{}_{}_{}_L.pdf".format("FB", name2, name3))
            FB_tag = True
        elif search_AF[0][0] and search_AF[0][-1] and len(search_AF[0]) > 10 and not TF_tag:
            # 原样本准确公平，搜索结果为true fair
            loss1 = mean_squared_error(search_pre1, search_cate)
            loss2 = mean_squared_error(search_pre2, search_cate)
            NLP1 = restore_adv_bank_data(search_items1, search_labels, search_pre1)
            NLP2 = restore_adv_bank_data(search_items2, search_labels, search_pre2)
            effects1 = get_items_perturbation_effect(embedd_model, model, search_items1)
            effects2 = get_items_perturbation_effect(embedd_model, model, search_items2)
            # 合并信息
            NLP = []
            for ii in range(len(NLP1)):
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result1 = NLP1[ii]
                NLP_result1.append("PII:{:.4f}".format(effects1[0][ii]))
                NLP_result1.append("input pert:{:.4f}".format(effects1[1][ii]))
                NLP_result1.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects1[3][ii]])))
                NLP_result1.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_pre1[ii]])))
                # 将数据还原为NLP形式，保存扰动、扰动影响、扰动方向、预测结果
                NLP_result2 = NLP2[ii]
                NLP_result2.append("PII:{:.4f}".format(effects2[0][ii]))
                NLP_result2.append("input pert:{:.4f}".format(effects2[1][ii]))
                NLP_result2.append("PID[{}]".format(", ".join([format(f, ".4f") for f in effects2[3][ii]])))
                NLP_result2.append("output[{}]".format(", ".join([format(f, ".4f") for f in search_pre2[ii]])))
                NLP.append([NLP_result1, NLP_result2])
            with open("../dataset/bank/result/{}_{}_{}_N.txt".format("TF", name2, name3), "w") as nlp_file:
                write_NLP(nlp_file, NLP)
            nlp_file.close()
            search_pre = [search_pre1, search_pre2]
            draw_adv_pre(search_pre, search_cate, ["item", "counterpart"], "TF",
                         "../dataset/bank/result/{}_{}_{}_P.pdf".format("TF", name2, name3))
            loss = [loss1, loss2, loss1 + loss2]
            draw_adv_loss(loss, ["item", "counterpart", "optimization"], "iteration", "loss", "FB",
                          "../dataset/bank/result/{}_{}_{}_L.pdf".format("TF", name2, name3))
            TF_tag = True

        if TF_tag and TB_tag and FF_tag and FB_tag:
            return


if __name__ == "__main__":
    # 获取对抗样本生成过程中扰动影响等信息
    get_adv_items_perturbation_effect_information("age", "AutoInt_BL0")
