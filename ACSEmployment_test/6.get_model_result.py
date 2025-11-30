import os
import random

import numpy
import pandas
import torch
import torch.nn.functional as F

from ACSEmployment_test.model import MLP, WideDeep, DeepFM, AutoInt, TabTransformer


def load_pytorch_model(model_class, model_path, init_args: dict, device="cpu"):
    # ===============================================================
    # 加载PyTorch模型
    # ===============================================================
    """
    model_class: 类 (MLP, WideDeep, DeepFM, AutoInt...)
    model_path: .pt / .pth
    init_args: 初始化参数字典，例如：
        {"input_dim":100}  或 {"num_fields":20,"num_categories":num_categories}
    """
    model = model_class(**init_args)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_model(model, x_i, x_v, encode_tag):
    """
    对样本进行模型预测，自动处理连续/离散组合或二者相乘的情况
    返回 (prob, pred_class)
    """
    model.eval()

    if encode_tag == "E":
        # E 模型：x_i 离散 + x_v 连续
        x_i = torch.tensor(x_i).long().unsqueeze(0)  # shape [1, num_fields]
        x_v = torch.tensor(x_v).float().unsqueeze(0)  # shape [1, num_fields]
        logits = model(x_i, x_v)
    else:
        # 非 E 模型：直接传入 (x_i * x_v)
        x = torch.tensor(x_v).float().unsqueeze(0)
        logits = model(x)

    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
    pred = int(prob.argmax())

    return prob, pred


def get_input(x_i, x_v, encode_tag):
    """
    获取模型输入
    返回 (prob, pred_class)
    """

    if encode_tag == "E":
        return numpy.concatenate([numpy.atleast_1d(x_i), numpy.atleast_1d(x_v)])
    else:
        return x_v


def analysis_False_or_Bias_test_data(model, attack_name, model_name, encode_tag, data_tag):
    """
    获取模型在测试集上的accuracy，fairness, accurate fairness
    :return:
    """
    N_O_test_index1 = numpy.load("../dataset/ACS/employment/data/N_O_test_i.npy", allow_pickle=True)
    N_O_test_value1 = numpy.load("../dataset/ACS/employment/data/N_O_test_V.npy", allow_pickle=True)
    N_O_test_index2 = numpy.load("../dataset/ACS/employment/data/N_O_aug_test_i.npy", allow_pickle=True)
    N_O_test_value2 = numpy.load("../dataset/ACS/employment/data/N_O_aug_test_V.npy", allow_pickle=True)
    N_O_test_label = numpy.load("../dataset/ACS/employment/data/N_O_test_y.npy", allow_pickle=True)

    N_E_test_index1 = numpy.load("../dataset/ACS/employment/data/N_E_test_i.npy", allow_pickle=True)
    N_E_test_value1 = numpy.load("../dataset/ACS/employment/data/N_E_test_V.npy", allow_pickle=True)
    N_E_test_index2 = numpy.load("../dataset/ACS/employment/data/N_E_aug_test_i.npy", allow_pickle=True)
    N_E_test_value2 = numpy.load("../dataset/ACS/employment/data/N_E_aug_test_V.npy", allow_pickle=True)
    N_E_test_label = numpy.load("../dataset/ACS/employment/data/N_E_test_y.npy", allow_pickle=True)

    # 判断扰动结果状态 FB, FF, FB
    FB, FF, TB = [False] * len(N_E_test_label), [False] * len(N_E_test_label), [False] * len(N_E_test_label)

    for i in range(N_O_test_index1.shape[0]):
        s_i = random.randint(0, N_O_test_index2.shape[0] - 1)
        if encode_tag == "E":
            prob_a1, pred_a1 = predict_model(model, N_E_test_index1[i], N_E_test_value1[i], encode_tag)
            prob_a2, pred_a2 = predict_model(model, N_E_test_index2[s_i][i], N_E_test_value2[s_i][i], encode_tag)
            true_label=N_E_test_label[i]
        else:
            prob_a1, pred_a1 = predict_model(model, None, N_O_test_index1[i]*N_O_test_value1[i], encode_tag)
            prob_a2, pred_a2 = predict_model(model, None, N_O_test_index2[s_i][i]*N_O_test_value2[s_i][i], encode_tag)
            true_label = N_O_test_label[i]
        # 判定攻击是否成功 false or bias
        if pred_a1 != true_label and pred_a2 == true_label and not FB[i]:
                FB[i] = True
        elif pred_a1 != true_label and pred_a2 != true_label and not FF[i]:
                FF[i] = True
        elif pred_a1 == true_label and pred_a2 != true_label and not TB[i]:
                TB[i] = True

    # ---- Final success rate ----
    TF = numpy.logical_not(numpy.logical_or(numpy.logical_or(FB, FF), TB))
    print(f"Test dataset: FB: {numpy.mean(FB):.2f}, FF: {numpy.mean(FF):.2f}, TB: {numpy.mean(TB):.2f}, "
          f"TF: {numpy.mean(TF):.2f} acc:{numpy.mean(TF)+numpy.mean(TB):.2f} fair:{numpy.mean(TF)+numpy.mean(FF):.2f}")
    return numpy.array(FB), numpy.array(FF), numpy.array(TB), numpy.array(TF)


def analyze_False_or_Bias_adv_results(model, attack_name, model_name, encode_tag, data_tag):
    """
    分析 false-biased-attack 的结果
    打印前 show_first_n 个样本的预测对比
    """
    results = numpy.load(f"../dataset/ACS/employment/adv/{attack_name}_{data_tag}_{model_name}.npy", allow_pickle=True).tolist()

    # 判断扰动结果状态 FB, FF, FB
    FB, FF, TB = [False] * len(results), [False] * len(results), [False] * len(results)

    for i, r in enumerate(results):
        history = r["history"]
        true_label = int(numpy.argmax(r["y"]))

        for j, item in enumerate(history):
            # ----------- 对抗样本，对抗相似样本 预测结果 -------------
            prob_a1, pred_a1 = predict_model(model, item["adv_xi1"], item["adv_xv1"], encode_tag)
            prob_a2, pred_a2 = predict_model(model, item["adv_xi2"], item["adv_xv2"], encode_tag)

            if j > 0 and pred_a1 != true_label and pred_a2 == true_label and not FB[i]:
                FB[i] = True
            elif j > 0 and pred_a1 != true_label and pred_a2 != true_label and not FF[i]:
                FF[i] = True
            elif j > 0 and pred_a1 == true_label and pred_a2 != true_label and not TB[i]:
                TB[i] = True

    # ---- Final success rate ----
    TF = numpy.logical_not(numpy.logical_or(numpy.logical_or(FB, FF), TB))
    return numpy.array(FB), numpy.array(FF), numpy.array(TB), TF


def get_False_or_Bias_adv_results(FB_result, FF_result, TB_result):
    """
    分析 false-biased-attack 的结果
    打印前 show_first_n 个样本的预测对比
    """
    FB_cond = numpy.logical_or(numpy.logical_or(FB_result[0], FF_result[0]), TB_result[0])
    TB_cond = numpy.logical_or(numpy.logical_or(FB_result[1], FF_result[1]), TB_result[1])
    FF_cond = numpy.logical_or(numpy.logical_or(FB_result[2], FF_result[2]), TB_result[2])
    TF_cond = numpy.logical_and(numpy.logical_and(FB_result[3], FF_result[3]), TB_result[3])
    FB_rate = numpy.sum(FB_cond) / len(FB_cond)
    FF_rate = numpy.sum(FF_cond) / len(FF_cond)
    TB_rate = numpy.sum(TB_cond) / len(TB_cond)
    TF_rate = numpy.sum(TF_cond) / len(TF_cond)
    False_rate = numpy.sum(numpy.logical_or(FB_cond, FF_cond)) / len(FB_cond)
    Bias_rate = numpy.sum(numpy.logical_or(FB_cond, TB_cond)) / len(TB_cond)


    # 获取一次、两次或三次攻击成功率
    A = numpy.stack([FB_cond, TB_cond, FF_cond], axis=1)
    # 每个样本的攻击次数（True 的数量）
    attack_count = numpy.sum(A, axis=1)  # 可能是 0,1,2,3
    # 恰好 1 次攻击
    attack1_rate = numpy.mean(attack_count == 1)
    # 恰好 2 次攻击
    attack2_rate = numpy.mean(attack_count == 2)
    # 恰好 3 次攻击
    attack3_rate = numpy.mean(attack_count == 3)
    print(f"1 attack: {attack1_rate:.4f}, 2 attacks: {attack2_rate:.4f}, 3 attacks: {attack3_rate:.4f}")

    attack3 = numpy.logical_and(numpy.logical_and(FB_cond, TB_cond), FF_cond)
    true_positions_attack3 = numpy.where(attack3)[0]
    if len(true_positions_attack3) > 0:
        random_pos_attack3 = numpy.random.choice(true_positions_attack3)
        print("attack3 随机 True 位置:", random_pos_attack3)
    else:
        print("attack3 中没有 True")


    return False_rate, Bias_rate, FB_rate, FF_rate, TB_rate, TF_rate, attack1_rate, attack2_rate, attack3_rate


def get_False_or_Bias_adv_condition(FB_result, FF_result, TB_result):
    """
    分析 false-biased-attack 的结果
    打印前 show_first_n 个样本的预测对比
    """
    FB_cond=numpy.logical_or(numpy.logical_or(FB_result[0], FF_result[0]), TB_result[0])
    TB_cond=numpy.logical_or(numpy.logical_or(FB_result[1], FF_result[1]), TB_result[1])
    FF_cond=numpy.logical_or(numpy.logical_or(FB_result[2], FF_result[2]), TB_result[2])
    TF_cond=numpy.logical_and(numpy.logical_and(FB_result[3], FF_result[3]), TB_result[3])

    return FB_cond, FF_cond, TB_cond, TF_cond


def analyze_False_results(model, attack_name, model_name, encode_tag, data_tag):
    """
    分析 false-attack 的结果
    """
    results = numpy.load(f"../dataset/ACS/employment/adv/{attack_name}_{data_tag}_{model_name}.npy", allow_pickle=True).tolist()

    # 判断扰动结果状态 FB, FF, FB
    false = [False] * len(results)

    for i, r in enumerate(results):
        history = r["history"]
        true_label = int(numpy.argmax(r["y"]))

        for j, item in enumerate(history):
            # ----------- 对抗样本，对抗相似样本 预测结果 -------------
            prob_a1, pred_a1 = predict_model(model, item["adv_xi1"], item["adv_xv1"], encode_tag)

            # 判定攻击是否成功 false
            if j > 0 and pred_a1 != true_label and not false[i]:
                false[i] = True

    # ---- Final success rate ----
    true = numpy.logical_not(false)
    false = numpy.array(false)
    return false, true


def get_False_result(F_result):
    """

    :return:
    """
    False_rate = numpy.sum(F_result[0]) / F_result[0].shape[0]

    return False_rate


def analyze_Bias_results(model, attack_name, model_name, encode_tag, data_tag):
    """
    分析 false-biased-attack 的结果
    打印前 show_first_n 个样本的预测对比
    """
    results = numpy.load(f"../dataset/ACS/employment/adv/{attack_name}_{data_tag}_{model_name}.npy",allow_pickle=True).tolist()

    # 判断扰动结果状态 FB, FF, FB
    bias = [False] * len(results)

    for i, r in enumerate(results):
        history = r["history"]

        for j, item in enumerate(history):
            # ----------- 对抗样本，对抗相似样本 预测结果 -------------
            prob_a1, pred_a1 = predict_model(model, item["adv_xi1"], item["adv_xv1"], encode_tag)
            prob_a2, pred_a2 = predict_model(model, item["adv_xi2"], item["adv_xv2"], encode_tag)

            # 判定攻击是否成功 bias
            if j > 0 and pred_a1 != pred_a2 and not bias[i]:
                bias[i] = True

    # ---- Final success rate ----
    fair = numpy.logical_not(bias)
    bias = numpy.array(bias)
    return bias, fair


def get_Bias_result(B_result):
    """

    :return:
    """
    Bias_rate = numpy.sum(B_result[0]) / B_result[0].shape[0]

    return Bias_rate


def make_attack_order_row(model_name, F, B):
    """
    将单个模型的三类攻击成功率指标汇总成一行 dict
    """
    return {
        "Model": model_name,
        "False": F,  # False Attack
        "Bias": B,  # Bias Attack
    }


def make_attack_times_row(model_name, FB):
    """
    将单个模型的三类攻击成功率指标汇总成一行 dict
    """
    return {
        "Model": model_name,
        "RIF_attack":1-FB[5],
        "False": FB[0],
        "Bias": FB[1],
        "FB": FB[2],
        "FF": FB[3],
        "TB": FB[4],
        "TF": FB[5],
        "attack 1": FB[6],
        "attack 2": FB[7],
        "attack 3": FB[8],
    }


def manipulate_accuracy_or_fairness(test_cond, adv_cond, manipulate_tag):
    test_FB, test_FF, test_TB, test_TF = test_cond
    adv_FB, adv_FF, adv_TB, adv_TF = adv_cond

    n = len(test_FB)

    # new 标签（最终结果）
    new_FB = test_FB.copy()
    new_FF = test_FF.copy()
    new_TB = test_TB.copy()
    new_TF = test_TF.copy()

    # 可替换位置（每种策略不同）
    if manipulate_tag == "increased_accuracy":
        # 用 TB/TF 替换 FB/FF
        replace_mask = numpy.logical_or(adv_TB, adv_TF)
        target_mask = numpy.logical_or(test_FB, test_FF)
        # 最终可替换位置 = adv 有样本 且 test 允许替换
        mask = numpy.logical_and(replace_mask, target_mask)
        # 清零所有旧标签
        new_FB[mask] = False
        new_FF[mask] = False
        new_TB[mask] = False
        new_TF[mask] = False
        # 样本可能面临多种攻击,但只能选择一种替换
        for m in range(len(mask)):
            if adv_TB[m] and mask[m] :
                new_TB[m] = True
                continue
            elif adv_TF[m] and mask[m]:
                new_TF[m] = True



    elif manipulate_tag == "increased_fairness":
        # 用 TF/FF 替换 TB/FB
        replace_mask = numpy.logical_or(adv_TF, adv_FF)
        target_mask = numpy.logical_or(test_TB, test_FB)
        # 最终可替换位置 = adv 有样本 且 test 允许替换
        mask = numpy.logical_and(replace_mask, target_mask)
        # 清零所有旧标签
        new_FB[mask] = False
        new_FF[mask] = False
        new_TB[mask] = False
        new_TF[mask] = False
        # 样本可能面临多种攻击,但只能选择一种替换
        for m in range(len(mask)):
            if adv_FF[m] and mask[m]:
                new_FF[m] = True
                continue
            elif adv_TF[m] and mask[m]:
                new_TF[m] = True

    elif manipulate_tag == "increased_both":
        # 用 TF 替换 TB/FB/FF
        replace_mask = adv_TF
        target_mask = numpy.logical_or.reduce([test_TB, test_FB, test_FF])
        # 最终可替换位置 = adv 有样本 且 test 允许替换
        mask = numpy.logical_and(replace_mask, target_mask)
        # 清零所有旧标签
        new_FB[mask] = False
        new_FF[mask] = False
        new_TB[mask] = False
        new_TF[mask] = False
        # 样本可能面临多种攻击,但只能选择一种替换
        for m in range(len(mask)):
            if adv_TF[m] and mask[m]:
                new_TF[m] = True

    elif manipulate_tag == "increased_accuracy_decrease_fairness":
        # 用 TB 替换 TF/FB/FF
        replace_mask = adv_TB
        target_mask = numpy.logical_or.reduce([test_TF, test_FB, test_FF])
        # 最终可替换位置 = adv 有样本 且 test 允许替换
        mask = numpy.logical_and(replace_mask, target_mask)
        # 清零所有旧标签
        new_FB[mask] = False
        new_FF[mask] = False
        new_TB[mask] = False
        new_TF[mask] = False
        # 样本可能面临多种攻击,但只能选择一种替换
        for m in range(len(mask)):
            if adv_TB[m] and mask[m]:
                new_TB[m] = True

    elif manipulate_tag == "increased_fairness_decrease_accuracy":
        # 用 FF 替换 TF/FB/TB
        replace_mask = adv_FF
        target_mask = numpy.logical_or.reduce([test_TF, test_FB, test_TB])
        # 最终可替换位置 = adv 有样本 且 test 允许替换
        mask = numpy.logical_and(replace_mask, target_mask)
        # 清零所有旧标签
        new_FB[mask] = False
        new_FF[mask] = False
        new_TB[mask] = False
        new_TF[mask] = False
        # 样本可能面临多种攻击,但只能选择一种替换
        for m in range(len(mask)):
            if adv_FF[m] and mask[m]:
                new_FF[m] = True

    else:
        raise ValueError(f"Unknown manipulate_tag: {manipulate_tag}")



    print(f"Manipulation: {manipulate_tag}")
    print(f"Test dataset: FB:{numpy.mean(new_FB):.2f}, FF:{numpy.mean(new_FF):.2f}, "
          f"TB:{numpy.mean(new_TB):.2f}, TF:{numpy.mean(new_TF):.2f} "
          f"acc:{numpy.mean(new_TF) + numpy.mean(new_TB):.2f} "
          f"fair:{numpy.mean(new_TF) + numpy.mean(new_FF):.2f}")

    return new_FB, new_FF, new_TB, new_TF


def make_manipulate_row(model_name, FB, FF, TB,TF):
    """
    获取模型在RIF测试集合下的false，bias，false or bias 信息，对比不同adversarial retraining提升效果
    """
    dic= {
        "Model": model_name,
        "Acc Rate": numpy.mean(TF)+numpy.mean(TB),  # False Attack
        "Fair Rate": numpy.mean(FF)+numpy.mean(TF),  # Bias Attack
        "FB Rate": numpy.mean(FB),
        "FF Rate": numpy.mean(FF),
        "TB Rate": numpy.mean(TB),
        "TF Rate": numpy.mean(TF),
    }
    return dic


def get_manipulate_result(test_result, fb_result, ff_result,tb_cond,model_name):
    adv_result = get_False_or_Bias_adv_condition(fb_result, ff_result, tb_cond)
    acc_up_result = manipulate_accuracy_or_fairness(test_result, adv_result, "increased_accuracy")
    fair_up_result = manipulate_accuracy_or_fairness(test_result, adv_result, "increased_fairness")
    both_up_result = manipulate_accuracy_or_fairness(test_result, adv_result, "increased_both")
    acc_up_fair_down = manipulate_accuracy_or_fairness(test_result, adv_result,"increased_accuracy_decrease_fairness")
    fair_up_acc_down = manipulate_accuracy_or_fairness(test_result, adv_result,"increased_fairness_decrease_accuracy")

    manipulate_result = []
    clean_manipulate_row = make_manipulate_row("clean", test_result[0], test_result[1], test_result[2], test_result[3])
    manipulate_result.append(clean_manipulate_row)
    wd_attack_row = make_manipulate_row("increased_accuracy", acc_up_result[0], acc_up_result[1], acc_up_result[2],acc_up_result[3])
    manipulate_result.append(wd_attack_row)
    dfm_attack_row = make_manipulate_row("increased_fairness", fair_up_result[0], fair_up_result[1], fair_up_result[2], fair_up_result[3])
    manipulate_result.append(dfm_attack_row)
    ai_attack_row = make_manipulate_row("increased_both", both_up_result[0], both_up_result[1], both_up_result[2], both_up_result[3])
    manipulate_result.append(ai_attack_row)
    tt_attack_row = make_manipulate_row("increased_accuracy_decrease_fairness", acc_up_fair_down[0],acc_up_fair_down[1], acc_up_fair_down[2], acc_up_fair_down[3])
    manipulate_result.append(tt_attack_row)
    tt_attack_row = make_manipulate_row("increased_fairness_decrease_accuracy", fair_up_acc_down[0],fair_up_acc_down[1],fair_up_acc_down[2], fair_up_acc_down[3])
    manipulate_result.append(tt_attack_row)
    # ---------------- 转成 DataFrame ----------------
    manipulate_df = pandas.DataFrame(manipulate_result)
    manipulate_excel_path = f"../dataset/compas/result/manipulate_application_{model_name}.xlsx"
    manipulate_df.to_excel(manipulate_excel_path, index=False)
    print(f"\n✔ Excel saved to: {manipulate_excel_path}")


def get_average_manipulate_result():
    """
    Calculate the average manipulate results across different models.

    model_names: list of str, e.g. ["MLP", "WideDeep", "DeepFM", "AutoInt", "TabTransformer"]
    base_path: path where each model's Excel file is saved
    """
    dfs = []
    base_path = "../dataset/ACS/employment/result/"
    for model_name in ["MLP", "WideDeep", "DeepFM", "AutoInt", "TabTransformer","transformer","bert"]:
        excel_path = f"{base_path}manipulate_application_{model_name}.xlsx"
        df = pandas.read_excel(excel_path)
        dfs.append(df)

    # Stack them vertically and compute the mean for each column
    combined_df = pandas.concat(dfs, axis=0, ignore_index=True)

    # Identify numeric columns only
    numeric_cols = combined_df.select_dtypes(include='number').columns

    # Compute mean for each numeric column, grouped by the attack type
    avg_df = combined_df.groupby(combined_df.index % len(dfs[0]))[numeric_cols].mean()

    avg_df.insert(0, 'Model', dfs[0]['Model'])

    avg_excel_path = f"{base_path}/manipulate_average.xlsx"
    avg_df.to_excel(avg_excel_path, index=False)
    print(f"✔ Average result saved to: {avg_excel_path}")

    return avg_df


if __name__ == "__main__":
    # num_fields = 16
    # # MLP
    # mlp_path = "../dataset/ACS/employment/model/MLP.pt"
    # mlp_model = load_pytorch_model(MLP, mlp_path, init_args={"input_dim": num_fields})
    # print("========== Attack Analysis ==========")
    # # 分析测试集结果
    # mlp_result = analysis_False_or_Bias_test_data(mlp_model, None, "MLP", encode_tag="O", data_tag="test")
    # # 分析F&B攻击结果
    # fb_result = analyze_False_or_Bias_adv_results(mlp_model, "FB", "MLP", encode_tag="O", data_tag="test")
    # ff_result = analyze_False_or_Bias_adv_results(mlp_model, "FF", "MLP", encode_tag="O", data_tag="test")
    # tb_cond = analyze_False_or_Bias_adv_results(mlp_model, "TB", "MLP", encode_tag="O", data_tag="test")
    # mlp_FB_result = get_False_or_Bias_adv_results(fb_result, ff_result, tb_cond)
    # # 分析robustness攻击结果
    # mlp_F_result = analyze_False_results(mlp_model, "False", "MLP", encode_tag="O", data_tag="test")
    # mlp_F_result = get_False_result(mlp_F_result)
    # # 分析fairness攻击结果
    # mlp_B_result = analyze_Bias_results(mlp_model, "Bias", "MLP", encode_tag="O", data_tag="test")
    # mlp_B_result = get_Bias_result(mlp_B_result)
    # get_manipulate_result(mlp_result, fb_result, ff_result, tb_cond, "MLP")
    # print("MLP")
    #
    # # WideDeep
    # wd_path = "../dataset/ACS/employment/model/wideDeep.pt"
    # wd_model = load_pytorch_model(WideDeep, wd_path, init_args={"num_fields": num_fields})
    # print("========== Attack Analysis ==========")
    # # 分析测试集结果
    # wd_result = analysis_False_or_Bias_test_data(wd_model, None, "WideDeep", encode_tag="E", data_tag="test")
    # # 分析F&B攻击结果
    # fb_result = analyze_False_or_Bias_adv_results(wd_model, "FB", "WideDeep", encode_tag="E", data_tag="test")
    # ff_result = analyze_False_or_Bias_adv_results(wd_model, "FF", "WideDeep", encode_tag="E", data_tag="test")
    # tb_cond = analyze_False_or_Bias_adv_results(wd_model, "TB", "WideDeep", encode_tag="E", data_tag="test")
    # wd_FB_result = get_False_or_Bias_adv_results(fb_result, ff_result, tb_cond)
    # # 分析robustness攻击结果
    # wd_F_result = analyze_False_results(wd_model, "False", "WideDeep", encode_tag="E", data_tag="test")
    # wd_F_result = get_False_result(wd_F_result)
    # # 分析fairness攻击结果
    # wd_B_result = analyze_Bias_results(wd_model, "Bias", "WideDeep", encode_tag="E", data_tag="test")
    # wd_B_result = get_Bias_result(wd_B_result)
    # get_manipulate_result(wd_result, fb_result, ff_result, tb_cond, "WideDeep")
    # print("WideDeep")
    #
    # # DeepFM
    # dfm_path = "../dataset/ACS/employment/model/DeepFM.pt"
    # dfm_model = load_pytorch_model(DeepFM, dfm_path, init_args={"num_fields": num_fields})
    # print("========== Attack Analysis ==========")
    # # 分析测试集结果
    # dfm_result = analysis_False_or_Bias_test_data(dfm_model, None, "DeepFM", encode_tag="E", data_tag="test")
    # # 分析F&B攻击结果
    # fb_result = analyze_False_or_Bias_adv_results(dfm_model, "FB", "DeepFM", encode_tag="E", data_tag="test")
    # ff_result = analyze_False_or_Bias_adv_results(dfm_model, "FF", "DeepFM", encode_tag="E", data_tag="test")
    # tb_cond = analyze_False_or_Bias_adv_results(dfm_model, "TB", "DeepFM", encode_tag="E", data_tag="test")
    # dfm_FB_result = get_False_or_Bias_adv_results(fb_result, ff_result, tb_cond)
    # # 分析robustness攻击结果
    # dfm_F_result = analyze_False_results(dfm_model, "False", "DeepFM", encode_tag="E", data_tag="test")
    # dfm_F_result = get_False_result(dfm_F_result)
    # # 分析fairness攻击结果
    # dfm_B_result = analyze_Bias_results(dfm_model, "Bias", "DeepFM", encode_tag="E", data_tag="test")
    # dfm_B_result = get_Bias_result(dfm_B_result)
    # get_manipulate_result(dfm_result, fb_result, ff_result, tb_cond, "DeepFM")
    # print("DeepFM")
    #
    # # AutoInt
    # ai_path = "../dataset/ACS/employment/model/AutoInt.pt"
    # ai_model = load_pytorch_model(AutoInt, ai_path, init_args={})
    # print("========== Attack Analysis ==========")
    # # 分析测试集结果
    # ai_result = analysis_False_or_Bias_test_data(ai_model, None, "AutoInt", encode_tag="E", data_tag="test")
    # # 分析F&B攻击结果
    # fb_result = analyze_False_or_Bias_adv_results(ai_model, "FB", "AutoInt", encode_tag="E", data_tag="test")
    # ff_result = analyze_False_or_Bias_adv_results(ai_model, "FF", "AutoInt", encode_tag="E", data_tag="test")
    # tb_cond = analyze_False_or_Bias_adv_results(ai_model, "TB", "AutoInt", encode_tag="E", data_tag="test")
    # ai_FB_result = get_False_or_Bias_adv_results(fb_result, ff_result, tb_cond)
    # # 分析robustness攻击结果
    # ai_F_result = analyze_False_results(ai_model, "False", "AutoInt", encode_tag="E", data_tag="test")
    # ai_F_result = get_False_result(ai_F_result)
    # # 分析fairness攻击结果
    # ai_B_result = analyze_Bias_results(ai_model, "Bias", "AutoInt", encode_tag="E", data_tag="test")
    # ai_B_result = get_Bias_result(ai_B_result)
    # get_manipulate_result(ai_result, fb_result, ff_result, tb_cond, "AutoInt")
    # print("AutoInt")
    #
    # # TabTransformer
    # tt_path = "../dataset/ACS/employment/model/TabTransformer.pt"
    # tt_model = load_pytorch_model(TabTransformer, tt_path, init_args={"num_fields": num_fields})
    # print("========== Attack Analysis ==========")
    # # 分析测试集结果
    # tt_result = analysis_False_or_Bias_test_data(tt_model, None, "TabTransformer", encode_tag="O", data_tag="test")
    # # 分析F&B攻击结果
    # fb_result = analyze_False_or_Bias_adv_results(tt_model, "FB", "TabTransformer", encode_tag="O", data_tag="test")
    # ff_result = analyze_False_or_Bias_adv_results(tt_model, "FF", "TabTransformer", encode_tag="O", data_tag="test")
    # tb_cond = analyze_False_or_Bias_adv_results(tt_model, "TB", "TabTransformer", encode_tag="O", data_tag="test")
    # tt_FB_result = get_False_or_Bias_adv_results(fb_result, ff_result, tb_cond)
    # # 分析robustness攻击结果
    # tt_F_result = analyze_False_results(tt_model, "False", "TabTransformer", encode_tag="O", data_tag="test")
    # tt_F_result = get_False_result(tt_F_result)
    # # 分析fairness攻击结果
    # tt_B_result = analyze_Bias_results(tt_model, "Bias", "TabTransformer", encode_tag="O", data_tag="test")
    # tt_B_result = get_Bias_result(tt_B_result)
    # get_manipulate_result(tt_result, fb_result, ff_result, tb_cond, "TabTransformer")
    # print("TabTransformer")
    #
    # order_results, times_results = [], []
    # # ---------------- MLP ----------------
    # mlp_order_row = make_attack_order_row("MLP", mlp_F_result, mlp_B_result)
    # mlp_times_row = make_attack_times_row("MLP", mlp_FB_result)
    # order_results.append(mlp_order_row)
    # times_results.append(mlp_times_row)
    # # ---------------- WideDeep ----------------
    # wd_order_row = make_attack_order_row("WideDeep", wd_F_result, wd_B_result)
    # wd_times_row = make_attack_times_row("WideDeep", wd_FB_result)
    # order_results.append(wd_order_row)
    # times_results.append(wd_times_row)
    # # ---------------- DeepFM ----------------
    # dfm_order_row = make_attack_order_row("DeepFM", dfm_F_result, dfm_B_result)
    # dfm_times_row = make_attack_times_row("DeepFM", dfm_FB_result)
    # order_results.append(dfm_order_row)
    # times_results.append(dfm_times_row)
    # # ---------------- AutoInt ----------------
    # ai_order_row = make_attack_order_row("AutoInt", ai_F_result, ai_B_result)
    # ai_times_row = make_attack_times_row("AutoInt",  ai_FB_result)
    # order_results.append(ai_order_row)
    # times_results.append(ai_times_row)
    # # ---------------- TabTransformer ----------------
    # tt_order_row = make_attack_order_row("TabTransformer", tt_F_result, tt_B_result)
    # tt_times_row = make_attack_times_row("TabTransformer", tt_FB_result)
    # order_results.append(tt_order_row)
    # times_results.append(tt_times_row)
    # # ---------------- 转成 DataFrame ----------------
    # order_df = pandas.DataFrame(order_results)
    # print("\n===== Final Attack Summary Table =====")
    # # 保存oder信息
    # order_excel_path = "../dataset/ACS/employment/result/attack_order_motivation.xlsx"
    # parent_dir = os.path.dirname(order_excel_path)
    # # 如果父目录不存在，就创建它
    # if not os.path.exists(parent_dir):
    #     os.makedirs(parent_dir, exist_ok=True)
    # # 对攻击成功率进行排序
    # # Attack Success Rate 排名（越高越靠前 → rank=1）
    # order_df["R_Rank"] = order_df["False"].rank(ascending=True, method='dense').astype(int)
    # # Discrimination Success Rate 排名
    # order_df["B_Rank"] = order_df["Bias"].rank(ascending=True, method='dense').astype(int)
    # # Robustness & Fairness 排名
    # print("\n===== Table with Ranking Added =====")
    # print(order_df.to_string(index=False))
    # sorted_by_bias = order_df.sort_values(by="R_Rank", ascending=True)
    # order_result = sorted_by_bias[["Model", "False", "R_Rank", "Bias", "B_Rank"]]
    # order_result.to_excel(order_excel_path, index=False)
    # print(f"\n✔ Excel saved to: {order_excel_path}")
    #
    # # ---------------- 转成 DataFrame ----------------
    # # 保存攻击次数信息
    # times_df = pandas.DataFrame(times_results)
    # # Attack Success Rate 排名（越高越靠前 → rank=1）
    # times_df["R_Rank"] = times_df["False"].rank(ascending=True, method='dense').astype(int)
    # times_df["B_Rank"] = times_df["Bias"].rank(ascending=True, method='dense').astype(int)
    # times_df["RIF_Rank"] = times_df["RIF_attack"].rank(ascending=True, method='dense').astype(int)
    #
    # # Robustness & Fairness 排名
    # print("\n===== Table with Ranking Added =====")
    # print(times_df.to_string(index=False))
    # times_by_bias = times_df.sort_values(by="RIF_Rank", ascending=True)
    # times_result = times_by_bias[["Model", "RIF_attack", "RIF_Rank", "False", "R_Rank","Bias","B_Rank","TB","FB","FF","TF","attack 1","attack 2","attack 3"]]
    # times_excel_path = "../dataset/ACS/employment/result/attack_times_motivation.xlsx"
    # times_result.to_excel(times_excel_path, index=False)
    # print(f"\n✔ Excel saved to: {times_excel_path}")
    get_average_manipulate_result()



# import os
#
# import numpy
# import pandas
# import torch
# import torch.nn.functional as F
# from scipy.stats import wasserstein_distance
#
# from ACSEmployment_test.model import MLP, WideDeep, DeepFM, AutoInt, TabTransformer
#
#
# def load_pytorch_model(model_class, model_path, init_args: dict, device="cpu"):
#     # ===============================================================
#     # 加载PyTorch模型
#     # ===============================================================
#     """
#     model_class: 类 (MLP, WideDeep, DeepFM, AutoInt...)
#     model_path: .pt / .pth
#     init_args: 初始化参数字典，例如：
#         {"input_dim":100}  或 {"num_fields":20,"num_categories":num_categories}
#     """
#     model = model_class(**init_args)
#     state = torch.load(model_path, map_location=device)
#     model.load_state_dict(state)
#     model.to(device)
#     model.eval()
#     return model
#
#
# def predict_model(model, x_i, x_v, encode_tag):
#     """
#     对样本进行模型预测，自动处理连续/离散组合或二者相乘的情况
#     返回 (prob, pred_class)
#     """
#     model.eval()
#
#     if encode_tag == "E":
#         # E 模型：x_i 离散 + x_v 连续
#         x_i = torch.tensor(x_i).long().unsqueeze(0)  # shape [1, num_fields]
#         x_v = torch.tensor(x_v).float().unsqueeze(0)  # shape [1, num_fields]
#         logits = model(x_i, x_v)
#     else:
#         # 非 E 模型：直接传入 (x_i * x_v)
#         x = torch.tensor(x_v).float().unsqueeze(0)
#         logits = model(x)
#
#     prob = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
#     pred = int(prob.argmax())
#
#     return prob, pred
#
#
# def get_input(x_i, x_v, encode_tag):
#     """
#     获取模型输入
#     返回 (prob, pred_class)
#     """
#
#     if encode_tag == "E":
#         return numpy.concatenate([numpy.atleast_1d(x_i), numpy.atleast_1d(x_v)])
#     else:
#         return x_v
#
#
# def analyze_False_or_Bias_results(model, attack_name, model_name, encode_tag, data_tag):
#     """
#     分析 false-biased-attack 的结果
#     打印前 show_first_n 个样本的预测对比
#     """
#     results = numpy.load(f"../dataset/ACS/employment/adv/{attack_name}_{data_tag}_{model_name}.npy",
#                          allow_pickle=True).tolist()
#
#     # 判断扰动结果状态 FB, FF, FB
#     FB, FF, TB = [False] * len(results), [False] * len(results), [False] * len(results)
#     PII1, PII2 = [], []
#
#     for i, r in enumerate(results):
#         history = r["history"]
#         true_label = int(numpy.argmax(r["y"]))
#
#         orig_1 = orig_2 = orig_pre1 = orig_pre2 = None
#         # pert_1 = pert_2 = pert_pre1 = pert_pre2 = None
#
#         prev_x1 = prev_x2 = None  # 上一次的输入
#         prev_pre1 = prev_pre2 = None  # 上一次的预测
#         for j, item in enumerate(history):
#             # ----------- 对抗样本，对抗相似样本 预测结果 -------------
#             prob_a1, pred_a1 = predict_model(model, item["adv_xi1"], item["adv_xv1"], encode_tag)
#             prob_a2, pred_a2 = predict_model(model, item["adv_xi2"], item["adv_xv2"], encode_tag)
#
#             if j == 0:
#                 # 第一轮：没有 perturbation，PII=0
#                 # PII1.append(0)
#                 # PII2.append(0)
#                 var = None
#             else:
#                 # 计算扰动影响（避免除零 & 避免 inf）
#                 denom1 = numpy.linalg.norm(prev_x1 - get_input(item["adv_xi1"], item["adv_xv1"], encode_tag))
#                 denom2 = numpy.linalg.norm(prev_x2 - get_input(item["adv_xi2"], item["adv_xv2"], encode_tag))
#
#                 pii1 = numpy.linalg.norm(prev_pre1 - prob_a1) / denom1 if denom1 > 1e-12 else 0
#                 pii2 = numpy.linalg.norm(prev_pre2 - prob_a2) / denom2 if denom2 > 1e-12 else 0
#
#                 PII1.append(pii1)
#                 PII2.append(pii2)
#
#             # 更新 previous state
#             prev_x1 = get_input(item["adv_xi1"], item["adv_xv1"], encode_tag)
#             prev_x2 = get_input(item["adv_xi2"], item["adv_xv2"], encode_tag)
#             prev_pre1, prev_pre2 = prob_a1, prob_a2
#
#             # # ---- 保存原始输入与预测结果 ----
#             # if j == 0:
#             #     orig_1 = get_input( item["adv_xi1"], item["adv_xv1"], encode_tag)
#             #     orig_2 = get_input( item["adv_xi2"], item["adv_xv2"], encode_tag)
#             #     orig_pre1=prob_a1
#             #     orig_pre2=prob_a2
#             #
#             # # ---- 保存扰动后输入与预测结果 ----
#             # if j >0:
#             #     pert_1 = get_input( item["adv_xi1"], item["adv_xv1"], encode_tag)
#             #     pert_2 = get_input( item["adv_xi2"], item["adv_xv2"], encode_tag)
#             #     pert_pre1=prob_a1
#             #     pert_pre2=prob_a2
#             #     # 计算扰动影响
#             #     if numpy.linalg.norm(orig_1 - pert_1) !=0:
#             #         PII1.append(numpy.linalg.norm(orig_pre1 - pert_pre1) / numpy.linalg.norm(orig_1 - pert_1))
#             #
#             #     if numpy.linalg.norm(orig_2 - pert_2)!=0:
#             #         PII2.append(numpy.linalg.norm(orig_pre2 - pert_pre2) / numpy.linalg.norm(orig_2 - pert_2))
#
#             # 判定攻击是否成功 false or bias
#             if j > 0 and pred_a1 != true_label and pred_a2 == true_label and not FB[i]:
#                 FB[i] = True
#             elif j > 0 and pred_a1 != true_label and pred_a2 != true_label and not FF[i]:
#                 FF[i] = True
#             elif j > 0 and pred_a1 == true_label and pred_a2 != true_label and not TB[i]:
#                 TB[i] = True
#
#     # ---- Final success rate ----
#     TF = numpy.logical_not(numpy.logical_or(numpy.logical_or(FB, FF), TB))
#     # print(f"{attack_name} attack => FB: {numpy.mean(FB):.2f}, FF: {numpy.mean(FF):.2f}, TB: {numpy.mean(TB):.2f}, "
#     #       f"TF: {numpy.mean(TF):.2f} PII1: {numpy.mean(PII1):.4f}, PII2: {numpy.mean(PII2):.4f}")
#     return FB, FF, TB, TF, PII1, PII2
#
#
# def get_False_or_Bias_results(FB_result, FF_result, TB_result):
#     """
#     分析 false-biased-attack 的结果
#     打印前 show_first_n 个样本的预测对比
#     """
#     FB_cond = numpy.logical_or(numpy.logical_or(FB_result[0], FF_result[0]), TB_result[0])
#     TB_cond = numpy.logical_or(numpy.logical_or(FB_result[1], FF_result[1]), TB_result[1])
#     FF_cond = numpy.logical_or(numpy.logical_or(FB_result[2], FF_result[2]), TB_result[2])
#     TF_cond = numpy.logical_and(numpy.logical_and(FB_result[3], FF_result[3]), TB_result[3])
#     FB_rate = numpy.sum(FB_cond) / len(FB_cond)
#     FF_rate = numpy.sum(FF_cond) / len(FF_cond)
#     TB_rate = numpy.sum(TB_cond) / len(TB_cond)
#     TF_rate = numpy.sum(TF_cond) / len(TF_cond)
#     False_rate = numpy.sum(numpy.logical_or(FB_cond, FF_cond)) / len(FB_cond)
#     Bias_rate = numpy.sum(numpy.logical_or(FB_cond, TB_cond)) / len(TB_cond)
#     PII1 = numpy.mean(FB_result[4] + FF_result[4] + TB_result[4])
#     PII2 = numpy.mean(FB_result[5] + FF_result[5] + TB_result[5])
#
#     # 获取一次、两次或三次攻击成功率
#     A = numpy.stack([FB_cond, TB_cond, FF_cond], axis=1)
#     # 每个样本的攻击次数（True 的数量）
#     attack_count = numpy.sum(A, axis=1)  # 可能是 0,1,2,3
#     # 恰好 1 次攻击
#     attack1_rate = numpy.mean(attack_count == 1)
#     # 恰好 2 次攻击
#     attack2_rate = numpy.mean(attack_count == 2)
#     # 恰好 3 次攻击
#     attack3_rate = numpy.mean(attack_count == 3)
#     print(f"1 attack: {attack1_rate:.4f}, 2 attacks: {attack2_rate:.4f}, 3 attacks: {attack3_rate:.4f}")
#
#     attack3 = numpy.logical_and(numpy.logical_and(FB_cond, TB_cond), FF_cond)
#     true_positions_attack3 = numpy.where(attack3)[0]
#     if len(true_positions_attack3) > 0:
#         random_pos_attack3 = numpy.random.choice(true_positions_attack3)
#         print("attack3 随机 True 位置:", random_pos_attack3)
#     else:
#         print("attack3 中没有 True")
#
#     coverage1 = max(FB_result[4] + FF_result[4] + TB_result[4]) - min(FB_result[4] + FF_result[4] + TB_result[4])
#     coverage2 = max(FB_result[5] + FF_result[5] + TB_result[5]) - min(FB_result[5] + FF_result[5] + TB_result[5])
#     d = wasserstein_distance(FB_result[4] + FF_result[4] + TB_result[4], FB_result[5] + FF_result[5] + TB_result[5])
#     print(f"PII1: {numpy.mean(coverage1):.4f}, PII2: {numpy.mean(coverage2):.4f} wasserstein_distance:{d:.4f}")
#
#     print(f"F&F attack => False rate: {False_rate:.4f} Bias rate: {Bias_rate:.4f} "
#           f"FB rate: {FB_rate:.4f} FF rate: {FF_rate:.4f} TB rate: {TB_rate:.4f} TF rate: {TF_rate:.4f} "
#           f"PII1: {numpy.mean(PII1):.4f}, PII2: {numpy.mean(PII2):.4f}")
#     return False_rate, Bias_rate, FB_rate, FF_rate, TB_rate, TF_rate, PII1, PII2, attack1_rate, attack2_rate, attack3_rate
#
#
# def analyze_False_results(model, attack_name, model_name, encode_tag, data_tag):
#     """
#     分析 false-attack 的结果
#     """
#     results = numpy.load(f"../dataset/ACS/employment/adv/{attack_name}_{data_tag}_{model_name}.npy",
#                          allow_pickle=True).tolist()
#
#     # 判断扰动结果状态 FB, FF, FB
#     false = [False] * len(results)
#
#     for i, r in enumerate(results):
#         history = r["history"]
#         true_label = int(numpy.argmax(r["y"]))
#
#         for j, item in enumerate(history):
#             # ----------- 对抗样本，对抗相似样本 预测结果 -------------
#             prob_a1, pred_a1 = predict_model(model, item["adv_xi1"], item["adv_xv1"], encode_tag)
#
#             # 判定攻击是否成功 false
#             if j > 0 and pred_a1 != true_label and not false[i]:
#                 false[i] = True
#
#     # ---- Final success rate ----
#     true = numpy.logical_not(false)
#     # print(f"false rate: {numpy.mean(false):.2f}")
#     return false, true
#
#
# def get_False_result(F_result, FB_result):
#     """
#
#     :return:
#     """
#     False_rate = numpy.sum(F_result[0]) / len(F_result[0])
#     Bias_rate = numpy.sum(numpy.logical_or(FB_result[0], FB_result[2])) / len(F_result[0])
#     False_rate1 = numpy.sum(numpy.logical_or(FB_result[0], FB_result[1])) / len(F_result[0])
#     FB_rate = numpy.sum(FB_result[0]) / len(F_result[0])
#     FF_rate = numpy.sum(FB_result[1]) / len(F_result[0])
#     TB_rate = numpy.sum(FB_result[2]) / len(F_result[0])
#     TF_rate = numpy.sum(FB_result[3]) / len(F_result[0])
#     PII1 = numpy.mean(FB_result[4])
#     PII2 = numpy.mean(FB_result[5])
#
#     coverage1 = max(FB_result[4]) - min(FB_result[4])
#     coverage2 = max(FB_result[5]) - min(FB_result[5])
#     d = wasserstein_distance(FB_result[4], FB_result[5])
#     print(f"PII1: {numpy.mean(coverage1):.4f}, PII2: {numpy.mean(coverage2):.4f} wasserstein_distance:{d:.4f}")
#
#     print(f"False attack => False rate: {False_rate:.4f} Bias rate: {Bias_rate:.4f} "
#           f"FB rate: {FB_rate:.4f} FF rate: {FF_rate:.4f} TB rate: {TB_rate:.4f} TF rate: {TF_rate:.4f} "
#           f"PII1: {numpy.mean(PII1):.4f}, PII2: {numpy.mean(PII2):.4f}")
#     return False_rate, Bias_rate, FB_rate, FF_rate, TB_rate, TF_rate, PII1, PII2
#
#
# def analyze_Bias_results(model, attack_name, model_name, encode_tag, data_tag):
#     """
#     分析 false-biased-attack 的结果
#     打印前 show_first_n 个样本的预测对比
#     """
#     results = numpy.load(f"../dataset/ACS/employment/adv/{attack_name}_{data_tag}_{model_name}.npy",
#                          allow_pickle=True).tolist()
#
#     # 判断扰动结果状态 FB, FF, FB
#     bias = [False] * len(results)
#
#     for i, r in enumerate(results):
#         history = r["history"]
#         true_label = int(numpy.argmax(r["y"]))
#
#         for j, item in enumerate(history):
#             # ----------- 对抗样本，对抗相似样本 预测结果 -------------
#             prob_a1, pred_a1 = predict_model(model, item["adv_xi1"], item["adv_xv1"], encode_tag)
#             prob_a2, pred_a2 = predict_model(model, item["adv_xi2"], item["adv_xv2"], encode_tag)
#
#             # 判定攻击是否成功 bias
#             if j > 0 and pred_a1 != pred_a2 and not bias[i]:
#                 bias[i] = True
#
#     # ---- Final success rate ----
#     fair = numpy.logical_not(bias)
#     # print(f"bias rate: {numpy.mean(bias):.2f}")
#     return bias, fair
#
#
# def get_Bias_result(B_result, FB_result):
#     """
#
#     :return:
#     """
#     Bias_rate = numpy.sum(B_result[0]) / len(B_result[0])
#     False_rate = numpy.sum(numpy.logical_or(FB_result[0], FB_result[1])) / len(B_result[0])
#     FB_rate = numpy.sum(FB_result[0]) / len(B_result[0])
#     FF_rate = numpy.sum(FB_result[1]) / len(B_result[0])
#     TB_rate = numpy.sum(FB_result[2]) / len(B_result[0])
#     TF_rate = numpy.sum(FB_result[3]) / len(B_result[0])
#     PII1 = numpy.mean(FB_result[4])
#     PII2 = numpy.mean(FB_result[5])
#
#     coverage1 = max(FB_result[4]) - min(FB_result[4])
#     coverage2 = max(FB_result[5]) - min(FB_result[5])
#     d = wasserstein_distance(FB_result[4], FB_result[5])
#     print(f"PII1: {numpy.mean(coverage1):.4f}, PII2: {numpy.mean(coverage2):.4f} wasserstein_distance:{d:.4f}")
#
#     # print(f"Bias rate: {Bias_rate1:.4f}")
#     print(f"Bias attack => False rate: {False_rate:.4f} Bias rate: {Bias_rate:.4f} "
#           f"FB rate: {FB_rate:.4f} FF rate: {FF_rate:.4f} TB rate: {TB_rate:.4f} TF rate: {TF_rate:.4f} "
#           f"PII1: {numpy.mean(PII1):.4f}, PII2: {numpy.mean(PII2):.4f}")
#     return False_rate, Bias_rate, FB_rate, FF_rate, TB_rate, TF_rate, PII1, PII2
#
#
# def make_attack_order_row(model_name, F, B, FB):
#     """
#     将单个模型的三类攻击成功率指标汇总成一行 dict
#     """
#     return {
#         "Model": model_name,
#         "False Rate": F[0],  # False Attack
#         "Bias Rate": B[1],  # Bias Attack
#         "False-or-Bias Rate": 1 - FB[5]  # Combined Attack
#     }
#
#
# def make_attack_times_row(model_name, F, B, FB):
#     """
#     将单个模型的三类攻击成功率指标汇总成一行 dict
#     """
#     return {
#         "Model": model_name,
#         # "FB Rate": FB[2],
#         # "FF Rate": FB[3],
#         # "TB Rate": FB[4],
#         "TF Rate": FB[5],
#         # "PII1": FB[6],
#         # "PII2": FB[7],
#         "attack 1": FB[8],
#         "attack 2": FB[9],
#         "attack 3": FB[10],
#     }
#
#
# if __name__ == "__main__":
#     num_fields = 16
#     # MLP
#     mlp_path = "../dataset/ACS/employment/model/MLP.pt"
#     mlp_model = load_pytorch_model(MLP, mlp_path, init_args={"input_dim": num_fields})
#     print("========== Attack Analysis ==========")
#     # 分析F&B攻击结果
#     fb_result = analyze_False_or_Bias_results(mlp_model, "FB", "MLP", encode_tag="O", data_tag="test")
#     ff_result = analyze_False_or_Bias_results(mlp_model, "FF", "MLP", encode_tag="O", data_tag="test")
#     tb_cond = analyze_False_or_Bias_results(mlp_model, "TB", "MLP", encode_tag="O", data_tag="test")
#     mlp_FB_result = get_False_or_Bias_results(fb_result, ff_result, tb_cond)
#     # 分析robustness攻击结果
#     mlp_F_result = analyze_False_results(mlp_model, "False", "MLP", encode_tag="O", data_tag="test")
#     F_FB_cond = analyze_False_or_Bias_results(mlp_model, "False", "MLP", encode_tag="O", data_tag="test")
#     mlp_F_result = get_False_result(mlp_F_result, F_FB_cond)
#     # 分析fairness攻击结果
#     mlp_B_result = analyze_Bias_results(mlp_model, "Bias", "MLP", encode_tag="O", data_tag="test")
#     B_FB_cond = analyze_False_or_Bias_results(mlp_model, "Bias", "MLP", encode_tag="O", data_tag="test")
#     mlp_B_result = get_Bias_result(mlp_B_result, B_FB_cond)
#     print("MLP")
#
#     # WideDeep
#     wd_path = "../dataset/ACS/employment/model/wideDeep.pt"
#     wd_model = load_pytorch_model(WideDeep, wd_path, init_args={"num_fields": num_fields})
#     print("========== Attack Analysis ==========")
#     # 分析F&B攻击结果
#     fb_result = analyze_False_or_Bias_results(wd_model, "FB", "WideDeep", encode_tag="E", data_tag="test")
#     ff_result = analyze_False_or_Bias_results(wd_model, "FF", "WideDeep", encode_tag="E", data_tag="test")
#     tb_cond = analyze_False_or_Bias_results(wd_model, "TB", "WideDeep", encode_tag="E", data_tag="test")
#     wd_FB_result = get_False_or_Bias_results(fb_result, ff_result, tb_cond)
#     # 分析robustness攻击结果
#     wd_F_result = analyze_False_results(wd_model, "False", "WideDeep", encode_tag="E", data_tag="test")
#     F_FB_cond = analyze_False_or_Bias_results(wd_model, "False", "WideDeep", encode_tag="E", data_tag="test")
#     wd_F_result = get_False_result(wd_F_result, F_FB_cond)
#     # 分析fairness攻击结果
#     wd_B_result = analyze_Bias_results(wd_model, "Bias", "WideDeep", encode_tag="E", data_tag="test")
#     B_FB_cond = analyze_False_or_Bias_results(wd_model, "Bias", "WideDeep", encode_tag="E", data_tag="test")
#     wd_B_result = get_Bias_result(wd_B_result, B_FB_cond)
#     print("WideDeep")
#
#     # DeepFM
#     dfm_path = "../dataset/ACS/employment/model/DeepFM.pt"
#     dfm_model = load_pytorch_model(DeepFM, dfm_path, init_args={"num_fields": num_fields})
#     print("========== Attack Analysis ==========")
#     # 分析F&B攻击结果
#     fb_result = analyze_False_or_Bias_results(dfm_model, "FB", "DeepFM", encode_tag="E", data_tag="test")
#     ff_result = analyze_False_or_Bias_results(dfm_model, "FF", "DeepFM", encode_tag="E", data_tag="test")
#     tb_cond = analyze_False_or_Bias_results(dfm_model, "TB", "DeepFM", encode_tag="E", data_tag="test")
#     dfm_FB_result = get_False_or_Bias_results(fb_result, ff_result, tb_cond)
#     # 分析robustness攻击结果
#     dfm_F_result = analyze_False_results(dfm_model, "False", "DeepFM", encode_tag="E", data_tag="test")
#     F_FB_cond = analyze_False_or_Bias_results(dfm_model, "False", "DeepFM", encode_tag="E", data_tag="test")
#     dfm_F_result = get_False_result(dfm_F_result, F_FB_cond)
#     # 分析fairness攻击结果
#     dfm_B_result = analyze_Bias_results(dfm_model, "Bias", "DeepFM", encode_tag="E", data_tag="test")
#     B_FB_cond = analyze_False_or_Bias_results(dfm_model, "Bias", "DeepFM", encode_tag="E", data_tag="test")
#     dfm_B_result = get_Bias_result(dfm_B_result, B_FB_cond)
#     print("DeepFM")
#
#     # AutoInt
#     ai_path = "../dataset/ACS/employment/model/AutoInt.pt"
#     ai_model = load_pytorch_model(AutoInt, ai_path, init_args={})
#     print("========== Attack Analysis ==========")
#     fb_result = analyze_False_or_Bias_results(ai_model, "FB", "AutoInt", encode_tag="E", data_tag="test")
#     ff_result = analyze_False_or_Bias_results(ai_model, "FF", "AutoInt", encode_tag="E", data_tag="test")
#     tb_cond = analyze_False_or_Bias_results(ai_model, "TB", "AutoInt", encode_tag="E", data_tag="test")
#     ai_FB_result = get_False_or_Bias_results(fb_result, ff_result, tb_cond)
#     # 分析robustness攻击结果
#     ai_F_result = analyze_False_results(ai_model, "False", "AutoInt", encode_tag="E", data_tag="test")
#     F_FB_cond = analyze_False_or_Bias_results(ai_model, "False", "AutoInt", encode_tag="E", data_tag="test")
#     ai_F_result = get_False_result(ai_F_result, F_FB_cond)
#     # 分析fairness攻击结果
#     ai_B_result = analyze_Bias_results(ai_model, "Bias", "AutoInt", encode_tag="E", data_tag="test")
#     B_FB_cond = analyze_False_or_Bias_results(ai_model, "Bias", "AutoInt", encode_tag="E", data_tag="test")
#     ai_B_result = get_Bias_result(ai_B_result, B_FB_cond)
#     print("AutoInt")
#
#     # TabTransformer
#     tt_path = "../dataset/ACS/employment/model/TabTransformer.pt"
#     tt_model = load_pytorch_model(TabTransformer, tt_path, init_args={"num_fields": num_fields})
#     print("========== Attack Analysis ==========")
#     fb_result = analyze_False_or_Bias_results(tt_model, "FB", "TabTransformer", encode_tag="O", data_tag="test")
#     ff_result = analyze_False_or_Bias_results(tt_model, "FF", "TabTransformer", encode_tag="O", data_tag="test")
#     tb_cond = analyze_False_or_Bias_results(tt_model, "TB", "TabTransformer", encode_tag="O", data_tag="test")
#     tt_FB_result = get_False_or_Bias_results(fb_result, ff_result, tb_cond)
#     # 分析robustness攻击结果
#     tt_F_result = analyze_False_results(tt_model, "False", "TabTransformer", encode_tag="O", data_tag="test")
#     F_FB_cond = analyze_False_or_Bias_results(tt_model, "False", "TabTransformer", encode_tag="O", data_tag="test")
#     tt_F_result = get_False_result(tt_F_result, F_FB_cond)
#     # 分析fairness攻击结果
#     tt_B_result = analyze_Bias_results(tt_model, "Bias", "TabTransformer", encode_tag="O", data_tag="test")
#     B_FB_cond = analyze_False_or_Bias_results(tt_model, "Bias", "TabTransformer", encode_tag="O", data_tag="test")
#     tt_B_result = get_Bias_result(tt_B_result, B_FB_cond)
#     print("TabTransformer")
#
#     order_results, times_results, attack_result = [], [], []
#     # ---------------- MLP ----------------
#     mlp_order_row = make_attack_order_row("MLP", mlp_F_result, mlp_B_result, mlp_FB_result)
#     mlp_times_row = make_attack_times_row("MLP", mlp_F_result, mlp_B_result, mlp_FB_result)
#     # mlp_attack_row= make_attack_result_row("MLP", mlp_F_result, mlp_B_result, mlp_FB_result)
#     order_results.append(mlp_order_row)
#     times_results.append(mlp_times_row)
#     # attack_result.append(mlp_attack_row)
#     # ---------------- WideDeep ----------------
#     wd_order_row = make_attack_order_row("WideDeep", wd_F_result, wd_B_result, wd_FB_result)
#     wd_times_row = make_attack_times_row("WideDeep", wd_F_result, wd_B_result, wd_FB_result)
#     # wd_attack_row=make_attack_result_row("WideDeep", wd_F_result, wd_B_result, wd_FB_result)
#     order_results.append(wd_order_row)
#     times_results.append(wd_times_row)
#     # attack_result.append(wd_attack_row)
#     # ---------------- DeepFM ----------------
#     dfm_order_row = make_attack_order_row("DeepFM", dfm_F_result, dfm_B_result, dfm_FB_result)
#     dfm_times_row = make_attack_times_row("DeepFM", dfm_F_result, dfm_B_result, dfm_FB_result)
#     # dfm_attack_row=make_attack_result_row("DeepFM", dfm_F_result, dfm_B_result, dfm_FB_result)
#     order_results.append(dfm_order_row)
#     times_results.append(dfm_times_row)
#     # attack_result.append(dfm_attack_row)
#     # ---------------- AutoInt ----------------
#     ai_order_row = make_attack_order_row("AutoInt", ai_F_result, ai_B_result, ai_FB_result)
#     ai_times_row = make_attack_times_row("AutoInt", ai_F_result, ai_B_result, ai_FB_result)
#     # ai_attack_row=make_attack_result_row("AutoInt", ai_F_result, ai_B_result, ai_FB_result)
#     order_results.append(ai_order_row)
#     times_results.append(ai_times_row)
#     # attack_result.append(ai_attack_row)
#     # ---------------- TabTransformer ----------------
#     tt_order_row = make_attack_order_row("TabTransformer", tt_F_result, tt_B_result, tt_FB_result)
#     tt_times_row = make_attack_times_row("TabTransformer", tt_F_result, tt_B_result, tt_FB_result)
#     # tt_attack_row=make_attack_result_row("TabTransformer", tt_F_result, tt_B_result, tt_FB_result)
#     order_results.append(tt_order_row)
#     times_results.append(tt_times_row)
#     # attack_result.append(tt_attack_row)
#     # ---------------- 转成 DataFrame ----------------
#     order_df = pandas.DataFrame(order_results)
#     times_df = pandas.DataFrame(times_results)
#     attack_df = pandas.DataFrame(attack_result)
#     print("\n===== Final Attack Summary Table =====")
#     # 保存oder信息
#     order_excel_path = "../dataset/ACS/employment/result/attack_order_motivation.xlsx"
#     parent_dir = os.path.dirname(order_excel_path)
#     # 如果父目录不存在，就创建它
#     if not os.path.exists(parent_dir):
#         os.makedirs(parent_dir, exist_ok=True)
#     # 对攻击成功率进行排序
#     print(order_df.to_string(index=False))
#     # Attack Success Rate 排名（越高越靠前 → rank=1）
#     order_df["Robustness Rank"] = order_df["False Rate"].rank(ascending=True, method='dense').astype(int)
#     # Discrimination Success Rate 排名
#     order_df["Fairness Rank"] = order_df["Bias Rate"].rank(ascending=True, method='dense').astype(int)
#     # Robustness & Fairness 排名
#     order_df["R&F Rank"] = order_df["False-or-Bias Rate"].rank(ascending=True, method='dense').astype(int)
#     print("\n===== Table with Ranking Added =====")
#     print(order_df.to_string(index=False))
#     sorted_by_bias = order_df.sort_values(by="False Rate", ascending=True)
#     order_result = sorted_by_bias[
#         ["Model", "False Rate", "Robustness Rank", "Bias Rate", "Fairness Rank", "False-or-Bias Rate", "R&F Rank"]]
#     order_result.to_excel(order_excel_path, index=False)
#     print(f"\n✔ Excel saved to: {order_excel_path}")
#     # 保存攻击次数信息
#     times_excel_path = "../dataset/ACS/employment/result/attack_times_motivation.xlsx"
#     times_df.to_excel(times_excel_path, index=False)
#     print(f"\n✔ Excel saved to: {times_excel_path}")
#     # 保存attack多种信息
#     # attack_excel_path = "../dataset/ACS/employment/result/attack_information_BL_tradeoff.xlsx"
#     # # 计算 attack_df 各列的均值
#     # # 只对数值列求均值（避免 Model 字段出错）
#     # mean_row = attack_df.mean(numeric_only=True)
#     # mean_row["Model"] = "BL Mean"  # 给均值行加一个标签
#     # mean_row = pandas.DataFrame([mean_row])
#     # # 将均值行添加到最后
#     # attack_df_with_mean = pandas.concat([attack_df, mean_row], axis=0)
#     # attack_df_with_mean.to_excel(attack_excel_path, index=False)
#     # print(f"\n✔ Excel saved to: {attack_excel_path}")
