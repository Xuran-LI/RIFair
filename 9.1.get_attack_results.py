import pickle
import numpy
import pandas
from torch.cuda.nccl import init_rank


def get_robust_attack_result(result_path, attack_path, ori_pre, sim_pre):
    """
           Calculates whether the model remained robust (True) or was successfully attacked (False).

           Returns:
               list[bool]: True if model prediction matches ground truth (Robust). False if flipped.
               float: Robust Accuracy (Percentage).
           """
    print(f"Loading results from {result_path}...")
    with open(result_path, 'rb') as f:
        attack_result = pickle.load(f)
    df_results = pandas.DataFrame(attack_result)

    print(f"Loading original labels from {attack_path}...")
    with open(attack_path, 'rb') as f:
        ori_attack = pickle.load(f)
    df_ori = pandas.DataFrame(ori_attack)

    avg_attack_times = df_results["attack_step"].mean()

    robustness_flags = [False] * len(df_ori)

    # Iterate through the results
    for i in range(len(df_results)):
        row = df_results.iloc[i]

        # --- FIX 1: Safe Index Mapping ---
        # Use 'original_example_id' to find the correct ground truth label
        # If your results are strictly 1-to-1 aligned, you can use 'i',
        # but using the ID is much safer.
        if 'original_example_id' in row:
            original_idx = int(row['original_example_id'])
        else:
            original_idx = i  # Fallback

        # --- FIX 2: Correct DataFrame Access ---
        # old: ori_attack[i]["label"] -> Wrong
        # new: df_ori.iloc[idx]["label"] -> Correct
        true_label = int(df_ori.iloc[original_idx]["label"])

        # --- FIX 3: Check Prediction ---
        probs = row["probs_ori"]

        if probs is None:
            # If probs is None, the attack code usually implies "No valid candidate found"
            # This generally means the model output didn't change, so we assume it kept its original state.
            # However, strictly speaking, we count it as 'False' (Failed to evaluate) or 'True' (Robust)
            # usually, failure to find an adversarial example implies Robustness.
            pred_label = numpy.argmax(ori_pre[original_idx])
            is_robust = (pred_label == true_label)
            robustness_flags[original_idx] = is_robust
        else:
            pred_label = numpy.argmax(probs)

            # If prediction matches ground truth -> Model is Robust (True)
            # If prediction changed -> Model Failed / Attack Succeeded (False)
            is_robust = (pred_label == true_label)
            robustness_flags[original_idx] = is_robust

    # Calculate Summary Stats
    robust_acc = numpy.mean(robustness_flags) if robustness_flags else 0.0
    print(f"Robust Accuracy: {robust_acc:.2%} (Model stayed correct)")
    print(f"Attack Success Rate: {1 - robust_acc:.2%} (Model flipped)")

    return robust_acc, avg_attack_times

    # print(f"Loading data from {result_path}...")
    # with open(result_path, 'rb') as f:
    #     attack_result = pickle.load(f)
    # attack_result=pandas.DataFrame(attack_result)
    # with open(attack_path, 'rb') as f:
    #     ori_attack = pickle.load(f)
    # ori_attack=pandas.DataFrame(ori_attack)
    # result=[]
    # for i in range(len(attack_result)):
    #     if attack_result.iloc[i]["probs_ori"] is None:
    #         result.append(False)
    #     else:
    #         if numpy.argmax(attack_result.iloc[i]["probs_ori"])==ori_attack[i]["label"]:
    #             result.append(True)
    #         else:
    #             result.append(False)
    #
    # print()


def get_fair_attack_result(result_path, attack_path, ori_pre, sim_pre):
    """
           Calculates whether the model remained robust (True) or was successfully attacked (False).

           Returns:
               list[bool]: True if model prediction matches ground truth (Robust). False if flipped.
               float: Robust Accuracy (Percentage).
           """
    print(f"Loading results from {result_path}...")
    with open(result_path, 'rb') as f:
        attack_result = pickle.load(f)
    df_results = pandas.DataFrame(attack_result)

    print(f"Loading original labels from {attack_path}...")
    with open(attack_path, 'rb') as f:
        ori_attack = pickle.load(f)
    df_ori = pandas.DataFrame(ori_attack)
    avg_attack_times = df_results["attack_step"].mean()

    fairness_flags = [False] * len(df_ori)

    # Iterate through the results
    for i in range(len(df_results)):
        row = df_results.iloc[i]

        # --- FIX 1: Safe Index Mapping ---
        # Use 'original_example_id' to find the correct ground truth label
        # If your results are strictly 1-to-1 aligned, you can use 'i',
        # but using the ID is much safer.
        if 'original_example_id' in row:
            original_idx = int(row['original_example_id'])
        else:
            original_idx = i  # Fallback

        # --- FIX 2: Correct DataFrame Access ---
        # old: ori_attack[i]["label"] -> Wrong
        # new: df_ori.iloc[idx]["label"] -> Correct
        true_label = int(df_ori.iloc[original_idx]["label"])

        # --- FIX 3: Check Prediction ---
        probs_ori = row["probs_ori"]
        probs_sim = row["probs_sim"]

        if probs_ori is None:
            # If probs is None, the attack code usually implies "No valid candidate found"
            # This generally means the model output didn't change, so we assume it kept its original state.
            # However, strictly speaking, we count it as 'False' (Failed to evaluate) or 'True' (Robust)
            # usually, failure to find an adversarial example implies Robustness.
            ori_pred_label = numpy.argmax(ori_pre[original_idx])
            sim_pred_label = numpy.argmax(sim_pre[original_idx])

            # If prediction matches ground truth -> Model is Robust (True)
            # If prediction changed -> Model Failed / Attack Succeeded (False)
            is_fair = (ori_pred_label == sim_pred_label)
            fairness_flags[original_idx] = is_fair
        else:
            ori_pred_label = numpy.argmax(probs_ori)
            sim_pred_label = numpy.argmax(probs_sim)

            # If prediction matches ground truth -> Model is Robust (True)
            # If prediction changed -> Model Failed / Attack Succeeded (False)
            is_fair = (ori_pred_label == sim_pred_label)
            fairness_flags[original_idx] = is_fair

    # Calculate Summary Stats
    fail_acc = numpy.mean(fairness_flags) if fairness_flags else 0.0
    print(f"Fair Accuracy: {fail_acc:.2%} (Model stayed correct)")
    print(f"Attack Success Rate: {1 - fail_acc:.2%} (Model flipped)")

    return fail_acc, avg_attack_times

    # print(f"Loading data from {result_path}...")
    # with open(result_path, 'rb') as f:
    #     attack_result = pickle.load(f)
    # attack_result=pandas.DataFrame(attack_result)
    # with open(attack_path, 'rb') as f:
    #     ori_attack = pickle.load(f)
    # ori_attack=pandas.DataFrame(ori_attack)
    # result=[]
    # for i in range(len(attack_result)):
    #     if attack_result.iloc[i]["probs_ori"] is None:
    #         result.append(False)
    #     else:
    #         if numpy.argmax(attack_result.iloc[i]["probs_ori"])==ori_attack[i]["label"]:
    #             result.append(True)
    #         else:
    #             result.append(False)
    #
    # print()


def get_tb_ff_fb_cond(attack_results, attack_data, ori_pre, sim_pre):
    """

    :return:
    """
    tb_flags = [False] * len(attack_data)
    ff_flags = [False] * len(attack_data)
    fb_flags = [False] * len(attack_data)

    # Iterate through the results
    for i in range(len(attack_results)):
        row = attack_results.iloc[i]

        # --- FIX 1: Safe Index Mapping ---
        # Use 'original_example_id' to find the correct ground truth label
        # If your results are strictly 1-to-1 aligned, you can use 'i',
        # but using the ID is much safer.
        if 'original_example_id' in row:
            original_idx = int(row['original_example_id'])
        else:
            original_idx = i  # Fallback

        # --- FIX 2: Correct DataFrame Access ---
        # old: ori_attack[i]["label"] -> Wrong
        # new: df_ori.iloc[idx]["label"] -> Correct
        true_label = int(attack_data.iloc[original_idx]["label"])

        # --- FIX 3: Check Prediction ---
        probs_ori = row["probs_ori"]
        probs_sim = row["probs_sim"]

        if probs_ori is None:
            # If probs is None, the attack code usually implies "No valid candidate found"
            # This generally means the model output didn't change, so we assume it kept its original state.
            # However, strictly speaking, we count it as 'False' (Failed to evaluate) or 'True' (Robust)
            # usually, failure to find an adversarial example implies Robustness.
            ori_pred_label = numpy.argmax(ori_pre[original_idx])
            sim_pred_label = numpy.argmax(sim_pre[original_idx])
            is_robust = (ori_pred_label == true_label)
            is_fair = (ori_pred_label == sim_pred_label)
            if is_robust and not is_fair:
                tb_flags[original_idx] = True
                ff_flags[original_idx] = False
                fb_flags[original_idx] = False
            elif not is_robust and is_fair:
                ff_flags[original_idx] = True
                tb_flags[original_idx] = False
                fb_flags[original_idx] = False
            elif not is_robust and not is_fair:
                fb_flags[original_idx] = True
                tb_flags[original_idx] = False
                ff_flags[original_idx] = False
            else:
                tb_flags[original_idx] = False
                ff_flags[original_idx] = False
                fb_flags[original_idx] = False

        else:
            ori_pred_label = numpy.argmax(probs_ori)
            sim_pred_label = numpy.argmax(probs_sim)
            is_robust = (ori_pred_label == true_label)
            is_fair = (ori_pred_label == sim_pred_label)
            if is_robust and not is_fair:
                tb_flags[original_idx] = True
                ff_flags[original_idx] = False
                fb_flags[original_idx] = False
            elif not is_robust and is_fair:
                ff_flags[original_idx] = True
                tb_flags[original_idx] = False
                fb_flags[original_idx] = False
            elif not is_robust and not is_fair:
                fb_flags[original_idx] = True
                tb_flags[original_idx] = False
                ff_flags[original_idx] = False
            else:
                tb_flags[original_idx] = False
                ff_flags[original_idx] = False
                fb_flags[original_idx] = False
    return numpy.array(tb_flags), numpy.array(ff_flags), numpy.array(fb_flags)


def get_RIF_attack_result(TB_path, FF_path, FB_path, attack_path, ori_pre, sim_pre):
    """
           Calculates whether the model remained robust (True) or was successfully attacked (False).

           Returns:
               list[bool]: True if model prediction matches ground truth (Robust). False if flipped.
               float: Robust Accuracy (Percentage).
           """
    print(f"Loading results from {TB_path}...")
    with open(TB_path, 'rb') as f:
        TB_result = pickle.load(f)
    tb_results = pandas.DataFrame(TB_result)

    with open(FF_path, 'rb') as f:
        FF_result = pickle.load(f)
    ff_results = pandas.DataFrame(FF_result)

    with open(FB_path, 'rb') as f:
        FB_result = pickle.load(f)
    fb_results = pandas.DataFrame(FB_result)

    print(f"Loading original labels from {attack_path}...")
    with open(attack_path, 'rb') as f:
        ori_attack = pickle.load(f)
    df_ori = pandas.DataFrame(ori_attack)
    avg_tb_attack_times = tb_results["attack_step"].mean()
    avg_ff_attack_times = ff_results["attack_step"].mean()
    avg_fb_attack_times = fb_results["attack_step"].mean()
    avg_attack_times = (avg_tb_attack_times + avg_ff_attack_times + avg_fb_attack_times) / 3

    tb_flags1, fb_flags1, fb_flags1 = get_tb_ff_fb_cond(tb_results, df_ori, ori_pre, sim_pre)
    tb_flags2, fb_flags2, fb_flags2 = get_tb_ff_fb_cond(ff_results, df_ori, ori_pre, sim_pre)
    tb_flags3, fb_flags3, fb_flags3 = get_tb_ff_fb_cond(fb_results, df_ori, ori_pre, sim_pre)

    tb_flags = numpy.logical_or(numpy.logical_or(tb_flags1, tb_flags2), tb_flags3)
    fb_flags = numpy.logical_or(numpy.logical_or(fb_flags1, fb_flags2), fb_flags3)
    ff_flags = numpy.logical_or(numpy.logical_or(fb_flags1, fb_flags2), fb_flags3)

    RIF = numpy.logical_or(numpy.logical_or(tb_flags, fb_flags), ff_flags)

    # # Iterate through the results
    # for i in range(len(tb_results)):
    #     row = tb_results.iloc[i]
    #
    #     # --- FIX 1: Safe Index Mapping ---
    #     # Use 'original_example_id' to find the correct ground truth label
    #     # If your results are strictly 1-to-1 aligned, you can use 'i',
    #     # but using the ID is much safer.
    #     if 'original_example_id' in row:
    #         original_idx = int(row['original_example_id'])
    #     else:
    #         original_idx = i  # Fallback
    #
    #     # --- FIX 2: Correct DataFrame Access ---
    #     # old: ori_attack[i]["label"] -> Wrong
    #     # new: df_ori.iloc[idx]["label"] -> Correct
    #     true_label = int(df_ori.iloc[original_idx]["label"])
    #
    #     # --- FIX 3: Check Prediction ---
    #     probs_ori = row["probs_ori"]
    #     probs_sim = row["probs_sim"]
    #
    #     if probs_ori is None:
    #         # If probs is None, the attack code usually implies "No valid candidate found"
    #         # This generally means the model output didn't change, so we assume it kept its original state.
    #         # However, strictly speaking, we count it as 'False' (Failed to evaluate) or 'True' (Robust)
    #         # usually, failure to find an adversarial example implies Robustness.
    #         tb_flags.append(False)
    #         ff_flags.append(False)
    #         fb_flags.append(False)
    #     else:
    #         ori_pred_label = numpy.argmax(probs_ori)
    #         sim_pred_label = numpy.argmax(probs_sim)
    #         is_robust = (ori_pred_label == true_label)
    #         is_fair = (ori_pred_label == sim_pred_label)
    #         if is_robust and not is_fair:
    #             tb_flags.append(True)
    #             ff_flags.append(False)
    #             fb_flags.append(False)
    #         elif not is_robust and is_fair:
    #             ff_flags.append(True)
    #             tb_flags.append(False)
    #             fb_flags.append(False)
    #         elif not is_robust and not is_fair:
    #             fb_flags.append(True)
    #             tb_flags.append(False)
    #             ff_flags.append(False)
    #         else:
    #             tb_flags.append(False)
    #             ff_flags.append(False)
    #             fb_flags.append(False)

    # Calculate Summary Stats
    RIF_fail = numpy.mean(RIF)
    print(f"RIF Accuracy: {1 - RIF_fail:.2%} (Model stayed correct)")
    print(f"Attack Success Rate: {RIF_fail:.2%} (Model flipped)")

    return 1 - RIF_fail, avg_attack_times

    # print(f"Loading data from {result_path}...")
    # with open(result_path, 'rb') as f:
    #     attack_result = pickle.load(f)
    # attack_result=pandas.DataFrame(attack_result)
    # with open(attack_path, 'rb') as f:
    #     ori_attack = pickle.load(f)
    # ori_attack=pandas.DataFrame(ori_attack)
    # result=[]
    # for i in range(len(attack_result)):
    #     if attack_result.iloc[i]["probs_ori"] is None:
    #         result.append(False)
    #     else:
    #         if numpy.argmax(attack_result.iloc[i]["probs_ori"])==ori_attack[i]["label"]:
    #             result.append(True)
    #         else:
    #             result.append(False)
    #
    # print()


def get_tb_ff_fb_analysis(attack_results, attack_data, ori_pre, sim_pre):
    """

    :return:
    """
    acc_flags = [False] * len(attack_data)
    fair_flags = [False] * len(attack_data)
    RIF_flags = [False] * len(attack_data)

    # Iterate through the results
    for i in range(len(attack_results)):
        row = attack_results.iloc[i]
        if 'original_example_id' in row:
            original_idx = int(row['original_example_id'])
        else:
            original_idx = i  # Fallback

        true_label = int(attack_data.iloc[original_idx]["label"])
        if true_label <0:
            true_label=0

        # --- FIX 3: Check Prediction ---
        probs_ori = row["probs_ori"]
        probs_sim = row["probs_sim"]

        if probs_ori is None:
            ori_pred_label = numpy.argmax(ori_pre[original_idx])
            sim_pred_label = numpy.argmax(sim_pre[original_idx])
            is_robust = (ori_pred_label == true_label)
            is_fair = (ori_pred_label == sim_pred_label)

            if is_fair:
                fair_flags[original_idx] = True
            if is_robust:
                acc_flags[original_idx] = True
            if not is_robust or not is_fair:
                RIF_flags[original_idx] = True
        else:
            ori_pred_label = numpy.argmax(probs_ori)
            sim_pred_label = numpy.argmax(probs_sim)
            is_robust = (ori_pred_label == true_label)
            is_fair = (ori_pred_label == sim_pred_label)
            if is_fair:
                fair_flags[original_idx] = True
            if is_robust:
                acc_flags[original_idx] = True
            if not is_robust or not is_fair:
                RIF_flags[original_idx] = True

    return numpy.mean(acc_flags), numpy.mean(fair_flags), numpy.mean(RIF_flags)


def get_RIF_attack_analysis(TB_path, FF_path, FB_path, attack_path, ori_pre, sim_pre):
    """
           Calculates whether the model remained robust (True) or was successfully attacked (False).

           Returns:
               list[bool]: True if model prediction matches ground truth (Robust). False if flipped.
               float: Robust Accuracy (Percentage).
           """
    print(f"Loading results from {TB_path}...")
    with open(TB_path, 'rb') as f:
        TB_result = pickle.load(f)
    tb_results = pandas.DataFrame(TB_result)

    with open(FF_path, 'rb') as f:
        FF_result = pickle.load(f)
    ff_results = pandas.DataFrame(FF_result)

    with open(FB_path, 'rb') as f:
        FB_result = pickle.load(f)
    fb_results = pandas.DataFrame(FB_result)

    print(f"Loading original labels from {attack_path}...")
    with open(attack_path, 'rb') as f:
        ori_attack = pickle.load(f)
    df_ori = pandas.DataFrame(ori_attack)

    acc_flags1, fair_flags1, rif_flags1 = get_tb_ff_fb_analysis(tb_results, df_ori, ori_pre, sim_pre)
    acc_flags2, fair_flags2, rif_flags2 = get_tb_ff_fb_analysis(ff_results, df_ori, ori_pre, sim_pre)
    acc_flags3, fair_flags3, rif_flags3 = get_tb_ff_fb_analysis(fb_results, df_ori, ori_pre, sim_pre)

    result = {"acc_TB": acc_flags1, "fair_TB": fair_flags1, "RIF_TB": rif_flags1,
              "acc_FF": acc_flags2, "fair_FF": fair_flags2, "RIF_FF": rif_flags2,
              "acc_FB": acc_flags3, "fair_FB": fair_flags3, "RIF_FB": rif_flags3, }

    return result


if __name__ == "__main__":
    data_names = ["sentiment", "biasinbios", "jiasaw", "adultnew"]
    model_paths = {"bert-base-uncased": "bert-base-uncased", "roberta-base": "roberta-base",
                   "distilbert-base-uncased": "distilbert-base-uncased",
                   "microsoft/deberta-v3-base": "microsoft/deberta-v3-base"}
    R_result = []
    F_result = []
    RIF_result = []
    RIF_analysis = []

    for data_name in data_names:
        # 2. Iterate Models
        for model_key, model_path_raw in model_paths.items():
            clean_model_name = model_key.replace('/', '_')
            attack_file = f"{data_name}_data/attack_data.pkl"
            importance_file1 = f"{data_name}_data/{data_name}_{clean_model_name}_importance_ori.pkl"
            importance_file2 = f"{data_name}_data/{data_name}_{clean_model_name}_importance_sim.pkl"
            ori_pre = []
            sim_pre = []
            with open(importance_file1, 'rb') as f:
                batch_results = pickle.load(f)
            with open(importance_file2, 'rb') as f:
                sim_batch_results = pickle.load(f)
            for p in range(len(batch_results)):
                ori_pre.append(batch_results[p][0]['probs'])
                sim_pre.append(sim_batch_results[p][0]['probs'])

            R_result_file = f"{data_name}_data/textfooler/{clean_model_name}_attack_results.pkl"
            robust_acc, R_avg_attack = get_robust_attack_result(R_result_file, attack_file, ori_pre, sim_pre)
            R_result.append({"data_name": data_name, "methods": "textfooler", "model_name": clean_model_name,
                             "robust_acc": robust_acc, "R_attack_times": R_avg_attack})

            ADF_textfooler_file = f"{data_name}_data/ADF_TextFooler/{clean_model_name}_attack_results.pkl"
            fair_acc, F_avg_attack = get_fair_attack_result(ADF_textfooler_file, attack_file, ori_pre, sim_pre)
            F_result.append({"data_name": data_name, "methods": "adf-textfooler", "model_name": clean_model_name,
                             "fair_acc": fair_acc, "F_attack_times": F_avg_attack})

            MEFA_file = f"{data_name}_data/MEFA/{clean_model_name}_attack_results.pkl"
            fair_acc, F_avg_attack = get_fair_attack_result(MEFA_file, attack_file, ori_pre, sim_pre)
            F_result.append({"data_name": data_name, "methods": "MEFA", "model_name": clean_model_name,
                             "fair_acc": fair_acc, "F_attack_times": F_avg_attack})

            TB_file = f"{data_name}_data/RIFair/{clean_model_name}_TB_attack_results.pkl"
            FF_file = f"{data_name}_data/RIFair/{clean_model_name}_FF_attack_results.pkl"
            FB_file = f"{data_name}_data/RIFair/{clean_model_name}_FB_attack_results.pkl"

            RIF_acc, RIF_avg_attack = get_RIF_attack_result(TB_file, FF_file, FB_file, attack_file, ori_pre, sim_pre)
            RIF_result.append({"data_name": data_name, "methods": "RIF", "model_name": clean_model_name,
                               "RIF_acc": RIF_acc, "RIF_attack_times": RIF_avg_attack})

            RIF_attack_analysis = get_RIF_attack_analysis(TB_file, FF_file, FB_file, attack_file, ori_pre, sim_pre)
            RIF_attack_analysis["model_name"] = clean_model_name
            RIF_attack_analysis["data_name"] = data_name
            RIF_analysis.append(RIF_attack_analysis)

    robust_result = pandas.DataFrame(R_result)
    fair_result = pandas.DataFrame(F_result)
    rif_result = pandas.DataFrame(RIF_result)
    RIF_analysis = pandas.DataFrame(RIF_analysis)
    RIF_analysis_mean_values = RIF_analysis.groupby('model_name')[
        ["acc_TB", "fair_TB", "RIF_TB", "acc_FF", "fair_FF", "RIF_FF", "acc_FB", "fair_FB", "RIF_FB"]].mean()
    RIF_analysis_mean_values = RIF_analysis_mean_values.reset_index()
    RIF_analysis_mean_values.to_csv("result/RIF_analysis_mean_values.csv", index=False)

    rob_mean_values = robust_result.groupby('model_name')[['robust_acc', 'R_attack_times']].mean()
    fair_mean_values = fair_result.groupby('model_name')[['fair_acc', 'F_attack_times']].mean()
    rif_mean_values = rif_result.groupby('model_name')[['RIF_acc', 'RIF_attack_times']].mean()
    final_summary_df = pandas.concat([rob_mean_values, fair_mean_values, rif_mean_values], axis=1)
    final_summary_df['robust_rank'] = final_summary_df['robust_acc'].rank(ascending=False, method='min').astype(int)
    final_summary_df['fair_rank'] = final_summary_df['fair_acc'].rank(ascending=False, method='min').astype(int)
    final_summary_df['rif_rank'] = final_summary_df['RIF_acc'].rank(ascending=False, method='min').astype(int)

    final_summary_df = final_summary_df.reset_index()
    column_order = ["model_name", 'robust_acc', 'robust_rank', 'R_attack_times', 'fair_acc', 'fair_rank',
                    'F_attack_times', 'RIF_acc', 'rif_rank', 'RIF_attack_times']
    final_summary_df = final_summary_df[column_order]
    # 7. Final Sort (e.g., by RIF Rank) & Formatting
    final_summary_df = final_summary_df.sort_values(by='rif_rank')
    # Optional: Round floats for cleaner display
    numeric_cols = final_summary_df.select_dtypes(include=['float']).columns
    final_summary_df[numeric_cols] = final_summary_df[numeric_cols].round(3)
    final_summary_df.to_csv("result/final_model_attack_summary.csv", index=False)
    # Print without index for a clean table look
    print(final_summary_df.to_string(index=False))

    robust_result = pandas.DataFrame(R_result)
    # Group and calculate mean
    rob_grouped = robust_result.groupby(['model_name', 'methods'])[
        ['robust_acc', 'R_attack_times']].mean().reset_index()
    # Pivot to Wide Format
    pivot_rob_df = rob_grouped.pivot_table(index='model_name', columns='methods',
                                           values=['robust_acc', 'R_attack_times'])
    # Flatten columns: ('robust_acc', 'textfooler') -> 'robust_acc_textfooler'
    pivot_rob_df.columns = [f"{val}_{method}" for val, method in pivot_rob_df.columns]

    # --- 2. Fairness Processing ---
    fair_result = pandas.DataFrame(F_result)
    fair_grouped = fair_result.groupby(['model_name', 'methods'])[['fair_acc', 'F_attack_times']].mean().reset_index()
    pivot_fair_df = fair_grouped.pivot_table(index='model_name', columns='methods',
                                             values=['fair_acc', 'F_attack_times'])
    pivot_fair_df.columns = [f"{val}_{method}" for val, method in pivot_fair_df.columns]

    # --- 3. RIF Processing (Fixing the MultiIndex Issue) ---
    rif_result = pandas.DataFrame(RIF_result)
    rif_grouped = rif_result.groupby(['model_name', 'methods'])[['RIF_acc', 'RIF_attack_times']].mean().reset_index()
    # You MUST pivot this too, so it matches the index structure of the others
    pivot_rif_df = rif_grouped.pivot_table(index='model_name', columns='methods',
                                           values=['RIF_acc', 'RIF_attack_times'])
    pivot_rif_df.columns = [f"{val}_{method}" for val, method in pivot_rif_df.columns]

    # --- 4. Combine All ---
    # Now all three have 'model_name' as the Index and flattened columns
    final_all_attack = pandas.concat([pivot_rob_df, pivot_fair_df, pivot_rif_df], axis=1)

    # Reset index to make 'model_name' a normal column
    final_all_attack = final_all_attack.reset_index()

    # --- 5. Clean Up & Display ---
    # Optional: Sort or Round
    final_all_attack = final_all_attack.round(3)
    final_all_attack.to_csv("result/final_model_attacks.csv", index=False)
    print(final_all_attack.to_string(index=False))
