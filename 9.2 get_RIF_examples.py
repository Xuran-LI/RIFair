import pickle

import numpy
import pandas


def get_tb_ff_fb_example(attack_results, attack_data, ori_pre, sim_pre):
    """

    :return:
    """
    tb_flags = [False] * len(attack_data)
    ff_flags = [False] * len(attack_data)
    fb_flags = [False] * len(attack_data)

    # Iterate through the results
    for i in range(len(attack_results)):
        row = attack_results.iloc[i]
        if 'original_example_id' in row:
            original_idx = int(row['original_example_id'])
        else:
            original_idx = i  # Fallback
        true_label = int(attack_data.iloc[original_idx]["label"])
        clean_ori = attack_data.iloc[original_idx]["text_ori"]
        clean_sim = attack_data.iloc[original_idx]["text_sim"]

        if true_label<0:
            true_label=0
        probs_ori = row["probs_ori"]
        probs_sim = row["probs_sim"]
        adv_ori_text=row["final_text_ori"]
        adv_sim_text=row["final_text_sim"]


        if probs_ori is None:
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
            start_ori_pre=numpy.argmax(ori_pre[original_idx])
            start_sim_pre=numpy.argmax(sim_pre[original_idx])
            ori_pred_label = numpy.argmax(probs_ori)
            sim_pred_label = numpy.argmax(probs_sim)
            is_robust = (ori_pred_label == true_label)
            is_fair = (ori_pred_label == sim_pred_label)
            if is_robust and not is_fair and start_ori_pre==true_label:
                tb_flags[original_idx] = True
                ff_flags[original_idx] = False
                fb_flags[original_idx] = False
                print(f"clean text: {clean_ori}\nclean_sim: {clean_sim}\nclean ori pre: {start_ori_pre}, clean sim pre: {start_sim_pre}  label:{true_label}\n"
                      f"adv text: {adv_ori_text}\nadv sim: {adv_sim_text}\nadv ori pre: {ori_pred_label}, adv sim pre:{sim_pred_label}")
                return
            elif not is_robust and is_fair and start_ori_pre==true_label:
                ff_flags[original_idx] = True
                tb_flags[original_idx] = False
                fb_flags[original_idx] = False
                print(
                    f"clean text: {clean_ori}\nclean_sim: {clean_sim}\nclean ori pre: {start_ori_pre}, clean sim pre: {start_sim_pre}  label:{true_label}\n"
                    f"adv text: {adv_ori_text}\nadv sim: {adv_sim_text}\nadv ori pre: {ori_pred_label}, adv sim pre:{sim_pred_label}")
                return
            elif not is_robust and not is_fair and start_ori_pre==true_label:
                fb_flags[original_idx] = True
                tb_flags[original_idx] = False
                ff_flags[original_idx] = False
                print(
                    f"clean text: {clean_ori}\nclean_sim: {clean_sim}\nclean ori pre: {start_ori_pre}, clean sim pre: {start_sim_pre}  label:{true_label}\n"
                    f"adv text: {adv_ori_text}\nadv sim: {adv_sim_text}\nadv ori pre: {ori_pred_label}, adv sim pre:{sim_pred_label}")
                return
            else:
                tb_flags[original_idx] = False
                ff_flags[original_idx] = False
                fb_flags[original_idx] = False


def get_RIF_attack_example_result(TB_path, FF_path, FB_path, attack_path, ori_pre, sim_pre):
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

    # get_tb_ff_fb_example(tb_results, df_ori, ori_pre, sim_pre)
    get_tb_ff_fb_example(ff_results, df_ori, ori_pre, sim_pre)
    get_tb_ff_fb_example(fb_results, df_ori, ori_pre, sim_pre)








if __name__ == "__main__":
    data_names = ["sentiment", "biasinbios", "jiasaw", "adultnew"]
    data_names = ["sentiment"]
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

            TB_file = f"{data_name}_data/RIFair/{clean_model_name}_TB_attack_results.pkl"
            FF_file = f"{data_name}_data/RIFair/{clean_model_name}_FF_attack_results.pkl"
            FB_file = f"{data_name}_data/RIFair/{clean_model_name}_FB_attack_results.pkl"

            get_RIF_attack_example_result(TB_file, FF_file, FB_file, attack_file, ori_pre, sim_pre)
