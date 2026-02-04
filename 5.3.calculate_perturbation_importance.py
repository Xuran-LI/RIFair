import pickle

import numpy
import pandas
import pickle


def calculate_R_importance(importance_file,sim_importance_f, save_file):
    """
    Calculates the importance score of features based on probability shifts.
    Formula:
    - If pred is same: Score = Prob(Orig_Class)_Orig - Prob(Orig_Class)_Ablated
    - If pred flips: Score = (Prob(Orig_Class)_Orig - Prob(Orig_Class)_Ablated) +  (Prob(New_Class)_Ablated - Prob(New_Class)_Orig)
    """

    print(f"Loading results from {importance_file}...")
    with open(importance_file, 'rb') as f:
        batch_results = pickle.load(f)
    with open(sim_importance_f, 'rb') as f:
        sim_batch_results = pickle.load(f)

    results = []

    for batch_data, sim_batch_data in zip(batch_results, sim_batch_results):
        # Convert list of dicts to DataFrame
        batch_df = pandas.DataFrame(batch_data)
        sim_batch = pandas.DataFrame(sim_batch_data)
        orig_probs = batch_df.iloc[0]['probs']
        Y = numpy.argmax(orig_probs)
        F_Y_X = orig_probs[Y]
        feature_scores = []
        # Loop through all variations (Original + Ablated features)
        for i in range(len(batch_df)):
            if i == 0:
                feature_scores.append(0.0)
                continue
            ablated_probs = batch_df.iloc[i]['probs']
            Y_bar = numpy.argmax(ablated_probs)
            F_Y_X_wi = ablated_probs[Y]
            term1 = F_Y_X - F_Y_X_wi
            if Y == Y_bar:
                score = term1
            else:
                F_Ybar_X_wi = ablated_probs[Y_bar]  # Prob of new class in ablated
                F_Ybar_X = orig_probs[Y_bar]  # Prob of new class in original
                term2 = F_Ybar_X_wi - F_Ybar_X
                score = term1 + term2

            feature_scores.append(score)

        batch_df["score"] = feature_scores
        batch_df["result_text_sim"]=sim_batch["result_text"]
        batch_sorted = batch_df.sort_values(by="score", ascending=False)
        results.append(batch_sorted)

    print(f"Saving sorted results to {save_file}...")
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)

    print("Done.")


def calculate_F_importance(ori_importance_f, sim_importance_f, save_file):
    """
    Calculates Fairness Importance Score.
    Score = (Initial Gap) - (Gap after ablating feature i)
    Positive score = This feature contributes to the unfairness gap.
    """

    print(f"Loading results...")
    with open(ori_importance_f, 'rb') as f:
        ori_batch_results = pickle.load(f)
    with open(sim_importance_f, 'rb') as f:
        sim_batch_results = pickle.load(f)
    results = []
    for ori_batch_data, sim_batch_data in zip(ori_batch_results, sim_batch_results):

        ori_batch = pandas.DataFrame(ori_batch_data)
        sim_batch = pandas.DataFrame(sim_batch_data)
        target_class = 1
        prob_v = ori_batch.iloc[0]['probs'][target_class]
        prob_v_prime = sim_batch.iloc[0]['probs'][target_class]
        start_gap = abs(prob_v - prob_v_prime)
        feature_scores = []
        for i in range(len(ori_batch)):
            if i == 0:
                feature_scores.append(start_gap)
                continue
            prob_v_ablated = ori_batch.iloc[i]['probs'][target_class]
            prob_v_prime_ablated = sim_batch.iloc[i]['probs'][target_class]
            ablated_gap = abs(prob_v_ablated - prob_v_prime_ablated)
            score = start_gap - ablated_gap
            feature_scores.append(score)
        ori_batch["score"] = feature_scores
        ori_batch["result_text_sim"]=sim_batch["result_text"]
        batch_sorted = ori_batch.sort_values(by="score", ascending=False)
        results.append(batch_sorted)

    print(f"Saving fairness importance results to {save_file}")
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)


def calculate_RIFair_importance(label_file, ori_importance_f, sim_importance_f, save_file, ori_alpha, sim_alpha, beta):
    """
    Calculates the contribution of each token to the Unified RIFair Objective (J).
    Importance Score = (Initial J) - (J after ablating token i)
    High Score = This token contributes significantly to the violation (Robustness or Fairness).
    """
    print(f"Loading results...")

    # Load data
    with open(label_file, 'rb') as f:
        label_data = pickle.load(f)
    with open(ori_importance_f, 'rb') as f:
        ori_batch_results = pickle.load(f)
    with open(sim_importance_f, 'rb') as f:
        sim_batch_results = pickle.load(f)

    # Helper function defined cleanly outside the loop or inside (preference)
    def calculate_J(ori_probs, sim_probs, label_idx, alpha_v, alpha_v_prime, beta_val):
        """
        Calculates J = α_v * L(v) + α_v' * L(v') + β * D(v, v')
        """
        epsilon = 1e-8

        # 1. Get probability of the Ground Truth Class
        p_true_ori = ori_probs[label_idx]
        p_true_sim = sim_probs[label_idx]

        # 2. Robustness Terms (Cross Entropy: -log(p_true))
        loss_ori = -numpy.log(p_true_ori + epsilon)
        loss_sim = -numpy.log(p_true_sim + epsilon)

        # 3. Fairness Term (Gap)
        # Note: Depending on your D(), this might be absolute difference or KL divergence.
        # Here we use absolute difference of the target class probability.
        fair_gap = abs(p_true_ori - p_true_sim)

        # 4. Total J
        J = (alpha_v * loss_ori) + (alpha_v_prime * loss_sim) + (beta_val * fair_gap)
        return J

    results = []
    if isinstance(label_data, list) and isinstance(label_data[0], dict):
        labels_list = [item['label'] for item in label_data]
    else:
        # Fallback if it's already a flat list or DataFrame
        labels_list = pandas.DataFrame(label_data)['label'].tolist()

    # Iterate through batches
    for ori_batch_data, sim_batch_data, batch_label in zip(ori_batch_results, sim_batch_results, labels_list):
        ori_batch = pandas.DataFrame(ori_batch_data)
        sim_batch = pandas.DataFrame(sim_batch_data)

        # 1. Calculate Baseline J (Before Ablation) - Index 0
        base_ori_probs = ori_batch.iloc[0]['probs']
        base_sim_probs = sim_batch.iloc[0]['probs']

        start_J = calculate_J(base_ori_probs, base_sim_probs, batch_label, ori_alpha, sim_alpha, beta)

        feature_scores = []

        # 2. Calculate J for each ablated version
        for i in range(len(ori_batch)):
            if i == 0:
                # The original text has 0 importance relative to itself
                feature_scores.append(0.0)
                continue

            # Get probabilities for ablated token i
            # FIX: Pass the WHOLE array, not just the scalar `[batch_label]`
            curr_ori_probs = ori_batch.iloc[i]['probs']
            curr_sim_probs = sim_batch.iloc[i]['probs']

            J_ablated = calculate_J(curr_ori_probs, curr_sim_probs, batch_label, ori_alpha, sim_alpha, beta)

            # Score = How much did J drop when we removed this token?
            # Positive Score -> Token was bad (caused high J)
            score = start_J - J_ablated
            feature_scores.append(score)

        ori_batch["score"] = feature_scores
        ori_batch["result_text_sim"]=sim_batch["result_text"]

        # Sort by score descending (Most critical tokens first)
        batch_sorted = ori_batch.sort_values(by="score", ascending=False)
        results.append(batch_sorted)

    print(f"Saving RIFair importance results to {save_file}")
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    # for data_name in ["sentiment", "biasinbios", "jiasaw",  "adultnew"]:
    #     model_names = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "microsoft/deberta-v3-base"]
    #     for name in model_names:
    #         clean_name = name.replace('/', '_')
    #
    #         importance_file1 = f"{data_name}_data/{data_name}_{clean_name}_importance_ori.pkl"
    #         importance_file2 = f"{data_name}_data/{data_name}_{clean_name}_importance_sim.pkl"
    #         attack_file = f"{data_name}_data/attack_data.pkl"
    #         save_file_R = f"{data_name}_data/{clean_name}_R_importance.pkl"
    #         save_file_F = f"{data_name}_data/{clean_name}_F_importance.pkl"
    #         save_file_TB = f"{data_name}_data/{clean_name}_TB_importance.pkl"
    #         save_file_FF = f"{data_name}_data/{clean_name}_FF_importance.pkl"
    #         save_file_FB = f"{data_name}_data/{clean_name}_FB_importance.pkl"
    #
    #         calculate_R_importance(importance_file1, importance_file2,save_file_R)
    #         calculate_F_importance(importance_file1, importance_file2, save_file_F)
    #         calculate_RIFair_importance(attack_file, importance_file1, importance_file2, save_file_TB, 1, -1, -1)
    #         calculate_RIFair_importance(attack_file, importance_file1, importance_file2, save_file_FF, -1, -1, 1)
    #         calculate_RIFair_importance(attack_file, importance_file1, importance_file2, save_file_FB, -1, 1, -1)
    for data_name in ["biasinbios"]:
        model_names = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "microsoft/deberta-v3-base"]
        for name in model_names:
            clean_name = name.replace('/', '_')
            importance_file1 = f"{data_name}_data/{data_name}_{clean_name}_retrain_importance_ori.pkl"
            importance_file2 = f"{data_name}_data/{data_name}_{clean_name}_retrain_importance_sim.pkl"
            attack_file = f"{data_name}_data/attack_retrain_data.pkl"
            save_file_R = f"{data_name}_data/{clean_name}_retrain_R_importance.pkl"
            save_file_F = f"{data_name}_data/{clean_name}_retrain_F_importance.pkl"
            save_file_TB = f"{data_name}_data/{clean_name}_retrain_TB_importance.pkl"
            save_file_FF = f"{data_name}_data/{clean_name}_retrain_FF_importance.pkl"
            save_file_FB = f"{data_name}_data/{clean_name}_retrain_FB_importance.pkl"
            calculate_R_importance(importance_file1, importance_file2,save_file_R)
            calculate_F_importance(importance_file1, importance_file2, save_file_F)
            calculate_RIFair_importance(attack_file, importance_file1, importance_file2, save_file_TB, 1, -1, -1)
            calculate_RIFair_importance(attack_file, importance_file1, importance_file2, save_file_FF, -1, -1, 1)
            calculate_RIFair_importance(attack_file, importance_file1, importance_file2, save_file_FB, -1, 1, -1)

