# # 测试各个模型 Fairness step2: 根据属性重要性进行扰动
import pickle
import numpy
import os

import pandas
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# --- 1. Model Wrapper ---
class HuggingFaceWrapper(torch.nn.Module):
    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()

    def predict_proba(self, texts):
        inumpyuts = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inumpyuts)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()



# --- 2. Candidate Generation ---
def generate_candidates_instances(original_tex_list, similar_text_list, perturbation_index, token, similarity_df, cosine_threshold=None, Seq_threshold=None):
    """
    Generates candidate sentences by replacing a specific token with synonyms
    filtered by similarity thresholds.
    """
    result1 = []
    result2 = []
    valid_candidates = []

    # --- FIX 1: Correct Pandas Filtering Syntax ---
    # Step A: Filter rows where the 'value' column matches the target token
    # Syntax: df[df['column_name'] == value]
    token_matches = similarity_df[similarity_df["value"] == token]

    # Step B: Apply the specific threshold
    if cosine_threshold is not None:
        # Filter where cosine similarity is greater than the threshold
        perturbation_candidates = token_matches[token_matches["cosine"] >= cosine_threshold]
    elif Seq_threshold is not None:
        # Filter where Sequence score is greater than the threshold
        perturbation_candidates = token_matches[token_matches["Seq_score"] >= Seq_threshold]
    else:
        # Safety fallback if no threshold is provided
        return [], [], []

    # --- FIX 2: Iterate safely ---
    # .tolist() ensures we iterate over Python strings, not Series objects
    for cand in perturbation_candidates["candidate"].tolist():
        # 1. Create Copies (Crucial to avoid modifying the original list)
        perturb1 = original_tex_list.copy()
        perturb2 = similar_text_list.copy()

        # 2. Perform Replacement
        perturb1[perturbation_index] = cand
        perturb2[perturbation_index] = cand

        # 3. Reconstruct Strings
        text1 = " ".join(perturb1)
        text2 = " ".join(perturb2)

        # 4. Append Results
        result1.append(text1)
        result2.append(text2)
        valid_candidates.append(cand)

    return result1, result2, valid_candidates


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


# --- 3. Best Update Selection ---
def get_best_TB_update(candidate_ori_texts, candidate_sim_texts, ori_label, candidate_tokens, model, current_J):
    """
    Finds the candidate that MAXIMIZES the gap (Attack success).
    """
    if not candidate_ori_texts:
        return None, current_J, None, None, None, None

    # Batch Inference
    probs_ori_batch = model.predict_proba(candidate_ori_texts)  # Shape [N, 2]
    probs_sim_batch = model.predict_proba(candidate_sim_texts)  # Shape [N, 2]

    # 3. Find Extremes (The 4 Corners)
    # We use the probabil   ity of the TRUE LABEL to find the most "Robust" and most "Fragile" points
    idx_max_ori = numpy.argmax(probs_ori_batch[:, ori_label])
    idx_min_ori = numpy.argmin(probs_ori_batch[:, ori_label])
    idx_max_sim = numpy.argmax(probs_sim_batch[:, ori_label])
    idx_min_sim = numpy.argmin(probs_sim_batch[:, ori_label])
    pairs_to_check = [(idx_max_ori, idx_min_sim), (idx_min_ori, idx_max_sim), (idx_max_ori, idx_max_sim), (idx_min_ori, idx_min_sim)]
    # Initialize placeholders
    best_J = current_J
    best_adv_prob_vec = None
    best_sim_prob_vec = None
    adv_token=None
    sim_token=None

    # 4. Check the 4 Candidates
    for i, j in pairs_to_check:
        # Extract Probability Vectors [P(0), P(1)]
        p_ori_vec = probs_ori_batch[i]
        p_sim_vec = probs_sim_batch[j]
        new_J = calculate_J(p_ori_vec, p_sim_vec, ori_label, 1, -1, -1)
        # Optimization: We want to MINIMIZE J
        if new_J < best_J:
            best_J = new_J
            best_adv_prob_vec = p_ori_vec  # Store vector [p0, p1]
            best_sim_prob_vec = p_sim_vec  # Store vector [p0, p1]
            adv_token = candidate_tokens[i]
            sim_token = candidate_tokens[j]

    # 5. Handle Results
    if best_adv_prob_vec is not None:
        # CRITICAL FIX: Use argmax to determine the label from the vector
        pred_label_ori = numpy.argmax(best_adv_prob_vec)
        pred_label_sim = numpy.argmax(best_sim_prob_vec)
        # Acc Tag: Is the prediction correct?
        acc_tag = (pred_label_ori == ori_label)
        fair_tag = (pred_label_ori == pred_label_sim)
        if acc_tag and not fair_tag:
            TB_tag=True
        else:
            TB_tag=False

        return TB_tag, best_J, best_adv_prob_vec, best_sim_prob_vec, adv_token, sim_token
    else:
        # No improvement found
        return None,  current_J, None, None, None, None


def get_best_FB_update(candidate_ori_texts, candidate_sim_texts, ori_label, candidate_tokens, model, current_J):
    """
    Finds the candidate that MAXIMIZES the gap (Attack success).
    """
    if not candidate_ori_texts:
        return None, current_J, None, None, None, None

    # Batch Inference
    probs_ori_batch = model.predict_proba(candidate_ori_texts)  # Shape [N, 2]
    probs_sim_batch = model.predict_proba(candidate_sim_texts)  # Shape [N, 2]

    # 3. Find Extremes (The 4 Corners)
    # We use the probabil   ity of the TRUE LABEL to find the most "Robust" and most "Fragile" points
    idx_max_ori = numpy.argmax(probs_ori_batch[:, ori_label])
    idx_min_ori = numpy.argmin(probs_ori_batch[:, ori_label])
    idx_max_sim = numpy.argmax(probs_sim_batch[:, ori_label])
    idx_min_sim = numpy.argmin(probs_sim_batch[:, ori_label])
    pairs_to_check = [(idx_max_ori, idx_min_sim), (idx_min_ori, idx_max_sim), (idx_max_ori, idx_max_sim), (idx_min_ori, idx_min_sim)]
    # Initialize placeholders
    best_J = current_J
    best_adv_prob_vec = None
    best_sim_prob_vec = None
    adv_token=None
    sim_token=None

    # 4. Check the 4 Candidates
    for i, j in pairs_to_check:
        # Extract Probability Vectors [P(0), P(1)]
        p_ori_vec = probs_ori_batch[i]
        p_sim_vec = probs_sim_batch[j]
        new_J = calculate_J(p_ori_vec, p_sim_vec, ori_label, -1, 1, -1)
        # Optimization: We want to MINIMIZE J
        if new_J < best_J:
            best_J = new_J
            best_adv_prob_vec = p_ori_vec  # Store vector [p0, p1]
            best_sim_prob_vec = p_sim_vec  # Store vector [p0, p1]
            adv_token = candidate_tokens[i]
            sim_token = candidate_tokens[j]

    # 5. Handle Results
    if best_adv_prob_vec is not None:
        # CRITICAL FIX: Use argmax to determine the label from the vector
        pred_label_ori = numpy.argmax(best_adv_prob_vec)
        pred_label_sim = numpy.argmax(best_sim_prob_vec)
        # Acc Tag: Is the prediction correct?
        acc_tag = (pred_label_ori == ori_label)
        fair_tag = (pred_label_ori == pred_label_sim)
        if not acc_tag and not fair_tag:
            FB_tag=True
        else:
            FB_tag=False

        return FB_tag, best_J, best_adv_prob_vec, best_sim_prob_vec, adv_token, sim_token
    else:
        # No improvement found
        return None, current_J, None, None, None, None


def get_best_FF_update(candidate_ori_texts, candidate_sim_texts, ori_label, candidate_tokens, model, current_J):
    """
    Finds the candidate that MAXIMIZES the gap (Attack success).
    """
    if not candidate_ori_texts:
        return None, current_J, None, None, None, None

    # Batch Inference
    probs_ori_batch = model.predict_proba(candidate_ori_texts)  # Shape [N, 2]
    probs_sim_batch = model.predict_proba(candidate_sim_texts)  # Shape [N, 2]

    # 3. Find Extremes (The 4 Corners)
    # We use the probabil   ity of the TRUE LABEL to find the most "Robust" and most "Fragile" points
    idx_max_ori = numpy.argmax(probs_ori_batch[:, ori_label])
    idx_min_ori = numpy.argmin(probs_ori_batch[:, ori_label])
    idx_max_sim = numpy.argmax(probs_sim_batch[:, ori_label])
    idx_min_sim = numpy.argmin(probs_sim_batch[:, ori_label])
    pairs_to_check = [(idx_max_ori, idx_min_sim), (idx_min_ori, idx_max_sim), (idx_max_ori, idx_max_sim), (idx_min_ori, idx_min_sim)]
    # Initialize placeholders
    best_J = current_J
    best_adv_prob_vec = None
    best_sim_prob_vec = None
    adv_token=None
    sim_token=None

    # 4. Check the 4 Candidates
    for i, j in pairs_to_check:
        # Extract Probability Vectors [P(0), P(1)]
        p_ori_vec = probs_ori_batch[i]
        p_sim_vec = probs_sim_batch[j]
        new_J = calculate_J(p_ori_vec, p_sim_vec, ori_label, -1, -1, 1)
        # Optimization: We want to MINIMIZE J
        if new_J < best_J:
            best_J = new_J
            best_adv_prob_vec = p_ori_vec  # Store vector [p0, p1]
            best_sim_prob_vec = p_sim_vec  # Store vector [p0, p1]
            adv_token = candidate_tokens[i]
            sim_token = candidate_tokens[j]

    # 5. Handle Results
    if best_adv_prob_vec is not None:
        # CRITICAL FIX: Use argmax to determine the label from the vector
        pred_label_ori = numpy.argmax(best_adv_prob_vec)
        pred_label_sim = numpy.argmax(best_sim_prob_vec)
        # Acc Tag: Is the prediction correct?
        acc_tag = (pred_label_ori == ori_label)
        fair_tag = (pred_label_ori == pred_label_sim)
        if not acc_tag and fair_tag:
            FF_tag=True
        else:
            FF_tag=False

        return FF_tag, best_J, best_adv_prob_vec, best_sim_prob_vec, adv_token, sim_token
    else:
        # No improvement found
        return None, current_J, None, None, None, None


def run_TB_attack_pipeline(model, importance_path, attack_path, output_path, attack_similarity, top_k=10):
    """
    Runs the RIFair attack on a specific dataset using pre-calculated feature importance.

    Args:
        model: The loaded HuggingFaceWrapper model.
        importance_path (str): Path to the _F_importance.pkl file.
        attack_path : Path for attack data
        output_path (str): Path to save the _attack_results.pkl file.
        attack_similarity : Dictionary of {word: [synonyms]}. and the cosine and Seq evaluation
        top_k (int): Number of top important tokens to attempt attacking per example.

    Returns:
        list: A list of successful attack result dictionaries.
    """
    results = []

    try:
        print(f"Loading importance file: {importance_path}")
        with open(importance_path, 'rb') as f:
            attack_importance = pickle.load(f)
        with open(attack_path, 'rb') as f:
            attack_data = pickle.load(f)

    except FileNotFoundError:
        print(f"Skipping: File not found {importance_path}")
        return []
    except Exception as e:
        print(f"Error loading {importance_path}: {e}")
        return []

    # Iterate through each example in the dataset
    for i, example_df in enumerate(attack_importance):
        try:
            # We look for the row where instance_index == -1 (metadata row)
            if 'instance_index' in example_df.columns:
                row = example_df[example_df['instance_index'] == -1]
                if row.empty:
                    row = example_df.iloc[[0]]
            else:
                row = example_df.iloc[[0]]

            # Convert text string to list of tokens
            ori_text = row.iloc[0]['result_text'].strip().split()
            sim_text = row.iloc[0]['result_text_sim'].strip().split()
            initial_gap = row.iloc[0]['score']
            initial_label = attack_data[i]['label']

            # --- 2. Prioritize Tokens ---
            # Sort by importance score (Descending = Most important first)
            example_df = example_df.sort_values(by="score", ascending=False)

            # Filter out the metadata row so we only attack actual tokens
            valid_attacks = example_df[example_df['instance_index'] != -1]
            perturbation_token_indexes = valid_attacks["instance_index"].tolist()

            # --- 3. Calculate Initial State ---
            current_gap = initial_gap
            p_ori=None
            p_sim=None
            # --- 4. Greedy Attack Loop ---
            # We try to perturb the top_k most important tokens
            steps_limit = min(top_k, len(perturbation_token_indexes))

            for attack_step in range(steps_limit):
                idx_to_attack = perturbation_token_indexes[attack_step]

                # Boundary check
                if idx_to_attack >= len(ori_text):
                    continue

                target_word = ori_text[idx_to_attack]
                # A. Generate Candidates
                c_ori, c_sim, c_tokens = generate_candidates_instances(ori_text, sim_text, idx_to_attack, target_word, attack_similarity, Seq_threshold=0.8)
                if len(c_ori) <1:
                    continue

                # B. Find Best Candidate (The one that maximizes the Gap)
                TB_tag, new_J, p_ori, p_sim, ori_token, sim_token = get_best_TB_update(c_ori, c_sim, initial_label, c_tokens, model, current_gap)

                # C. Update State if we found a better candidate
                if ori_token is not None:
                    # Apply the change permanently for this loop (Greedy approach)
                    ori_text[idx_to_attack] = ori_token
                    sim_text[idx_to_attack] = sim_token
                    current_gap = new_J

                    # D. Log Success (TB Broken)
                    if TB_tag:
                        results.append({"original_example_id": i, "attack_step": attack_step,"final_gap": float(new_J), "final_text_ori": " ".join(ori_text), "final_text_sim": " ".join(sim_text), "probs_ori": p_ori, "probs_sim": p_sim})
                        print(f"true bias attack success {i}")
                        break
            results.append({"original_example_id": i, "attack_step": steps_limit, "final_gap": float(current_gap), "final_text_ori": " ".join(ori_text), "final_text_sim": " ".join(sim_text),  "probs_ori": p_ori, "probs_sim": p_sim})

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue

    # --- 5. Save Results ---
    if results:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"  [Saved] {len(results)} successful attacks to {output_path}")
    else:
        print(f"  [Info] No successful attacks found for {output_path}")

    return results


def run_FB_attack_pipeline(model, importance_path, attack_path, output_path, attack_similarity, top_k=10):
    """
    Runs the RIFair attack on a specific dataset using pre-calculated feature importance.

    Args:
        model: The loaded HuggingFaceWrapper model.
        importance_path (str): Path to the _F_importance.pkl file.
        attack_path : Path for attack data
        output_path (str): Path to save the _attack_results.pkl file.
        attack_similarity ): Dictionary of {word: [synonyms]}.
        top_k (int): Number of top important tokens to attempt attacking per example.

    Returns:
        list: A list of successful attack result dictionaries.
    """
    results = []

    try:
        print(f"Loading importance file: {importance_path}")
        with open(importance_path, 'rb') as f:
            attack_importance = pickle.load(f)
        with open(attack_path, 'rb') as f:
            attack_data = pickle.load(f)

    except FileNotFoundError:
        print(f"Skipping: File not found {importance_path}")
        return []
    except Exception as e:
        print(f"Error loading {importance_path}: {e}")
        return []

    # Iterate through each example in the dataset
    for i, example_df in enumerate(attack_importance):
        try:
            # We look for the row where instance_index == -1 (metadata row)
            if 'instance_index' in example_df.columns:
                row = example_df[example_df['instance_index'] == -1]
                if row.empty:
                    row = example_df.iloc[[0]]
            else:
                row = example_df.iloc[[0]]

            # Convert text string to list of tokens
            ori_text = row.iloc[0]['result_text'].strip().split()
            sim_text = row.iloc[0]['result_text_sim'].strip().split()
            initial_gap = row.iloc[0]['score']
            initial_label = attack_data[i]['label']

            # --- 2. Prioritize Tokens ---
            # Sort by importance score (Descending = Most important first)
            example_df = example_df.sort_values(by="score", ascending=False)

            # Filter out the metadata row so we only attack actual tokens
            valid_attacks = example_df[example_df['instance_index'] != -1]
            perturbation_token_indexes = valid_attacks["instance_index"].tolist()

            # --- 3. Calculate Initial State ---
            current_gap = initial_gap
            p_ori=None
            p_sim=None
            # --- 4. Greedy Attack Loop ---
            # We try to perturb the top_k most important tokens
            steps_limit = min(top_k, len(perturbation_token_indexes))

            for attack_step in range(steps_limit):
                idx_to_attack = perturbation_token_indexes[attack_step]

                # Boundary check
                if idx_to_attack >= len(ori_text):
                    continue

                target_word = ori_text[idx_to_attack]

                # A. Generate Candidates
                c_ori, c_sim, c_tokens = generate_candidates_instances(ori_text, sim_text, idx_to_attack, target_word, attack_similarity, Seq_threshold=0.8)

                # B. Find Best Candidate (The one that maximizes the Gap)
                TB_tag, new_J, p_ori, p_sim, ori_token, sim_token = get_best_FB_update(c_ori, c_sim, initial_label, c_tokens, model, current_gap)

                # C. Update State if we found a better candidate
                if ori_token is not None:
                    # Apply the change permanently for this loop (Greedy approach)
                    ori_text[idx_to_attack] = ori_token
                    sim_text[idx_to_attack] = sim_token
                    current_gap = new_J

                    # D. Log Success (FB Broken)
                    if TB_tag:
                        results.append({"original_example_id": i, "attack_step": attack_step,"final_gap": float(new_J), "final_text_ori": " ".join(ori_text), "final_text_sim": " ".join(sim_text), "probs_ori": p_ori, "probs_sim": p_sim})
                        print(f"false bias attack success {i}")
                        break
            results.append({"original_example_id": i, "attack_step": steps_limit, "final_gap": float(current_gap), "final_text_ori": " ".join(ori_text), "final_text_sim": " ".join(sim_text),  "probs_ori": p_ori, "probs_sim": p_sim})

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue

    # --- 5. Save Results ---
    if results:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"  [Saved] {len(results)} successful attacks to {output_path}")
    else:
        print(f"  [Info] No successful attacks found for {output_path}")

    return results


def run_FF_attack_pipeline(model, importance_path, attack_path, output_path, attack_similarity, top_k=10):
    """
    Runs the RIFair attack on a specific dataset using pre-calculated feature importance.

    Args:
        model: The loaded HuggingFaceWrapper model.
        importance_path (str): Path to the _F_importance.pkl file.
        attack_path : Path for attack data
        output_path (str): Path to save the _attack_results.pkl file.
        attack_similarity): Dictionary of {word: [synonyms]}.
        top_k (int): Number of top important tokens to attempt attacking per example.

    Returns:
        list: A list of successful attack result dictionaries.
    """
    results = []

    try:
        print(f"Loading importance file: {importance_path}")
        with open(importance_path, 'rb') as f:
            attack_importance = pickle.load(f)
        with open(attack_path, 'rb') as f:
            attack_data = pickle.load(f)

    except FileNotFoundError:
        print(f"Skipping: File not found {importance_path}")
        return []
    except Exception as e:
        print(f"Error loading {importance_path}: {e}")
        return []

    # Iterate through each example in the dataset
    for i, example_df in enumerate(attack_importance):
        try:
            # We look for the row where instance_index == -1 (metadata row)
            if 'instance_index' in example_df.columns:
                row = example_df[example_df['instance_index'] == -1]
                if row.empty:
                    row = example_df.iloc[[0]]
            else:
                row = example_df.iloc[[0]]

            # Convert text string to list of tokens
            ori_text = row.iloc[0]['result_text'].strip().split()
            sim_text = row.iloc[0]['result_text_sim'].strip().split()
            initial_gap = row.iloc[0]['score']
            initial_label = attack_data[i]['label']

            # --- 2. Prioritize Tokens ---
            # Sort by importance score (Descending = Most important first)
            example_df = example_df.sort_values(by="score", ascending=False)

            # Filter out the metadata row so we only attack actual tokens
            valid_attacks = example_df[example_df['instance_index'] != -1]
            perturbation_token_indexes = valid_attacks["instance_index"].tolist()

            # --- 3. Calculate Initial State ---
            current_gap = initial_gap
            p_ori=None
            p_sim=None
            # --- 4. Greedy Attack Loop ---
            # We try to perturb the top_k most important tokens
            steps_limit = min(top_k, len(perturbation_token_indexes))

            for attack_step in range(steps_limit):
                idx_to_attack = perturbation_token_indexes[attack_step]

                # Boundary check
                if idx_to_attack >= len(ori_text):
                    continue

                target_word = ori_text[idx_to_attack]

                # A. Generate Candidates
                c_ori, c_sim, c_tokens = generate_candidates_instances(ori_text, sim_text, idx_to_attack, target_word, attack_similarity, Seq_threshold=0.8)

                # B. Find Best Candidate (The one that maximizes the Gap)
                FF_tag, new_J, p_ori, p_sim, ori_token, sim_token = get_best_FF_update(c_ori, c_sim, initial_label, c_tokens, model, current_gap)

                # C. Update State if we found a better candidate
                if ori_token is not None:
                    # Apply the change permanently for this loop (Greedy approach)
                    ori_text[idx_to_attack] = ori_token
                    sim_text[idx_to_attack] = sim_token
                    current_gap = new_J

                    # D. Log Success (FF Broken)
                    if FF_tag:
                        results.append({"original_example_id": i, "attack_step": attack_step,"final_gap": float(new_J), "final_text_ori": " ".join(ori_text), "final_text_sim": " ".join(sim_text), "probs_ori": p_ori, "probs_sim": p_sim})
                        print(f"false fair attack success {i}")
                        break
            results.append({"original_example_id": i, "attack_step": steps_limit, "final_gap": float(current_gap), "final_text_ori": " ".join(ori_text), "final_text_sim": " ".join(sim_text),  "probs_ori": p_ori, "probs_sim": p_sim})

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue

    # --- 5. Save Results ---
    if results:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"  [Saved] {len(results)} successful attacks to {output_path}")
    else:
        print(f"  [Info] No successful attacks found for {output_path}")

    return results


def transform_attack_similarity_into_table(similarity_path):
    # 1. Load Data
    with open(similarity_path, 'rb') as dic_f:
        attack_similarity = pickle.load(dic_f)

    # 2. Extract Data (Do not include headers here)
    data_rows = []
    for (value, candidate), cosine_score, Seq_score in attack_similarity:
        data_rows.append([value, candidate, cosine_score, Seq_score])

    # 3. Create DataFrame (Pass headers to the 'columns' argument)
    df = pandas.DataFrame(data_rows, columns=["value", "candidate", "cosine", "Seq_score"])

    return df


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    data_names = ["sentiment", "biasinbios", "jiasaw", "adultnew"]
    model_paths = { "bert-base-uncased": "bert-base-uncased", "roberta-base": "roberta-base", "distilbert-base-uncased": "distilbert-base-uncased", "microsoft/deberta-v3-base": "microsoft/deberta-v3-base" }
    for data_name in data_names:
        print(f"\n=== Processing Dataset: {data_name} ===")
        # 1. Load Dictionary
        similarity_path = f"/kaggle/input/{data_name}/{data_name}_attack_similarity.pkl"
        similarity_table = transform_attack_similarity_into_table(similarity_path)

        # 2. Iterate Models
        for model_key, model_path_raw in model_paths.items():
            clean_model_name = model_key.replace('/', '_')
            model_file = f"/kaggle/input/{data_name}/{clean_model_name}_final"
            attack_file = f"/kaggle/input/{data_name}/attack_data.pkl"
            importance_file = f"/kaggle/input/{data_name}/{clean_model_name}_TB_importance.pkl"
            output_file = f"{data_name}_data/{clean_model_name}_TB_attack_results.pkl"
            # Load Model
            try:
                print(f"--- Loading Model: {clean_model_name} ---")
                model = HuggingFaceWrapper(model_file, device)
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
                continue
            # Run the Pipeline Method
            run_TB_attack_pipeline(model=model, importance_path=importance_file, attack_path=attack_file, output_path=output_file, attack_similarity=similarity_table, top_k=20)

    for data_name in data_names:
        print(f"\n=== Processing Dataset: {data_name} ===")
        # 1. Load Dictionary
        similarity_path = f"/kaggle/input/{data_name}/{data_name}_attack_similarity.pkl"
        similarity_table = transform_attack_similarity_into_table(similarity_path)
        # 2. Iterate Models
        for model_key, model_path_raw in model_paths.items():
            clean_model_name = model_key.replace('/', '_')
            model_file = f"/kaggle/input/{data_name}/{clean_model_name}_final"
            attack_file = f"/kaggle/input/{data_name}/attack_data.pkl"
            importance_file = f"/kaggle/input/{data_name}/{clean_model_name}_FB_importance.pkl"
            output_file = f"{data_name}_data/{clean_model_name}_FB_attack_results.pkl"
            # Load Model
            try:
                print(f"--- Loading Model: {clean_model_name} ---")
                model = HuggingFaceWrapper(model_file, device)
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
                continue
            # Run the Pipeline Method
            run_FB_attack_pipeline(model=model, importance_path=importance_file, attack_path=attack_file, output_path=output_file, attack_similarity=similarity_table, top_k=20)


    for data_name in data_names:
        print(f"\n=== Processing Dataset: {data_name} ===")
        # 1. Load Dictionary
        similarity_path = f"/kaggle/input/{data_name}/{data_name}_attack_similarity.pkl"
        similarity_table = transform_attack_similarity_into_table(similarity_path)
        # 2. Iterate Models
        for model_key, model_path_raw in model_paths.items():
            clean_model_name = model_key.replace('/', '_')
            model_file = f"/kaggle/input/{data_name}/{clean_model_name}_final"
            attack_file = f"/kaggle/input/{data_name}/attack_data.pkl"
            importance_file = f"/kaggle/input/{data_name}/{clean_model_name}_FF_importance.pkl"
            output_file = f"{data_name}_data/{clean_model_name}_FF_attack_results.pkl"
            # Load Model
            try:
                print(f"--- Loading Model: {clean_model_name} ---")
                model = HuggingFaceWrapper(model_file, device)
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
                continue
            # Run the Pipeline Method
            run_FF_attack_pipeline(model=model, importance_path=importance_file, attack_path=attack_file, output_path=output_file, attack_similarity=similarity_table, top_k=20)



