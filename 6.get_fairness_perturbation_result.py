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
        # Use .copy() on the passed arguments, NOT global variables
        perturb1 = original_tex_list.copy()
        perturb2 = similar_text_list.copy()

        perturb1[perturbation_index] = cand
        perturb2[perturbation_index] = cand

        text1 = " ".join(perturb1)
        text2 = " ".join(perturb2)

        result1.append(text1)
        result2.append(text2)
        valid_candidates.append(cand)

    return result1, result2, valid_candidates


# --- 3. Best Update Selection ---
def get_best_update(candidate_ori_texts, candidate_sim_texts, candidate_tokens, model, current_gap):
    """
    Finds the candidate that MAXIMIZES the gap (Attack success).
    """
    if not candidate_ori_texts:
        return None, current_gap, None, None, None

    # Batch Inference
    probs_ori_batch = model.predict_proba(candidate_ori_texts)  # Shape [N, 2]
    probs_sim_batch = model.predict_proba(candidate_sim_texts)  # Shape [N, 2]

    target_class = 1
    gaps = numpy.abs(probs_ori_batch[:, target_class] - probs_sim_batch[:, target_class])

    max_gap_idx = numpy.argmax(gaps)
    max_batch_gap = gaps[max_gap_idx]

    # If the found gap is larger than our current best, we take it
    if max_batch_gap > current_gap:
        replace_token = candidate_tokens[max_gap_idx]
        replace_pre_ori = probs_ori_batch[max_gap_idx]
        replace_pre_sim = probs_sim_batch[max_gap_idx]
        pred_label_ori = numpy.argmax(replace_pre_ori)
        pred_label_sim = numpy.argmax(replace_pre_sim)
        # Tag is False if labels are different (Successful attack on fairness consistency)
        fair_tag = (pred_label_ori == pred_label_sim)
        return fair_tag, max_batch_gap, replace_token, replace_pre_ori, replace_pre_sim
    return None, current_gap, None, None, None


def run_fairness_attack_pipeline(model, importance_path, output_path, attack_similarity, top_k=10):
    """
    Runs the RIFair attack on a specific dataset using pre-calculated feature importance.

    Args:
        model: The loaded HuggingFaceWrapper model.
        importance_path (str): Path to the _F_importance.pkl file.
        output_path (str): Path to save the _attack_results.pkl file.
        attack_similarity : Dictionary of {word: [synonyms]}.
        top_k (int): Number of top important tokens to attempt attacking per example.

    Returns:
        list: A list of successful attack result dictionaries.
    """
    results = []

    try:
        print(f"Loading importance file: {importance_path}")
        with open(importance_path, 'rb') as f:
            attack_importance = pickle.load(f)

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

            # --- 2. Prioritize Tokens ---
            # Sort by importance score (Descending = Most important first)
            example_df = example_df.sort_values(by="score", ascending=True)

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

                # # Check if we have synonyms for this word
                # if target_word not in attack_similarity:
                #     continue
                #
                # candidates = attack_similarity[target_word]

                # A. Generate Candidates
                c_ori, c_sim, c_tokens = generate_candidates_instances(ori_text, sim_text, idx_to_attack, target_word, attack_similarity, cosine_threshold=0.8)

                # B. Find Best Candidate (The one that maximizes the Gap)
                fair_tag, new_gap, replace_token, p_ori, p_sim = get_best_update(c_ori, c_sim, c_tokens, model, current_gap)

                # C. Update State if we found a better candidate
                if replace_token is not None:
                    # Apply the change permanently for this loop (Greedy approach)
                    ori_text[idx_to_attack] = replace_token
                    sim_text[idx_to_attack] = replace_token
                    current_gap = new_gap

                    # D. Log Success (Fairness Broken)
                    if not fair_tag:
                        results.append({"original_example_id": i, "attack_step": attack_step,"final_gap": float(new_gap), "final_text_ori": " ".join(ori_text), "final_text_sim": " ".join(sim_text), "probs_ori": p_ori, "probs_sim": p_sim})
                        print(f"bias attack success {i}")
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
            importance_file = f"/kaggle/input/{data_name}/{clean_model_name}_F_importance.pkl"
            output_file = f"{data_name}_data/MEFA/{clean_model_name}_attack_results.pkl"

            # Load Model
            try:
                print(f"--- Loading Model: {clean_model_name} ---")
                model = HuggingFaceWrapper(model_file, device)
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
                continue
            # Run the Pipeline Method
            run_fairness_attack_pipeline(model=model, importance_path=importance_file, output_path=output_file, attack_similarity=similarity_table, top_k=20)

    for data_name in data_names:
        print(f"\n=== Processing Dataset: {data_name} ===")
        # 1. Load Dictionary
        similarity_path = f"/kaggle/input/{data_name}/{data_name}_attack_similarity.pkl"
        similarity_table = transform_attack_similarity_into_table(similarity_path)
        # 2. Iterate Models
        for model_key, model_path_raw in model_paths.items():
            clean_model_name = model_key.replace('/', '_')
            # Paths
            model_file = f"/kaggle/input/{data_name}/{clean_model_name}_final"
            importance_file = f"/kaggle/input/{data_name}/{clean_model_name}_R_importance.pkl"
            output_file = f"{data_name}_data/ADF_TextFooler/{clean_model_name}_attack_results.pkl"
            # Load Model
            try:
                print(f"--- Loading Model: {clean_model_name} ---")
                model = HuggingFaceWrapper(model_file, device)
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
                continue
            # Run the Pipeline Method
            run_fairness_attack_pipeline(model=model, importance_path=importance_file, output_path=output_file, attack_similarity=similarity_table, top_k=20)


