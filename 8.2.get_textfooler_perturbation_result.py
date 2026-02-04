# # 测试各个模型 TextFooll,er step2: 根据属性重要性进行扰动
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
def get_best_update(candidate_ori_texts, candidate_label, candidate_tokens, model, current_prob):
    """
    Finds the candidate that MINIMIZES the probability of the true label (Attack success).
    """
    if not candidate_ori_texts:
        return None, current_prob, None, None

    # Batch Inference
    # Shape [N, 2] -> e.g., [[0.1, 0.9], [0.4, 0.6]]
    probs_ori_batch = model.predict_proba(candidate_ori_texts)

    # 1. FIX: Use ARGMIN. We want to DROP the probability of the true label.
    # Extract prob of true class for all candidates
    true_class_probs = probs_ori_batch[:, candidate_label]
    min_prob_idx = numpy.argmin(true_class_probs)

    best_candidate_prob = true_class_probs[min_prob_idx]

    # 2. FIX: Compare scalar vs scalar
    # If this candidate lowers the confidence more than our current best, take it
    if best_candidate_prob < current_prob:
        replace_token = candidate_tokens[min_prob_idx]

        # Get the full probability vector for logging (FIX 1: Capture Vector)
        replace_probs = probs_ori_batch[min_prob_idx]

        # Check if attack succeeded (Label Flip)
        pred_label = numpy.argmax(replace_probs)
        attack_success = (pred_label != candidate_label)

        # FIX 2: Return the vector 'replace_probs' too
        return attack_success, best_candidate_prob, replace_token, replace_probs

    # No improvement found
    return False, current_prob, None, None


def run_textfooler_attack_pipeline(model, importance_path, attack_path, output_path, attack_similarity, top_k=10):
    results = []

    try:
        print(f"Loading files...\n Importance: {importance_path}\n Attack Data: {attack_path}")
        with open(importance_path, 'rb') as f:
            attack_importance = pickle.load(f)
        with open(attack_path, 'rb') as f:
            attack_data = pickle.load(f)

    except FileNotFoundError as e:
        print(f"Skipping: {e}")
        return []
    except Exception as e:
        print(f"Error loading files: {e}")
        return []

    # Iterate through each example in the dataset
    for i, example_df in enumerate(attack_importance):
        try:
            # 1. Extract Data
            if 'instance_index' in example_df.columns:
                row = example_df[example_df['instance_index'] == -1]
                if row.empty: row = example_df.iloc[[0]]
            else:
                row = example_df.iloc[[0]]

            ori_text = row.iloc[0]['result_text'].strip().split()
            sim_text = row.iloc[0]['result_text_sim'].strip().split()

            # FIX: Ensure label is an integer
            initial_label = int(attack_data[i]['label'])

            # 2. Sort by Importance (Descending = Attack most important words first)
            # FIX: changed ascending=True to ascending=False
            example_df = example_df.sort_values(by="score", ascending=False)
            valid_attacks = example_df[example_df['instance_index'] != -1]
            perturbation_token_indexes = valid_attacks["instance_index"].tolist()

            # 3. Calculate Initial State (Scalar Probability)
            # We predict once to get the baseline confidence
            initial_probs = model.predict_proba([" ".join(ori_text)])[0]
            current_prob = initial_probs[initial_label]  # e.g., 0.95

            p_ori = initial_probs
            p_sim = None  # Will update if we change something

            # 4. Greedy Attack Loop
            steps_limit = min(top_k, len(perturbation_token_indexes))

            for attack_step in range(steps_limit):
                idx_to_attack = perturbation_token_indexes[attack_step]

                if idx_to_attack >= len(ori_text): continue
                target_word = ori_text[idx_to_attack]

                # A. Generate Candidates
                # Note: Assuming 'generate_candidates_instances' handles the similarity logic defined previously
                c_ori, c_sim, c_tokens = generate_candidates_instances(ori_text, sim_text, idx_to_attack, target_word, attack_similarity, cosine_threshold=0.8, Seq_threshold=None)

                # B. Find Best Candidate
                success_tag, new_prob, replace_token, new_probs_vec = get_best_update(c_ori, initial_label, c_tokens, model, current_prob)

                # C. Update State if better
                if replace_token is not None:
                    # Update Text
                    ori_text[idx_to_attack] = replace_token
                    sim_text[idx_to_attack] = replace_token  # Symmetric update
                    current_prob = new_prob
                    p_ori = new_probs_vec
                    if success_tag:  # Label has flipped
                        results.append({"original_example_id": i, "attack_step": attack_step, "final_prob": float(new_prob), "final_text_ori": " ".join(ori_text), "final_text_sim": " ".join(sim_text), "probs_ori": p_ori})
                        print(f"  [Success] ID {i}: TextFooler attack successful (Step {attack_step})")
                        break

            results.append({"original_example_id": i, "attack_step": steps_limit, "final_prob": float(current_prob), "final_text_ori": " ".join(ori_text), "final_text_sim": " ".join(sim_text), "probs_ori": p_ori})

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue

    # 5. Save Results
    if results:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"  [Saved] {len(results)} successful attacks to {output_path}")

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
            # Paths
            model_file = f"/kaggle/input/{data_name}/{clean_model_name}_final"
            importance_file = f"/kaggle/input/{data_name}/{clean_model_name}_R_importance.pkl"
            attack_file = f"/kaggle/input/{data_name}/attack_data.pkl"
            output_file = f"{data_name}_data/TextFooler/{clean_model_name}_attack_results.pkl"
            # Load Model
            try:
                print(f"--- Loading Model: {clean_model_name} ---")
                model = HuggingFaceWrapper(model_file, device)
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
                continue
            # Run the Pipeline Method
            run_textfooler_attack_pipeline(model=model, importance_path=importance_file, attack_path=attack_file, output_path=output_file, attack_similarity=similarity_table, top_k=20)


