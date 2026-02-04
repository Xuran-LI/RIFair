import pandas
import pickle
import os
from tqdm import tqdm


def extract_replacements(text1, text2):
    """
    Generates ablation variations by removing one common token at a time.
    """
    tokens1 = text1.split()
    tokens2 = text2.split()
    ori_variations = []
    sim_variations = []
    ori_variations.append({"replace_token": "None", "replace_index": -1, "result": " ".join(tokens1)})
    sim_variations.append({"replace_token": "None", "replace_index": -1, "result": " ".join(tokens2)})

    # Iterate through the length of the shorter list to avoid IndexErrors
    min_len = min(len(tokens1), len(tokens2))
    for i in range(min_len):
        # Only ablate if tokens are the same (common context)
        if tokens1[i] == tokens2[i]:
            # Create new lists excluding the token at index 'i'
            new_tokens1 = tokens1[:i] + tokens1[i + 1:]
            new_tokens2 = tokens2[:i] + tokens2[i + 1:]
            # Store original similar version
            ori_variations.append({"replace_token": tokens1[i], "replace_index": i,  "result": " ".join(new_tokens1)})
            sim_variations.append({"replace_token": tokens2[i], "replace_index": i, "result": " ".join(new_tokens2)})

    return ori_variations, sim_variations


def get_feature_ablation_sentences(data_file, save_file):
    """
    Loads data, generates ablation samples for every instance, and saves the batch.
    """
    print(f"Loading data from {data_file}...")
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, list):
            df = pandas.DataFrame(data)
        else:
            df = pandas.DataFrame.from_dict(data)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    if "text_ori" not in df.columns or "text_sim" not in df.columns:
        print("Error: DataFrame must contain 'text_ori' and 'text_sim' columns")
        return
    ori_texts = df["text_ori"].tolist()
    sim_texts = df["text_sim"].tolist()
    print(f"Generating ablation data for {len(df)} samples...")
    instance_batch_ori = []
    instance_batch_sim = []
    for ori, sim in tqdm(zip(ori_texts, sim_texts), total=len(ori_texts)):
        variations_ori, variations_sim = extract_replacements(str(ori), str(sim))
        instance_batch_ori.append(variations_ori)
        instance_batch_sim.append(variations_sim)
    print(f"Saving batches to {save_file}...")
    output_data = {"instance_batch_ori": instance_batch_ori,"instance_batch_sim": instance_batch_sim }
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, 'wb') as f:
        pickle.dump(output_data, f)
    print("Done.")


# --- USAGE ---
if __name__ == "__main__":
    # # Adjust paths as needed
    # input_path = "biasinbios_data/attack_data.pkl"  # Input file path
    # output_path = "biasinbios_data/ablation_data.pkl"  # Output file path
    #
    # if os.path.exists(input_path):
    #     get_feature_ablation_sentences(input_path, output_path)
    # else:
    #     print(f"File not found: {input_path}")
    #
    # input_path = "jiasaw_data/attack_data.pkl"  # Input file path
    # output_path = "jiasaw_data/ablation_data.pkl"  # Output file path
    #
    # if os.path.exists(input_path):
    #     get_feature_ablation_sentences(input_path, output_path)
    # else:
    #     print(f"File not found: {input_path}")
    #
    # input_path = "Sentiment_data/attack_data.pkl"  # Input file path
    # output_path = "Sentiment_data/ablation_data.pkl"  # Output file path
    #
    # if os.path.exists(input_path):
    #     get_feature_ablation_sentences(input_path, output_path)
    # else:
    #     print(f"File not found: {input_path}")

    # input_path = "adultnew_data/attack_data.pkl"  # Input file path
    # output_path = "adultnew_data/ablation_data.pkl"  # Output file path
    #
    # if os.path.exists(input_path):
    #     get_feature_ablation_sentences(input_path, output_path)
    # else:
    #     print(f"File not found: {input_path}")

    input_path = "biasinbios_data/attack_retrain_data.pkl"  # Input file path
    output_path = "biasinbios_data/ablation_retrain_data.pkl"  # Output file path

    if os.path.exists(input_path):
        get_feature_ablation_sentences(input_path, output_path)
    else:
        print(f"File not found: {input_path}")





