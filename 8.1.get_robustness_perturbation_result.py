import os
import pickle

import pandas
import torch
import textattack
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import BERTAttackLi2020
from textattack.datasets import Dataset


def load_custom_dataset(attack_path):
    """
    Loads validation data and converts it to TextAttack format: [(text, label), ...]
    """
    print(f"Loading data from {attack_path}...")
    with open(attack_path, 'rb') as f:
        attack_data = pickle.load(f)

    # CASE A: If input is a pandas DataFrame
    if isinstance(attack_data, pandas.DataFrame):
        # Ensure column names match your file (adjust 'result_text'/'label' if needed)
        data = list(zip(attack_data['result_text'], attack_data['label']))

    # CASE B: If input is a List of Dictionaries
    elif isinstance(attack_data, list) and isinstance(attack_data[0], dict):
        data = [(item['text_ori'], item['label']) for item in attack_data]

    else:
        data = attack_data

    return Dataset(data), len(data)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_names = ["sentiment", "biasinbios", "jiasaw", "adultnew"]
    model_paths = { "bert-base-uncased": "bert-base-uncased", "roberta-base": "roberta-base", "distilbert-base-uncased": "distilbert-base-uncased", "microsoft/deberta-v3-base": "microsoft/deberta-v3-base"}
    for data_name in data_names:
        # Load data ONCE per dataset (Efficiency optimization)
        attack_file = f"/kaggle/input/{data_name}/attack_data.pkl"

        try:
            dataset, num_example = load_custom_dataset(attack_file)
        except Exception as e:
            print(f"Skipping dataset {data_name}: {e}")
            continue
        for model_key, model_path_raw in model_paths.items():
            clean_model_name = model_key.replace('/', '_')
            model_file = f"/kaggle/input/{data_name}/{clean_model_name}_final"
            output_file = f"{data_name}_data/BERT/{clean_model_name}_attack_results.csv"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            try:
                print(f"\n--- Loading Model: {clean_model_name} ---")
                model = AutoModelForSequenceClassification.from_pretrained(model_file)
                tokenizer = AutoTokenizer.from_pretrained(model_file)
                model.to(device)

                model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

                # BUILD STANDARD BERT-ATTACK
                attack = BERTAttackLi2020.build(model_wrapper)

                attack_args = textattack.AttackArgs( num_examples=num_example, log_to_csv=output_file, checkpoint_interval=50, checkpoint_dir="checkpoints", disable_stdout=False, csv_coloring_style="plain" )

                attacker = textattack.Attacker(attack, dataset, attack_args)
                attacker.attack_dataset()
                print(f"✔ Results saved to {output_file}")

            except Exception as e:
                print(f"Error attacking {clean_model_name}: {e}")


