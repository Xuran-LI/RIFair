import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================
# 1. MODEL WRAPPER
# ==========================================
class HuggingFaceWrapper(torch.nn.Module):
    def __init__(self, model_path, device='cuda'):
        """
        Wraps a HuggingFace model for inference.
        """
        super().__init__()
        self.device = device
        print(f"Loading model from: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def predict_proba(self, texts):
        """
        Takes a list of strings -> Returns probability array [N, num_labels]
        """
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]

        # 1. Tokenize (Batch processing)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        # 2. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply Softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        return probs.cpu().numpy()


# ==========================================
# 2. BATCH PROCESSING FUNCTION
# ==========================================
def calculate_and_save_batch_outputs(model, batch_data, save_file):
    """
    Runs inference on batches of ablated sentences.

    Args:
        model: HuggingFaceWrapper instance.
        batch_data: List of Lists of Dicts (from ablation_data.pkl).
                    Structure: [ [ {token:'man', result:'...'}, ... ], ... ]
        save_file: Path to save results.
    """
    all_results = []
    print(f"Processing {len(batch_data)} instances...")
    for i in range(len(batch_data)):
        batch = batch_data[i]
        texts = [item["result"] for item in batch]
        replace_index=[item["replace_index"] for item in batch]
        replace_token=[item["replace_token"] for item in batch]
        probs = model.predict_proba(texts)
        instance_results = []
        for j in range(len(batch)):
            result_entry = {"instance_index": replace_index[j], "replace_token": replace_token[j], "result_text": texts[j], "probs": probs[j] }
            instance_results.append(result_entry)
        all_results.append(instance_results)
    # Save to File
    print(f"Saving results to {save_file}...")
    with open(save_file, 'wb') as sf:
        pickle.dump(all_results, sf)
    print(f"✔ Saved.")
    return all_results


# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # dataset_names = ["sentiment", "jiagsaw", "biasinbios"]
    # for data_name in dataset_names:
    #     data_path = f"/kaggle/input/{data_name}/ablation_data.pkl"
    #     model_dir = f"/kaggle/input/{data_name}"
    #     save_path = "/kaggle/working/"
    #     print(f"Loading ablation data from {data_path}...")
    #     with open(data_path, 'rb') as f:
    #         data = pickle.load(f)
    #     ori_texts_batch = data["instance_batch_ori"]
    #     sim_texts_batch = data["instance_batch_sim"]
    #     model_names = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "microsoft/deberta-v3-base"]
    #     for name in model_names:
    #         clean_name = name.replace('/', '_')
    #         model_path = f"{model_dir}/{clean_name}_final"
    #         # 1. Load Model
    #         model_wrapper = HuggingFaceWrapper(model_path, device)
    #         # 2. Process Original Text Ablations
    #         save_file_ori = f"{save_path}/{data_name}_{clean_name}_importance_ori.pkl"
    #         calculate_and_save_batch_outputs(model_wrapper, ori_texts_batch, save_file_ori)
    #         save_file_sim = f"{save_path}/{data_name}_{clean_name}_importance_sim.pkl"
    #         calculate_and_save_batch_outputs(model_wrapper, sim_texts_batch, save_file_sim)
    #     print("\nAll processing complete.")
    dataset_names = ["biasinbios"]
    for data_name in dataset_names:
        data_path = f"/kaggle/input/{data_name}/ablation_retrain_data.pkl"
        model_dir = f"/kaggle/input/{data_name}"
        save_path = "/kaggle/working/"
        print(f"Loading ablation data from {data_path}...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        ori_texts_batch = data["instance_batch_ori"]
        sim_texts_batch = data["instance_batch_sim"]
        model_names = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "microsoft/deberta-v3-base"]
        for name in model_names:
            clean_name = name.replace('/', '_')
            model_path = f"{model_dir}/{clean_name}_final"
            # 1. Load Model
            model_wrapper = HuggingFaceWrapper(model_path, device)
            # 2. Process Original Text Ablations
            save_file_ori = f"{save_path}/{data_name}_{clean_name}_retrain_importance_ori.pkl"
            calculate_and_save_batch_outputs(model_wrapper, ori_texts_batch, save_file_ori)
            save_file_sim = f"{save_path}/{data_name}_{clean_name}_retrain_importance_sim.pkl"
            calculate_and_save_batch_outputs(model_wrapper, sim_texts_batch, save_file_sim)
        print("\nAll processing complete.")

