# !pip install -U transformers datasets torch pandas numpy scikit-learn accelerate sentencepiece protobuf pyarrow
import os
import pickle

import numpy
import pandas
import torch
import warnings

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding)
from torch.utils.data import Dataset
from datasets import load_dataset

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_len)
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = numpy.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


def train_transformer(model_name, train_texts, train_labels, val_texts, val_labels, num_labels, output_dir):
    print(f"\n{'=' * 40}")
    print(f" TRAINING: {model_name} | Labels: {num_labels}")
    print(f"{'=' * 40}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    eval_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)

    # 4. Training Args (Optimized for Speed)
    clean_name = model_name.replace('/', '_')
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{clean_name}",
        num_train_epochs=3,  # 3 Epochs is standard
        per_device_train_batch_size=32,  # Increased to 32 (T4 GPU can handle this with dynamic padding)
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        dataloader_num_workers=2,  # Speeds up data loading
        fp16=torch.cuda.is_available(),  # Mixed Precision
        logging_dir='./logs',
        report_to=None
    )

    # 5. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator )

    trainer.train()

    # 6. Save Final Model
    final_path = f"{output_dir}/{clean_name}_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✔ Saved final model to: {final_path}")


def load_jigsaw_data():
    print("--- Loading Jigsaw Toxicity Data ---")
    try:
        path = "/kaggle/input/jiasaw/all_data.csv"
        raw_data = pandas.read_csv(path)
        train_df = raw_data[raw_data['split'] == 'train'].copy()
        val_df = raw_data[raw_data['split'] == 'test'].copy()
        train_df = train_df.dropna(subset=['comment_text'])
        val_df = val_df.dropna(subset=['comment_text'])
        label_col = 'toxic' if 'toxic' in train_df.columns else 'toxicity'
        # Train Split (5%)
        train_df['strat_col'] = (train_df[label_col] >= 0.5).astype(int)
        train_df, _ = train_test_split(train_df, train_size=0.02, stratify=train_df['strat_col'], random_state=42 )

        # Validation Split (5%) - FIX: Used train_test_split instead of val_df()
        val_df['strat_col'] = (val_df[label_col] >= 0.5).astype(int)
        val_df, _ = train_test_split(val_df,train_size=0.02, stratify=val_df['strat_col'],random_state=42)

        y_train = train_df['strat_col'].tolist()
        y_val = val_df['strat_col'].tolist()
        X_train = train_df['comment_text'].tolist()
        X_val = val_df['comment_text'].tolist()

        return X_train, y_train, X_val, y_val, 2
    except Exception as e:
        print(f"Error loading Jigsaw: {e}")
        return [], [], [], [], 0


def load_bias_in_bios_data():
    print("--- Loading Bias in Bios Data ---")
    try:
        splits = {'train': 'data/train-00000-of-00001-0ab65b32c47407e8.parquet',
                  'test': 'data/test-00000-of-00001-5598c840ce8de1ee.parquet'}
        path_prefix = "hf://datasets/LabHC/bias_in_bios/"
        train_df = pandas.read_parquet(path_prefix + splits["train"])
        val_df = pandas.read_parquet(path_prefix + splits["test"])
        # FIX: Added stratify to maintain class balance in small sample
        train_df, _ = train_test_split(train_df, train_size=0.2, stratify=train_df['profession'], random_state=42)
        # Validation is usually smaller, maybe keep 100% of validation or downsample carefully
        val_df, _ = train_test_split(val_df, train_size=0.5, stratify=val_df['profession'], random_state=42)
        X_train = train_df['hard_text'].tolist()
        X_val = val_df['hard_text'].tolist()
        # FIX: ENCODE LABELS (Strings -> Integers)
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(train_df['profession'])
        y_val = encoder.transform(val_df['profession'])
        # FIX: Calculate actual classes (28), do not return 2
        num_classes = len(encoder.classes_)
        print(f"Bios Loaded. Num Classes: {num_classes}")
        return X_train, y_train, X_val, y_val, num_classes
    except Exception as e:
        print(f"Error Bias in Bios: {e}")
        return [], [], [], [], 0
    


def load_sentiment_data():
    print("--- Loading Sentiment Data (SST-2) ---")
    try:
        dataset = load_dataset("glue", "sst2")
        train_df = dataset['train'].to_pandas()
        train_df, _ = train_test_split( train_df, train_size=0.2, stratify=train_df['label'], random_state=42 )
        val_data = dataset['validation']
        X_train = train_df['sentence'].tolist()
        y_train = train_df['label'].tolist()
        # Validation is small, keep as is
        X_val = val_data['sentence']
        y_val = val_data['label']
        return X_train, y_train, X_val, y_val, 2
    except Exception as e:
        print(f"Error loading SST-2: {e}")
        return [], [], [], [], 0


def get_retrain_data(adv_path, attack_path):
    """
    Loads RIFair adversarial data.
    Returns: List of texts, List of integer labels
    """
    try:
        with open(adv_path, 'rb') as f:
            retrain_data = pickle.load(f)
        with open(attack_path, 'rb') as f:
            attack_data = pickle.load(f)

        # Convert to DataFrame for easier handling
        df = pandas.DataFrame(retrain_data)

        retrain_text = []
        retrain_label = []

        # Iterate through adversarial results
        for _, row in df.iterrows():
            item_id = row["original_example_id"]
            # Ensure we get the ground truth label
            # Note: Ensure this label matches the format (int) of your clean data
            ground_truth = attack_data[item_id]["label"]
            # 1. Add the "Original" text (hard example)
            retrain_text.append(row["final_text_ori"])
            retrain_label.append(ground_truth)
            # 2. Add the "Adversarial/Similar" text
            retrain_text.append(row["final_text_sim"])
            retrain_label.append(ground_truth)

        print(f"Successfully loaded {len(retrain_text)} adversarial examples.")
        return retrain_text, retrain_label

    except Exception as e:
        print(f"Error loading adversarial data: {e}")
        return [], []


def create_adversarial_retraining_set(TB_adv_path, FB_adv_path, FF_adv_path, attack_path):
    """
    Combines Clean BiasInBios data with RIFair Adversarial data.
    Returns: HuggingFace Dataset object
    """

    # 1. Load Clean Data
    X_clean, y_clean, X_val, y_val, num_labels = load_bias_in_bios_data()

    # 2. Load Adversarial Data
    TB_X_adv, TB_y_adv = get_retrain_data(TB_adv_path, attack_path)
    FB_X_adv, FB_y_adv = get_retrain_data(FB_adv_path, attack_path)
    FF_X_adv, FF_y_adv = get_retrain_data(FF_adv_path, attack_path)


    # 3. Combine Lists
    # We convert y_clean (numpy array) to list to match y_adv
    x_combined = X_clean + TB_X_adv + FB_X_adv + FF_X_adv
    y_combined = list(y_clean) + TB_y_adv + FB_y_adv + FF_y_adv

    print(f"--- Combination Stats ---")
    print(f"Clean samples: {len(X_clean)}")
    print(f"Adversarial samples: {len(TB_X_adv)}")
    print(f"Total Retraining Size: {len(x_combined)}")
    return x_combined, y_combined, X_val, y_val, num_labels


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    DATASET_NAMES = [ "biasinbios"]
    BASE_OUTPUT_DIR = "/kaggle/working/models"  # Cleaner base dir
    # Standard models
    model_names = [ "bert-base-uncased", "roberta-base", "distilbert-base-uncased", "microsoft/deberta-v3-base" ]
    # --- TRAINING LOOP ---
    CURRENT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "biasinbios_retrain")
    for name in model_names:
        try:
            TB_adv_path=f"/kaggle/input/biasinbios/{name}_retrain_TB_attack_results.pkl"
            FB_adv_path=f"/kaggle/input/biasinbios/{name}_retrain_FB_attack_results.pkl"
            FF_adv_path=f"/kaggle/input/biasinbios/{name}_retrain_FF_attack_results.pkl"
            attack_path=f"/kaggle/input/biasinbios/attack_retrain_data.pkl"
            X_train, y_train, X_val, y_val, num_labels= create_adversarial_retraining_set(TB_adv_path, FB_adv_path, FF_adv_path, attack_path)

            train_transformer(name, X_train, y_train, X_val, y_val, num_labels, CURRENT_OUTPUT_DIR)
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
                
                

