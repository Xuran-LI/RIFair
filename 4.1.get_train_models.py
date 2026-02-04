# !pip install -U transformers datasets torch pandas numpy scikit-learn accelerate sentencepiece protobuf pyarrow
import os
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


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # DATASET_NAMES = ["SENTIMENT", "JIGSAW", "BIOS"]
    DATASET_NAMES = [ "JIGSAW", "BIOS"]
    BASE_OUTPUT_DIR = "/kaggle/working/models"  # Cleaner base dir

    # Standard models
    model_names = [ "bert-base-uncased", "roberta-base", "distilbert-base-uncased", "microsoft/deberta-v3-base" ]

    for DATASET_NAME in DATASET_NAMES:
        # --- DATA LOADING ---
        X_train, y_train, X_val, y_val, num_labels = [], [], [], [], 0

        # FIX: Reset Output Directory for each dataset loop
        if DATASET_NAME == "JIGSAW":
            X_train, y_train, X_val, y_val, num_labels = load_jigsaw_data()
            CURRENT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "jigsaw")

        elif DATASET_NAME == "BIOS":
            X_train, y_train, X_val, y_val, num_labels = load_bias_in_bios_data()
            CURRENT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "biasinbios")

        elif DATASET_NAME == "SENTIMENT":
            X_train, y_train, X_val, y_val, num_labels = load_sentiment_data()
            CURRENT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "sentiment")

        # --- VALIDATION ---
        if len(X_train) == 0:
            print(f"❌ Data loading failed for {DATASET_NAME}. Skipping.")
            continue

        print(f"Ready to train on {DATASET_NAME}.")
        print(f"Train Size: {len(X_train)} | Val Size: {len(X_val)} | Num Labels: {num_labels}")

        # --- TRAINING LOOP ---
        for name in model_names:
            try:
                train_transformer(name, X_train, y_train, X_val, y_val, num_labels, CURRENT_OUTPUT_DIR)
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                
                

