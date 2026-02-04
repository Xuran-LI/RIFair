import pandas
import pandas as pd
import pickle
import collections
import string
import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded (run once)
nltk.download('stopwords')
nltk.download('punkt')


def get_high_freq_tokens(file_path, save_path, column='text_sim', top_n=3000):
    """
    Loads pickle data and returns the most common tokens.
    """
    # 1. Load Data
    try:
        # Try loading as a list of dicts (standard pickle)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # Convert to DataFrame for easier handling
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data  # It's already a DataFrame
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    print(f"Loaded {len(df)} examples.")

    all_text = " ".join(df[column].astype(str).tolist())
    all_text = all_text.lower()
    translator = str.maketrans('', '', string.punctuation)
    all_text = all_text.translate(translator)
    tokens = nltk.word_tokenize(all_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    counter = collections.Counter(filtered_tokens)
    print(f"\n--- Top {top_n} High-Frequency Tokens in '{column}' ---")
    most_common = counter.most_common(top_n)
    df_tokens = pd.DataFrame(most_common, columns=['Token', 'Frequency'])
    df_tokens.to_csv(save_path, index=False)
    print(f"\nSuccessfully saved top {top_n} tokens to: {save_path}")



# --- USAGE ---
if __name__ == "__main__":
    get_high_freq_tokens( "jiasaw_data/attack_data.pkl", "jiasaw_data/attack_data_tokens.csv", column='text_sim', top_n=1000)
    get_high_freq_tokens( "biasinbios_data/attack_data.pkl", "biasinbios_data/attack_data_tokens.csv", column='text_sim', top_n=1000)
    get_high_freq_tokens( "Sentiment_data/attack_data.pkl", "Sentiment_data/attack_data_tokens.csv", column='text_sim', top_n=1000)
    get_high_freq_tokens("adultnew_data/attack_data.pkl", "adultnew_data/attack_data_tokens.csv", column='text_sim', top_n=1000)
    get_high_freq_tokens("biasinbios_data/attack_retrain_data.pkl", "biasinbios_data/attack_retrain_data_tokens.csv", column='text_sim', top_n=1000)









