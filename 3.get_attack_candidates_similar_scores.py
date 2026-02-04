# pip install -U sentence-transformers scipy
# 计算属性替换词的cosine相似度，形式化语义相似度
import pickle
from scipy.special import softmax
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder



def get_bidirectional_entailment(nli_model, original_text, candidate_text):
    """
    Checks if A -> B AND B -> A (Strict Equivalence).
    """
    # Prepare pairs: (Original, Candidate) and (Candidate, Original)
    pairs = [(original_text, candidate_text), (candidate_text, original_text)]
    logits = nli_model.predict(pairs)
    probs = softmax(logits, axis=1)
    if hasattr(nli_model, 'model'):
        config = nli_model.model.config
    else:
        config = nli_model.config  # Fallback if it's a raw HF model

    entail_idx = config.label2id.get('entailment')
    if entail_idx is None:
        entail_idx = 1
    forward_prob = probs[0][entail_idx]  # Probability that Original implies Candidate
    backward_prob = probs[1][entail_idx]  # Probability that Candidate implies Original
    strict_score = forward_prob * backward_prob

    return strict_score




def get_cosine_similarity(vector_model, original_text, candidate_text):
    """
    Calculates cosine similarity between embeddings.
    """
    embeddings = vector_model.encode([original_text, candidate_text])
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return score


if __name__ == "__main__":
    nli_model = CrossEncoder('MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
    vector_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    try:
        with open("/kaggle/input/biasinbios/biasinbios_attack_replacements.pkl", 'rb') as f:
            qwen_attack_dictionary = pickle.load(f)
        result_entry = (("feature_value", "candidate"), "cosine", "semantic")
        results_list = []
        ori_tokens = list(qwen_attack_dictionary.keys())
        for ori_t in ori_tokens:
            print(f"""{ori_t}""")
            cand_token = qwen_attack_dictionary[ori_t]
            for cnd_t in cand_token:
                similar1 = get_cosine_similarity(vector_model, ori_t, cnd_t)
                similar2 = get_bidirectional_entailment(nli_model, ori_t, cnd_t)
                result_entry = ((ori_t, cnd_t), similar1, similar2)
                results_list.append(result_entry)
        with open("/kaggle/working/biasinbios_attack_similarity.pkl", 'wb') as f:
            pickle.dump(results_list, f)

    except FileNotFoundError:
        print("Error: Input file 'qwen_adult_feature_substitutions.pkl' not found. Check your Kaggle dataset path.")

    try:
        with open("/kaggle/input/jiasaw/jiasaw_attack_replacements.pkl", 'rb') as f:
            qwen_attack_dictionary = pickle.load(f)
        result_entry = (("feature_value", "candidate"), "cosine", "semantic")
        results_list = []
        ori_tokens = list(qwen_attack_dictionary.keys())
        for ori_t in ori_tokens:
            print(f"""{ori_t}""")
            cand_token = qwen_attack_dictionary[ori_t]
            for cnd_t in cand_token:
                similar1 = get_cosine_similarity(vector_model, ori_t, cnd_t)
                similar2 = get_bidirectional_entailment(nli_model, ori_t, cnd_t)
                result_entry = ((ori_t, cnd_t), similar1, similar2)
                results_list.append(result_entry)
        with open("/kaggle/working/jiasaw_attack_similarity.pkl", 'wb') as f:
            pickle.dump(results_list, f)

    except FileNotFoundError:
        print("Error: Input file 'qwen_adult_feature_substitutions.pkl' not found. Check your Kaggle dataset path.")

    try:
        with open("/kaggle/input/sentiment/sentiment_attack_replacements.pkl", 'rb') as f:
            qwen_attack_dictionary = pickle.load(f)
        result_entry = (("feature_value", "candidate"), "cosine", "semantic")
        results_list = []
        ori_tokens = list(qwen_attack_dictionary.keys())
        for ori_t in ori_tokens:
            print(f"""{ori_t}""")
            cand_token = qwen_attack_dictionary[ori_t]
            for cnd_t in cand_token:
                similar1 = get_cosine_similarity(vector_model, ori_t, cnd_t)
                similar2 = get_bidirectional_entailment(nli_model, ori_t, cnd_t)
                result_entry = ((ori_t, cnd_t), similar1, similar2)
                results_list.append(result_entry)
        with open("/kaggle/working/sentiment_attack_similarity.pkl", 'wb') as f:
            pickle.dump(results_list, f)

    except FileNotFoundError:
        print("Error: Input file 'qwen_adult_feature_substitutions.pkl' not found. Check your Kaggle dataset path.")


    try:
        with open("/kaggle/input/adult_new/adult_new_attack_replacements.pkl", 'rb') as f:
            qwen_attack_dictionary = pickle.load(f)
        result_entry = (("feature_value", "candidate"), "cosine", "semantic")
        results_list = []
        ori_tokens = list(qwen_attack_dictionary.keys())
        for ori_t in ori_tokens:
            print(f"""{ori_t}""")
            cand_token = qwen_attack_dictionary[ori_t]
            for cnd_t in cand_token:
                similar1 = get_cosine_similarity(vector_model, ori_t, cnd_t)
                similar2 = get_bidirectional_entailment(nli_model, ori_t, cnd_t)
                result_entry = ((ori_t, cnd_t), similar1, similar2)
                results_list.append(result_entry)
        with open("/kaggle/working/adult_new_attack_similarity.pkl", 'wb') as f:
            pickle.dump(results_list, f)

    except FileNotFoundError:
        print("Error: Input file 'qwen_adult_feature_substitutions.pkl' not found. Check your Kaggle dataset path.")


