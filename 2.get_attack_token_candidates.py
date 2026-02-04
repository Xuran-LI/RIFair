# Install the latest versions to fix the 'is_offline_mode' error
# !pip install -U transformers huggingface_hub accelerate bitsandbytes peft sentence-transformers pandas tqdm
import torch
import json
import pickle
import textwrap
import pandas as pd
from tqdm import tqdm  # Added for progress bar
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig


def generate_candidates_qwen(attack_tokens, save_file, pipe):
    """
    Generates substitution candidates using Qwen2.5.
    """
    system_prompt = textwrap.dedent("""You are an expert Computational Linguist helping to stress-test an auditing system. Your goal is to provide exactly semantic equivalent substitutions.
    Guidelines: 
    1. Valid Semantic Equivalents: Provide strict synonyms that preserve the exact truth conditions of the context.
    2. Output Format: Return ONLY a raw JSON list of strings. Do not use Markdown formatting or explanations.
    Example Output: ["Physician", "Medical Practitioner", "MD", "Medical Doctor"]""").strip()

    token_replacements = {}

    # Using tqdm for a progress bar
    for token in tqdm(attack_tokens, desc="Generating Synonyms"):
        # print(f"Processing: {token}...") # tqdm handles this

        # Fixed syntax error in f-string (added closing quote for token)
        user_msg = textwrap.dedent(f"""Task: Generate 10-20 strict Synonyms for the expression: "{token}". 
        Remember: Return ONLY a raw JSON list.""").strip()
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Generate
        outputs = pipe(prompt, return_full_text=False)
        generated_text = outputs[0]["generated_text"]
        try:
            # parsing logic
            start = generated_text.find('[')
            end = generated_text.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = generated_text[start:end]
                candidates = json.loads(json_str)
                token_replacements[token] = candidates
                # print(f"  -> Generated {len(candidates)} items.")
            else:
                print(f"  -> No JSON found for '{token}'")
                token_replacements[token] = []
        except Exception as e:
            print(f"  -> Error parsing '{token}': {e}")
            token_replacements[token] = []

    # Save results
    print(f"Saving to {save_file}...")
    with open(save_file, 'wb') as f:
        pickle.dump(token_replacements, f)

    return token_replacements


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16)
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True)

    # 2. Load Data
    csv_path = "/kaggle/input/biasinbios/attack_data_tokens.csv"
    output_path = "/kaggle/working/biasinbios_attack_replacements.pkl"

    try:
        attack_tokens_df = pd.read_csv(csv_path)
        if 'Token' in attack_tokens_df.columns:
            token_list = attack_tokens_df['Token'].astype(str).tolist()
        else:
            print("Warning: Column 'Token' not found. Using the first column.")
            token_list = attack_tokens_df.iloc[:, 0].astype(str).tolist()
        generate_candidates_qwen(token_list, output_path, pipe)

        print("Finished.")

    except FileNotFoundError:
        print(f"Error: Could not find file at {csv_path}")

    csv_path = "/kaggle/input/jiasaw/attack_data_tokens.csv"
    output_path = "/kaggle/working/jiasaw_attack_replacements.pkl"

    try:
        attack_tokens_df = pd.read_csv(csv_path)
        if 'Token' in attack_tokens_df.columns:
            token_list = attack_tokens_df['Token'].astype(str).tolist()
        else:
            print("Warning: Column 'Token' not found. Using the first column.")
            token_list = attack_tokens_df.iloc[:, 0].astype(str).tolist()
        generate_candidates_qwen(token_list, output_path, pipe)

        print("Finished.")

    except FileNotFoundError:
        print(f"Error: Could not find file at {csv_path}")

    sv_path = "/kaggle/input/sentiment/attack_data_tokens.csv"
    output_path = "/kaggle/working/sentiment_attack_replacements.pkl"

    try:
        attack_tokens_df = pd.read_csv(csv_path)
        if 'Token' in attack_tokens_df.columns:
            token_list = attack_tokens_df['Token'].astype(str).tolist()
        else:
            print("Warning: Column 'Token' not found. Using the first column.")
            token_list = attack_tokens_df.iloc[:, 0].astype(str).tolist()
            generate_candidates_qwen(token_list, output_path, pipe)

        print("Finished.")

    except FileNotFoundError:
            print(f"Error: Could not find file at {csv_path}")

    csv_path = "/kaggle/input/adultnew/attack_data_tokens.csv"
    output_path = "/kaggle/working/adultnew_attack_replacements.pkl"

    try:
        attack_tokens_df = pd.read_csv(csv_path)
        if 'Token' in attack_tokens_df.columns:
            token_list = attack_tokens_df['Token'].astype(str).tolist()
        else:
            print("Warning: Column 'Token' not found. Using the first column.")
            token_list = attack_tokens_df.iloc[:, 0].astype(str).tolist()
        generate_candidates_qwen(token_list, output_path, pipe)

        print("Finished.")

    except FileNotFoundError:
        print(f"Error: Could not find file at {csv_path}")



