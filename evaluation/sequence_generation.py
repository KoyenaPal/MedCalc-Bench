from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import re
import torch
import random
import numpy as np

# === CONFIGURATION ===
SEED = 42  # Or any integer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (may slow down and not support all ops)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === LOAD MODELS ===
def load_model_and_tokenizer(name, cache_dir="/workspace/hf"):
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir).to("cpu")
    model.eval()
    return tokenizer, model


# eval_tokenizers_models = [load_model_and_tokenizer(m) for m in evaluation_models]

# === GENERATE CANDIDATES ===


def truncate_to_first_sentence(text):
    """Truncate text to the first sentence (ending in . ! or ?)."""
    match = re.search(r'(.+?[.!?])(\s|$)', text.strip())
    return match.group(1).strip() if match else text.strip()

def generate_candidates(context, tokenizer, model, num_return_sequences, max_gen_tokens=20):
    set_seed(SEED)
    model = model.to(device)
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_gen_tokens,
            do_sample=True,
            top_k=50,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Truncate each generated continuation to the first sentence
    candidates = []
    for out in output:
        full_text = tokenizer.decode(out[input_ids.shape[-1]:], skip_special_tokens=True)
        first_sentence = truncate_to_first_sentence(full_text)
        candidates.append(first_sentence)
    return candidates


# === COMPUTE PERPLEXITY ===
def compute_perplexity(context, sentence, tokenizer, model, device="cuda"):
    try:
        full_text = context + sentence
        enc = tokenizer(full_text, return_tensors="pt", padding=True).to(device)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            model = model.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()
    except Exception as e:
        print(f"⚠️ Perplexity calculation failed: {e}")
        return None


# === MAIN LOOP ===
def iterative_generate(context, generation_models, gen_tokenizers_models, eval_tokenizers_models, max_total_tokens=100, num_candidates=5):
    while True:
        total_tokens = len(context.split())
        if total_tokens >= max_total_tokens:
            print("\n✅ Reached max length.")
            break

        all_candidates = []
        # Generate from both models
        for model_index, (tokenizer, model) in enumerate(gen_tokenizers_models):
            candidates = generate_candidates(context, tokenizer, model, num_candidates)
            print("MODEL", generation_models[model_index])
            print("CANDIDATES", candidates, flush=True)
            all_candidates.extend(candidates)
            model = model.to("cpu")
            torch.cuda.empty_cache()
            

        # Evaluate with all evaluation models
        scored_candidates = []
        for cand in all_candidates:
            perplexities = []
            for tokenizer, model in eval_tokenizers_models:
                try:
                    ppl = compute_perplexity(context, cand, tokenizer, model)
                    if ppl is not None and not torch.isnan(torch.tensor(ppl)):
                        perplexities.append(ppl)
                except Exception as e:
                    print(f"⚠️ Error scoring with model: {e}")
                    continue
            model = model.to("cpu")
            torch.cuda.empty_cache()
        
            if perplexities:
                avg_ppl = sum(perplexities) / len(perplexities)
                scored_candidates.append((cand, avg_ppl))
            else:
                print(f"⚠️ Skipping candidate (no valid perplexities): {cand}")
        # Select candidate with lowest avg perplexity
        best_candidate = min(scored_candidates, key=lambda x: x[1])[0]
        print(f"\n🔹 Selected: {best_candidate.strip()}")
        context += " " + best_candidate.strip()

    return context


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Context-aware sentence merging using perplexity.")
    # parser.add_argument("jsonl_files", nargs='+', help="Paths to input JSONL files.")
    parser.add_argument("--output", type=str, default="merged_output.csv", help="Path to save merged output.")
    parser.add_argument("--models", nargs='+', default=[
        "BytedTsinghua-SIA/DAPO-Qwen-32B",
        "Qwen/QwQ-32B"
    ], help="Hugging Face model names.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading models...")
    models = [
        (
            AutoModelForCausalLM.from_pretrained(name, cache_dir="/workspace/hf").to(device).eval(),
            AutoTokenizer.from_pretrained(name, cache_dir="/workspace/hf")
        )
        for name in args.models
    ]
    jsonl_files = ["MedCalc-Bench/evaluation/outputs/BytedTsinghua-SIA_DAPO-Qwen-32B_zero_shot.jsonl", "MedCalc-Bench/evaluation/outputs/Qwen_QwQ-32B_zero_shot.jsonl"]
    print("Loading input JSONL files...")
    paragraphs_per_file = load_paragraphs_from_jsonl_files(jsonl_files)

    print("Merging paragraphs (rows)...")
    merged_paragraphs = []
    
    df = pd.read_csv("../dataset/test_data.csv")
    df = df.sample(n=25, random_state=42)

    for index in tqdm.tqdm(range(len(df))):

        row = df.iloc[index]

        patient_note = row["Patient Note"]
        question = row["Question"] 
        calculator_id = str(row["Calculator ID"])
        note_id = str(row["Note ID"])
        system, user = zero_shot(patient_note, question)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    # for row_idx in range(26):
        
        row_paragraphs = [
            file_paragraphs[row_idx]
            for file_paragraphs in paragraphs_per_file
            if len(file_paragraphs) > row_idx
        ]
        if not row_paragraphs:
            merged_paragraphs.append("")
            continue

        merged = get_best_sentence_per_position_with_context(messages, row_paragraphs, models, device)
        merged_paragraphs.append(merged)

    print(f"Writing merged paragraphs to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for paragraph in merged_paragraphs:
            f.write(paragraph.strip() + "\n")

    print("✅ Done.")

if __name__ == "__main__":
    set_seed(SEED)
    generation_models = evaluation_models = ["Qwen/QwQ-32B", "BytedTsinghua-SIA/DAPO-Qwen-32B"]
    gen_tokenizers_models = eval_tokenizers_models = [load_model_and_tokenizer(m) for m in generation_models]
    # evaluation_models = ["Qwen/QwQ-32B", "BytedTsinghua-SIA/DAPO-Qwen-32B"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # === RUN ===
    initial_context = "Once upon a time"
    final_output = iterative_generate(initial_context, generation_models, gen_tokenizers_models, eval_tokenizers_models)
    print("\nFinal output:\n", final_output)
    # main()
