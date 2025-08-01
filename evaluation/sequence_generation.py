from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import re
import torch
import random
import numpy as np
import os
import csv
import tqdm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
from run import zero_shot
import gc
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

set_seed(SEED)

# def load_to_gpu(model):
#     model.to("cuda")
#     torch.cuda.empty_cache()

# def unload_to_cpu(model):
#     model.to("cpu")
#     torch.cuda.empty_cache()


# === LOAD MODELS ===
def load_model_and_tokenizer(name, cache_dir="/workspace/hf"):
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()
    return tokenizer, model


# eval_tokenizers_models = [load_model_and_tokenizer(m) for m in evaluation_models]

# === GENERATE CANDIDATES ===


def truncate_to_first_sentence(text):
    """Truncate text to the first sentence (ending in . ! or ?)."""
    match = re.search(r'(.+?[.!?])(\s|$)', text.strip())
    return match.group(1).strip() if match else text.strip()

def generate_candidates(context, tokenizer, model, num_return_sequences, max_gen_tokens=40):
    set_seed(SEED)
    #model = model.to(device)
    #input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    input_ids = tokenizer.encode(context, return_tensors="pt")
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)

    with torch.no_grad():
        # with torch.cuda.amp.autocast(enabled=False):
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
# def compute_perplexity(context, sentence, tokenizer, model, device="cuda"):
#     try:
#         full_text = context + sentence
#         # enc = tokenizer(full_text, return_tensors="pt", padding=True).to(device)
#         # input_ids = enc["input_ids"].to(device)
#         # attention_mask = enc["attention_mask"].to(device)
#         enc = tokenizer(full_text, return_tensors="pt", padding=True)
#         input_ids = enc["input_ids"]
#         model_device = next(model.parameters()).device
#         input_ids = input_ids.to(model_device)
#         attention_mask = enc["attention_mask"].to(model_device)
#         if attention_mask is not None:
#             attention_mask = attention_mask.to(model_device)

#         with torch.no_grad():
#             # with torch.cuda.amp.autocast(enabled=False):
#             # model = model.to(device)
#             outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
#             loss = outputs.loss
#         return torch.exp(loss).item()
#     except Exception as e:
#         print(f"⚠️ Perplexity calculation failed: {e}")
#         return None

def compute_perplexity(context, sentence, tokenizer, model):
    try:
        full_text = context + sentence
        enc = tokenizer(full_text, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(**enc, labels=enc["input_ids"])
            loss = outputs.loss
        return torch.exp(loss).item()
    except Exception as e:
        print(f"⚠️ Perplexity calc failed: {e}")
        return None



# === MAIN LOOP ===
# assuming that gen and eval models are the same for now -- need to change context tokenizer otherwise
def iterative_generate(context, generation_models, gen_tokenizers_models, eval_tokenizers_models, max_total_tokens=100, num_candidates=5):
    iteration_count = 0
    contexts = {}
    total_tokens = 0
    while True:
        if iteration_count > 0:
            if total_tokens >= max_total_tokens:
                print("\n✅ Reached max length.")
                break
            if "</think>" in contexts[0]:
                print("\n✅ Reached end of thought.")
                break
        all_candidates = []
        # Generate from both models
        for model_index, (tokenizer, model) in enumerate(gen_tokenizers_models):
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if iteration_count == 0:
                context_in_chat_temp = tokenizer.apply_chat_template(context,tokenize=False, add_generation_prompt=True)
                max_total_tokens = len(tokenizer.encode(context_in_chat_temp, add_special_tokens=True)) + 4096
                contexts[model_index] = context_in_chat_temp
                total_tokens = len(tokenizer.encode(contexts[model_index]))
            candidates = generate_candidates(contexts[model_index], tokenizer, model, num_candidates)
            print("MODEL", generation_models[model_index])
            print("CANDIDATES", candidates, flush=True)
            all_candidates.extend(candidates)
            # model = model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            

        # Evaluate with all evaluation models
        scored_candidates = []
        for cand in all_candidates:
            perplexities = []
            for model_index, (tokenizer, model) in enumerate(eval_tokenizers_models):
                try:
                    # load_to_gpu(model)
                    ppl = compute_perplexity(contexts[model_index], cand, tokenizer, model)
                    # unload_to_cpu(model)
                    if ppl is not None and not torch.isnan(torch.tensor(ppl)):
                        perplexities.append(ppl)
                except Exception as e:
                    print(f"⚠️ Error scoring with model: {e}")
                    continue
            # unload_to_cpu(model)
        
            if perplexities:
                avg_ppl = sum(perplexities) / len(perplexities)
                scored_candidates.append((cand, avg_ppl))
            else:
                print(f"⚠️ Skipping candidate (no valid perplexities): {cand}")
        # Select candidate with lowest avg perplexity
        best_candidate = min(scored_candidates, key=lambda x: x[1])[0]
        print(f"\n🔹 Selected: {best_candidate.strip()}")
        for key, curr_cont in contexts.items():
            contexts[key] = curr_cont + " " + best_candidate.strip()
        iteration_count += 1

    return contexts


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Context-aware sentence merging using perplexity.")
    # parser.add_argument("jsonl_files", nargs='+', help="Paths to input JSONL files.")
    parser.add_argument("--output", type=str, default="ensemble_outputs/merged_output.csv", help="Path to save merged output.")
    parser.add_argument("--models", nargs='+', default=[
        "BytedTsinghua-SIA/DAPO-Qwen-32B",
        "Qwen/QwQ-32B"
    ], help="Hugging Face model names.")
    set_seed(SEED)
    args = parser.parse_args()
    device = 'auto'
    generation_models = evaluation_models = args.models
    gen_tokenizers_models = eval_tokenizers_models = [load_model_and_tokenizer(m) for m in generation_models]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_header = not os.path.isfile(args.output)
    df = pd.read_csv("../dataset/test_data.csv")
    df = df.sample(n=25, random_state=42)

    with open(args.output, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Row Number", "Note ID", "Calculator ID", "Question", "Patient Note", "Ensembled Thought"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    
        if write_header:
            writer.writeheader()
        for index in tqdm.tqdm(range(len(df))):
            row = df.iloc[index]
    
            patient_note = row["Patient Note"]
            question = row["Question"]
            calculator_id = str(row["Calculator ID"])
            note_id = str(row["Note ID"])
    
            # Create messages
            system, user = zero_shot(patient_note, question)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
    
            # Generate output
            final_output = iterative_generate(
                messages,
                generation_models,
                gen_tokenizers_models,
                eval_tokenizers_models
            )
    
            # Write a single row to the CSV
            writer.writerow({
                "Row Number": int(row["Row Number"]),
                "Note ID": note_id,
                "Calculator ID": calculator_id,
                "Question": question,
                "Patient Note": patient_note,
                "Ensembled Thought": final_output[0].strip()
            })
            csvfile.flush()
            # after writing the row:
            gc.collect()
            torch.cuda.empty_cache()
    
            print(f"✅ Done. Row-wise output saved to: {args.output}")

if __name__ == "__main__":
    set_seed(SEED)
    device = "auto"
    # === RUN ===
    main()
