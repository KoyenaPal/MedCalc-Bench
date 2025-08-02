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
import time
import copy
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
    
    # Tokenize with attention_mask
    enc = tokenizer(context, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # ✅ this is the fix
            max_new_tokens=max_gen_tokens,
            do_sample=True,
            top_k=50,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )

    candidates = []
    for out in output:
        full_text = tokenizer.decode(out[input_ids.shape[-1]:], skip_special_tokens=True)
        first_sentence = truncate_to_first_sentence(full_text)
        candidates.append(first_sentence)
    return candidates


# === COMPUTE PERPLEXITY ===
# def compute_perplexity(context, sentence, tokenizer, model):
#     try:
#         full_text = context + sentence
#         enc = tokenizer(full_text, return_tensors="pt", padding=True).to(model.device)
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             outputs = model(**enc, labels=enc["input_ids"])
#             loss = outputs.loss
#         return torch.exp(loss).item()
#     except Exception as e:
#         print(f"⚠️ Perplexity calc failed: {e}")
#         return None



# === MAIN LOOP ===
# assuming that gen and eval models are the same for now -- need to change context tokenizer otherwise
# OLD
# def iterative_generate(context, generation_models, gen_tokenizers_models, eval_tokenizers_models, max_total_tokens=100, num_candidates=5):
#     iteration_count = 0
#     contexts = {}
#     total_tokens = 0
#     while True:
#         if iteration_count > 0:
#             if total_tokens >= max_total_tokens:
#                 print("\n✅ Reached max length.")
#                 break
#             if "</think>" in contexts[0]:
#                 print("\n✅ Reached end of thought.")
#                 break
#         all_candidates = []
#         # Generate from both models
#         for model_index, (tokenizer, model) in enumerate(gen_tokenizers_models):
#             if tokenizer.pad_token is None:
#                 tokenizer.pad_token = tokenizer.eos_token
#             if iteration_count == 0:
#                 context_in_chat_temp = tokenizer.apply_chat_template(context,tokenize=False, add_generation_prompt=True)
#                 context_in_chat_temp = context_in_chat_temp + "<think>"
#                 max_total_tokens = len(tokenizer.encode(context_in_chat_temp, add_special_tokens=True)) + 4096
#                 contexts[model_index] = context_in_chat_temp
#                 total_tokens = len(tokenizer.encode(contexts[model_index]))
#             candidates = generate_candidates(contexts[model_index], tokenizer, model, num_candidates)
#             print("MODEL", generation_models[model_index])
#             print("CANDIDATES", candidates, flush=True)
#             all_candidates.extend(candidates)
#             # model = model.to("cpu")
#             gc.collect()
#             torch.cuda.empty_cache()
            

#         # Evaluate with all evaluation models
#         scored_candidates = []
#         for cand in all_candidates:
#             perplexities = []
#             for model_index, (tokenizer, model) in enumerate(eval_tokenizers_models):
#                 try:
#                     # load_to_gpu(model)
#                     ppl = compute_perplexity(contexts[model_index], cand, tokenizer, model)
#                     # unload_to_cpu(model)
#                     if ppl is not None and not torch.isnan(torch.tensor(ppl)):
#                         perplexities.append(ppl)
#                 except Exception as e:
#                     print(f"⚠️ Error scoring with model: {e}")
#                     continue
#             # unload_to_cpu(model)
        
#             if perplexities:
#                 avg_ppl = sum(perplexities) / len(perplexities)
#                 scored_candidates.append((cand, avg_ppl))
#             else:
#                 print(f"⚠️ Skipping candidate (no valid perplexities): {cand}")
#         # Select candidate with lowest avg perplexity
#         best_candidate = min(scored_candidates, key=lambda x: x[1])[0]
#         print(f"\n🔹 Selected: {best_candidate.strip()}")
#         for key, curr_cont in contexts.items():
#             contexts[key] = curr_cont + " " + best_candidate.strip()
#         iteration_count += 1

#     return contexts

def compute_perplexities(context, candidates, tokenizer, model):
    """
    Compute perplexities for candidate completions given a shared context,
    using batching and masking only the candidate portion for loss.
    """
    import torch
    import torch.nn.functional as F

    model_device = next(model.parameters()).device

    # Tokenize context once to get tokenized length
    context_ids = tokenizer(context, return_tensors="pt")["input_ids"][0]
    context_len = len(context_ids)

    # Create input_ids and masks for all context+candidate combinations
    full_texts = [context + candidate for candidate in candidates]
    encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    )

    input_ids = encodings["input_ids"].to(model_device)
    attention_mask = encodings["attention_mask"].to(model_device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Compute token-wise cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            reduction='none'
        ).view(input_ids.size())

    # Compute where the candidate tokens start for each sequence
    loss_mask = torch.zeros_like(input_ids)
    for i, full_input in enumerate(full_texts):
        # Tokenize context + candidate and just context to get lengths
        full_ids = tokenizer(full_input, return_tensors="pt")["input_ids"][0]
        candidate = candidates[i]
        candidate_ids = tokenizer(candidate, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        start = len(full_ids) - len(candidate_ids)
        end = start + len(candidate_ids)
        loss_mask[i, start:end] = 1

    loss = loss * loss_mask  # apply candidate-only mask
    token_lengths = loss_mask.sum(dim=1)
    loss_per_seq = loss.sum(dim=1) / token_lengths
    perplexities = torch.exp(loss_per_seq).tolist()

    return perplexities



# def compute_perplexities(context, candidates, tokenizer, model):
#     """
#     Compute perplexity for each candidate in a batch given a shared context.

#     Args:
#         context (str): The shared prefix text.
#         candidates (list[str]): List of candidate completions.
#         tokenizer: Hugging Face tokenizer.
#         model: Hugging Face language model (e.g., GPT2LMHeadModel).

#     Returns:
#         list[float]: Perplexities for each candidate sequence.
#     """
#     model_device = next(model.parameters()).device

#     # Combine context + each candidate into full sequences
#     full_texts = [context + c for c in candidates]

#     # Tokenize batch, pad to max length, return tensors on CPU
#     enc = tokenizer(
#         full_texts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         return_attention_mask=True
#     )

#     input_ids = enc["input_ids"].to(model_device)
#     attention_mask = enc["attention_mask"].to(model_device)

#     with torch.no_grad():
#         # Forward pass with labels to get logits and loss
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=input_ids  # model handles shifting internally
#         )

#         # Compute per-token cross-entropy loss (no reduction)
#         losses = F.cross_entropy(
#             outputs.logits.view(-1, outputs.logits.size(-1)),
#             input_ids.view(-1),
#             reduction='none'
#         ).view(input_ids.shape)

#         # Mask out padding tokens in the loss
#         tokenwise_loss = losses * attention_mask

#         # Sum losses over tokens per sequence and normalize by number of valid tokens
#         seq_lens = attention_mask.sum(dim=1)
#         loss_per_seq = tokenwise_loss.sum(dim=1) / seq_lens

#         # Exponentiate average loss to get perplexity per sequence
#         perplexities = torch.exp(loss_per_seq).tolist()

#     return perplexities



def iterative_generate(context, generation_models, gen_tokenizers_models, eval_tokenizers_models, max_total_tokens=100, num_candidates=5):
    iteration_count = 0
    contexts = {}
    total_tokens = 0
    chosen_per_iteration = []

    while True:
        start_time = time.time()

        if iteration_count > 0:
            if total_tokens >= max_total_tokens:
                print("\n✅ Reached max length.")
                break
            if "</think>" in contexts[0]:
                print("\n✅ Reached end of thought.")
                break

        all_candidates = []
        candidates_per_model = {}

        for model_index, (tokenizer, model) in enumerate(gen_tokenizers_models):
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if iteration_count == 0:
                context_in_chat_temp = tokenizer.apply_chat_template(
                    context, tokenize=False, add_generation_prompt=True
                ) + "<think>"
                max_total_tokens = len(tokenizer.encode(context_in_chat_temp, add_special_tokens=True)) + 4096
                contexts[model_index] = context_in_chat_temp
                total_tokens = len(tokenizer.encode(contexts[model_index]))

            candidates = generate_candidates(
                contexts[model_index], tokenizer, model, num_candidates
            )
            print(f"\nMODEL: {generation_models[model_index]}")
            print("CANDIDATES:", candidates)

            all_candidates.extend(candidates)
            candidates_per_model[model_index] = candidates

        # === Batched Perplexity Evaluation ===
        scored_candidates_dict = {cand: [] for cand in all_candidates}
        for model_index, (tokenizer, model) in enumerate(eval_tokenizers_models):
            try:
                ppls = compute_perplexities(contexts[model_index], all_candidates, tokenizer, model)
                for i, cand in enumerate(all_candidates):
                    scored_candidates_dict[cand].append(ppls[i])
            except Exception as e:
                print(f"⚠️ Error in batched scoring: {e}")
                continue

        # === Aggregate scores ===
        scored_candidates = []
        for cand, ppl_list in scored_candidates_dict.items():
            if ppl_list:
                avg_ppl = sum(ppl_list) / len(ppl_list)
                scored_candidates.append((cand, avg_ppl))

        if not scored_candidates:
            print("❌ No valid candidates scored. Exiting.")
            break

        best_candidate, best_score = min(scored_candidates, key=lambda x: x[1])
        print(f"\n🔹 Selected: {best_candidate.strip()}")

        chosen_model_index = None
        for model_index, cands in candidates_per_model.items():
            if best_candidate in cands:
                chosen_model_index = model_index
                break

        chosen_per_iteration.append({
            "iteration": iteration_count,
            "selected_candidate": best_candidate,
            "selected_model_index": chosen_model_index,
            "all_candidates": copy.deepcopy(candidates_per_model),
            "score": best_score,
        })

        # Append selected to each context
        for key in contexts:
            contexts[key] += " " + best_candidate.strip()

        iteration_count += 1
        elapsed = time.time() - start_time
        print(f"⏱️ Iteration {iteration_count} completed in {elapsed:.2f} seconds.")

    return contexts, chosen_per_iteration



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
            final_output, selected_distributions = iterative_generate(
                messages,
                generation_models,
                gen_tokenizers_models,
                eval_tokenizers_models
            )
            print(f"🔍 Iteration-wise Selection Summary:")
            for d in selected_distributions:
                print(f"Iteration {d['iteration']}: Model {d['selected_model_index']} -> {d['selected_candidate']}")
            with open(f"ensemble_outputs/selected_distributions_row_{index}.json", "w") as f:
                json.dump(selected_distributions, f, indent=2)
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
