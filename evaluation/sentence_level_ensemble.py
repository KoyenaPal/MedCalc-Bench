import json
import re
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------- Load JSONL Files --------

def load_paragraphs_from_jsonl_files(filepaths: List[str]) -> List[List[str]]:
    all_paragraphs = []
    for path in filepaths:
        paragraphs = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    obj = json.loads(line)
                    paragraph = obj.get("LLM Thinking", "").strip()
                    if paragraph:
                        paragraphs.append(paragraph)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {i} in {path}: {e}")
        all_paragraphs.append(paragraphs)
    return all_paragraphs

# -------- Utility Functions --------

def split_into_sentences(paragraph: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', paragraph.strip())

@torch.no_grad()
def get_perplexity_full_text(text: str, model, tokenizer, device='cpu') -> float:
    print("IN GET PERPLEXTIY FULL TEXT", flush=True)
    print(text, flush=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

def get_best_sentence_per_position_with_context(
    messages,
    paragraphs,
    models,
    device='cpu'
) -> str:
    sentence_lists = [split_into_sentences(p) for p in paragraphs]
    max_len = max(len(s) for s in sentence_lists)
    selected_sentences = []

    for i in range(max_len):
        candidates = [s[i] for s in sentence_lists if i < len(s)]
        if not candidates:
            continue

        full_contexts = []
        for cand in candidates:
            combined_text = " ".join(selected_sentences + [cand]).strip()

            # Use the tokenizer from the *first* model to apply the chat template
            tokenizer = models[0][1]
            try:
                chat_input = tokenizer.apply_chat_template(
                    messages,tokenize=False, add_generation_prompt=True)
                chat_input = chat_input + combined_text
            except Exception as e:
                raise ValueError("Tokenizer does not support chat templates or failed to apply it.") from e
            full_contexts.append(chat_input)

        all_perplexities = []
        for model, tokenizer in models:
            perps = [get_perplexity_full_text(ctx, model, tokenizer, device) for ctx in full_contexts]
            all_perplexities.append(perps)

        best_index = min(
            range(len(candidates)),
            key=lambda j: min(all_perplexities[m][j] for m in range(len(models)))
        )

        selected_sentences.append(candidates[best_index])

    return " ".join(selected_sentences)


def zero_shot(note, question):
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}.'
    user_temp = f'Here is the patient note:\n{note}\n\nHere is the task:\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:'
    return system_msg, user_temp
# -------- Main --------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Context-aware sentence merging using perplexity.")
    # parser.add_argument("jsonl_files", nargs='+', help="Paths to input JSONL files.")
    parser.add_argument("--output", type=str, default="merged_output.txt", help="Path to save merged output.")
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
    main()
