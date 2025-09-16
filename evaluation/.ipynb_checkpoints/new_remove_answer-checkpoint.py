import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import nltk
import re
import argparse
import random
import numpy as np

nltk.download("punkt")
from nltk.tokenize import sent_tokenize
# from accelerate import Accelerator

# accelerator = Accelerator(mixed_precision="bf16")
# -----------------------
# Argument Parser
# -----------------------

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser(description="Process input file and generate output without answers.")
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to the input file (.csv or .jsonl)"
)
parser.add_argument(
    "--output_file",
    type=str,
    default=None,
    help="Path for the output file (optional). If not provided, a default path will be used."
)

args = parser.parse_args()
input_file = args.input_file
input_basename, input_ext = os.path.splitext(os.path.basename(input_file))
input_ext = input_ext.lower()

# Validate input extension
if input_ext not in (".jsonl", ".csv"):
    raise ValueError(f"Unsupported file type: {input_ext}")

# Determine output file
if args.output_file:
    output_file = args.output_file
else:
    output_folder = "without_answer"
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
    output_file = os.path.join(output_folder, f"{input_basename}_without_answer{input_ext}")

print(f"Input file: {input_file}")
print(f"Output file: {output_file}")


def extract_think_text(text):
    match = re.search(r"<think>(.*?)(</think>|$)", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[.,;:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# -----------------------
# Config
# -----------------------

# input_file = "ensemble_outputs_alternate/merged_output.csv"  # or .csv
# input_basename, input_ext = os.path.splitext(input_file)
# input_ext = input_ext.lower()

# if input_ext not in (".jsonl", ".csv"):
#     raise ValueError(f"Unsupported file type: {input_ext}")

# # Output file name will match input type
# #output_file = f"without_answer/{input_basename}_without_answer{input_ext}"
# output_file = f"without_answer/outputs/merged_output_alternate_without_answer{input_ext}"

# # -----------------------
# # Load Data
# # -----------------------
if input_ext == ".jsonl":
    df = pd.read_json(input_file, lines=True)
elif input_ext == ".csv":
    df = pd.read_csv(input_file)

if "original" in input_file.lower():
    df = df.sample(n=100, random_state=42)

# -----------------------
# Model setup
# -----------------------
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/workspace/hf")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True,
    cache_dir="/workspace/hf"
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# -----------------------
# Processing
# -----------------------
processed_texts = []

# Detect which column to process
if "LLM Thinking" in df.columns:
    input_col = "LLM Thinking"
elif "Ensembled Thought" in df.columns:
    input_col = "Ensembled Thought"
else:
    raise ValueError("No valid input column found in file.")

for i, text in enumerate(df[input_col], 1):
    text = extract_think_text(str(text))

    messages = [
        # {"role": "system", "content": "You are a helpful assistant that removes direct answers but keeps explanations or hints."},
        {"role": "user", "content": f"""
Task: Keep only the hints from the text and remove answer sentences.

Definition: 
- A "hint/explanation sentence" provides guidance that helps someone think about the problem without giving the final solution.  
- An "answer sentence" directly states the final answer, solution, result, or conclusion.

Instructions:  
1. Keep every hint/explanation sentence exactly as written.  
2. Remove all answer sentences. 
3. Preserve the original wording, order, and formatting of the remaining text.  
4. Output only the hints. 

Original text:
{text}
"""}
    ]
    chat_templated_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_tokens = tokenizer(chat_templated_text, return_tensors="pt")
    num_prompt_tokens = prompt_tokens.input_ids.shape[1]

    output = generator(
        chat_templated_text,
        max_new_tokens=num_prompt_tokens,
        do_sample=False,
        temperature=0,
    )

    generated_text = output[0]["generated_text"]

    original_sents = sent_tokenize(text)
    generated_sents = sent_tokenize(generated_text)

    normalized_original = {normalize_text(s): s for s in original_sents}
    filtered_sents = [
        normalized_original[norm_sent]
        for sent in generated_sents
        if (norm_sent := normalize_text(sent)) in normalized_original
    ]

    filtered_text = " ".join(filtered_sents)
    processed_texts.append(filtered_text)
    print(f"Filtered text: {filtered_text}", flush=True)
    print(f"Processed {i}/{len(df)}", flush=True)

# -----------------------
# Save
# -----------------------
output_col = f"{input_col} Without Answer"
df[output_col] = processed_texts

if input_ext == ".jsonl":
    df.to_json(output_file, orient="records", lines=True)
elif input_ext == ".csv":
    df.to_csv(output_file, index=False)

print(f"Saved processed file to: {output_file}")

