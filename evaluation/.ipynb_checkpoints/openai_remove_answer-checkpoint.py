#!/usr/bin/env python3
import os
import pandas as pd
import nltk
import re
import argparse
import random
import numpy as np
import time

# OpenAI client
from openai import OpenAI

# optional token estimator
try:
    import tiktoken
except Exception:
    tiktoken = None

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# -----------------------
# Arg parser, seeds, I/O
# -----------------------
torch_random_seed = 42
random.seed(torch_random_seed)
np.random.seed(torch_random_seed)

parser = argparse.ArgumentParser(description="Process input file and generate output without answers using o4-mini.")
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
parser.add_argument(
    "--api_key",
    type=str,
    default=None,
    help="OpenAI API key (optional). If not provided, the OPENAI_API_KEY env var will be used."
)
parser.add_argument(
    "--sleep_between_calls",
    type=float,
    default=0.0,
    help="Optional sleep (seconds) between API calls to avoid rate limits."
)

args = parser.parse_args()
input_file = args.input_file
input_basename, input_ext = os.path.splitext(os.path.basename(input_file))
input_ext = input_ext.lower()

if input_ext not in (".jsonl", ".csv"):
    raise ValueError(f"Unsupported file type: {input_ext}")

if args.output_file:
    output_file = args.output_file
else:
    output_folder = "without_answer"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{input_basename}_without_answer{input_ext}")

print(f"Input file: {input_file}")
print(f"Output file: {output_file}")

def extract_think_text(text):
    # explicitly remove answer
    match = re.search(r"<think>(.*?)(</think>|$)", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text

# -----------------------
# Load Data
# -----------------------
if input_ext == ".jsonl":
    df = pd.read_json(input_file, lines=True)
elif input_ext == ".csv":
    df = pd.read_csv(input_file)

if "original" in input_file.lower():
    df = df.sample(n=100, random_state=42)

# -----------------------
# OpenAI client setup
# -----------------------
OPENAI_API_KEY = args.api_key or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key missing. Set OPENAI_API_KEY env var or pass --api_key.")

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "o4-mini"  # using o4-mini via Responses API

# optional token count helper
def estimate_token_count(text, model="o4-mini"):
    if tiktoken is None:
        # fallback: rough words -> tokens estimate (words * 1.3)
        return max(32, int(len(text.split()) * 1.3))
    try:
        # try to get encoding for the model; fallback to gpt2
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("gpt2")
        return len(enc.encode(text))
    except Exception:
        return max(32, int(len(text.split()) * 1.3))

# -----------------------
# Detect which column to process
# -----------------------
if "LLM Thinking" in df.columns:
    input_col = "LLM Thinking"
elif "Ensembled Thought" in df.columns:
    input_col = "Ensembled Thought"
else:
    raise ValueError("No valid input column found in file.")

processed_texts = []

# -----------------------
# Main loop — call OpenAI o4-mini
# -----------------------
for i, text in enumerate(df[input_col], 1):
    text = extract_think_text(str(text))

    # Build the single user message (same content as your original messages variable)
    messages = [
        {
            "role": "user",
            "content": f"""
Task: Keep only the hints from the text and remove answer sentences.

Definition: 
- A "hint/explanation sentence" provides guidance that helps someone think about the problem without giving the final solution.  
- An "answer sentence" directly states the final answer, solution, result, or conclusion.

Instructions:  
1. Keep every hint/explanation sentence exactly as written.  
2. Remove all answer sentences and statements.
3. Preserve the original wording, order, and formatting of the remaining text.
4. Do not add, rephrase, or generate any new text beyond what is already in the original.
5. Output only the hints.

Original text:
{text}
"""
        }
    ]

    # build a prompt string or pass role-based input directly. We'll pass the role-based messages list to the Responses API.
    # Estimate tokens to set max_output_tokens (Responses API parameter)
    chat_templated_text = messages[0]["content"]
    num_prompt_tokens = estimate_token_count(chat_templated_text, model=MODEL_NAME)
    # match your original behavior where max_new_tokens == prompt_tokens
    max_output_tokens = max(32, num_prompt_tokens)

    # Call the Responses API
    try:
        prompt_len = estimate_token_count(messages[0]["content"], model=MODEL_NAME)
        resp = client.chat.completions.create(
            model="o4-mini",
            messages=messages,
            reasoning_effort="medium"
        )
    except Exception as e:
        print(f"[ERROR] API call failed for row {i}: {e}")
        processed_texts.append("")
        continue

    generated_text = extract_think_text(resp.choices[0].message.content)
    print("GENERATED TEXT", generated_text, flush=True)
    processed_texts.append(generated_text)

    print(f"Processed {i}/{len(df)}", flush=True)

    if args.sleep_between_calls and args.sleep_between_calls > 0:
        time.sleep(args.sleep_between_calls)

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
