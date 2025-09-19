import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import nltk
import re

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

def extract_think_text(text):
    # Pattern to extract text between <think> and </think>
    # If </think> is missing, take all text after <think>
    match = re.search(r"<think>(.*?)(</think>|$)", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # If no <think> tag found, return original or empty string as you prefer
        return text

def normalize_text(text):
    text = text.lower()
    text = text.strip()
    # Remove trailing punctuation like . , ; : at end of sentence
    text = re.sub(r"[.,;:]+$", "", text)
    # Replace multiple spaces with one
    text = re.sub(r"\s+", " ", text)
    return text

model_name = "mistralai/Ministral-8B-Instruct-2410"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/workspace/hf")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,cache_dir="/workspace/hf"
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

df = pd.read_csv("ensemble_outputs/merged_output.csv")

for i, text in enumerate(df["Ensembled Thought"], 1):
    text = extract_think_text(text)
    # Chat-style prompt: system + user messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant that removes direct answers but keeps explanations or hints."},
        {"role": "user", "content": f"""
Given the following text containing both answers and hints/explanations, remove any sentences that provide the answers. Do not add or change anything else. Keep only the explanations or hints.

Original text:
{text}
"""}]
    # prompt = f"""Given the following text containing both answers and hints/explanations, remove any sentences that provide the answers. Do not add or change anything else. Keep only the explanations or hints. Original text:{text}"""
    # Format the prompt for generation as plain text (concatenate roles)
    #prompt = "\n".join([f"{m['role']}: {m['content'].strip()}" for m in messages])
    chat_templated_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer(chat_templated_text, return_tensors="pt")
    num_prompt_tokens = prompt_tokens.input_ids.shape[1]

    output = generator(
        chat_templated_text,
        max_new_tokens=num_prompt_tokens,
        do_sample=False,
        temperature=0,
    )

    generated_text = output[0]["generated_text"]

    # Normalize and split sentences
    original_sents = sent_tokenize(text)
    generated_sents = sent_tokenize(generated_text)

    # Normalize sentences for comparison
    normalized_original = {normalize_text(s): s for s in original_sents}  # map normalized -> original
    filtered_sents = []

    for sent in generated_sents:
        norm_sent = normalize_text(sent)
        if norm_sent in normalized_original:
            filtered_sents.append(normalized_original[norm_sent])  # append original sentence for nice formatting

    filtered_text = " ".join(filtered_sents)
    processed_texts.append(filtered_text)
    print(f"Filtered text: {filtered_text}", flush=True)
    print(f"Processed {i}/{len(df)}", flush=True)

df["Ensembled Thought Without Answer"] = processed_texts
df.to_csv("ensemble_outputs/merged_output_without_answer.csv", index=False)
