import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def extract_answer_type(filename):
    filename = os.path.basename(filename).lower()
    if "ensembled_thought_without_last" in filename:
        return "ensembled_thought_without_last"
    elif "ensembled_thought" in filename:
        return "ensembled_thought"
    elif "original" in filename:
        return "original"
    elif "empty" in filename:
        return "empty"
    else:
        return "unknown"

def extract_model_name_from_filename(filename):
    base = os.path.basename(filename).replace('.jsonl', '')
    parts = base.split('_')
    if "dapo" in base.lower():
        return "BytedTsinghua-SIA/DAPO-Qwen-32B"
    elif "qwq" in base.lower():
        return "Qwen/QwQ-32B"
    elif "openthinker" in base.lower():
        return "open-thoughts/OpenThinker-7B"
    else:
        return "unknown_model"

# Load all JSONL files
jsonl_files = glob.glob("outputs/*.jsonl")  # Update path

all_items = []

for filepath in jsonl_files:
    answer_type = extract_answer_type(filepath)
    model_name_from_file = extract_model_name_from_filename(filepath)

    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            model_name = item.get("LLM Name")
            if not model_name or model_name.strip() == "":
                model_name = model_name_from_file
            item['Model'] = model_name
            item['Answer Type'] = answer_type
            all_items.append(item)

df = pd.DataFrame(all_items)

# Clean and prepare for accuracy calculation
df['Result_clean'] = df['Result'].str.lower().fillna("")
df['is_correct'] = df['Result_clean'] == "correct"


# Calculate accuracy per Model and Answer Type
accuracy_df = df.groupby(['Model', 'Answer Type']).agg(
    Total=('is_correct', 'size'),
    Correct=('is_correct', 'sum')
).reset_index()
accuracy_df['Accuracy'] = accuracy_df['Correct'] / accuracy_df['Total']

accuracy_df = accuracy_df[accuracy_df['Answer Type'] != 'unknown']

# Save accuracy to CSV
accuracy_df.to_csv("results/overall_model_accuracy_by_answer_type.csv", index=False)
print("Saved accuracy table to 'model_accuracy_by_answer_type.csv'")

# Visualization
plt.figure(figsize=(12, 10))
sns.barplot(
    data=accuracy_df,
    y='Model',       # swap axes
    x='Accuracy',
    hue='Answer Type',
    orient='h'
)
plt.xlim(0, 1)
plt.title('Model Accuracy by Answer Type')
plt.legend(title='Answer Type')
plt.tight_layout()

# Save the plot image
plt.savefig("visualizations/overall_model_accuracy_by_answer_type.png", dpi=300)
print("Saved plot to 'model_accuracy_by_answer_type.png'")

