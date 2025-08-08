import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

folder_path = "ensemble_outputs"

model_name_map = {0: "BytedTsinghua-SIA/DAPO-Qwen-32B", 1: "Qwen/QwQ-32B"}

overall_counts = defaultdict(int)
per_file_counts = {}

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model_counts = defaultdict(int)

        for item in data:
            if isinstance(item, dict) and len(item) == 1:
                inner = next(iter(item.values()))
            else:
                inner = item

            model_index = inner.get('selected_model_index')
            if model_index is not None:
                model_counts[model_index] += 1
                overall_counts[model_index] += 1

        # Map model indices to names for per file counts
        mapped_counts = {model_name_map.get(k, str(k)): v for k, v in model_counts.items()}
        per_file_counts[filename] = mapped_counts

# Map overall counts keys
mapped_overall_counts = {model_name_map.get(k, str(k)): v for k, v in overall_counts.items()}

# -------- Visualization -----------

# 1. Overall counts bar plot
plt.figure(figsize=(8, 5))
model_names = list(mapped_overall_counts.keys())
counts = [mapped_overall_counts[m] for m in model_names]

sns.barplot(x=model_names, y=counts)
plt.title('Overall Sentence Selection Frequency per Model')
plt.xlabel('Model')
plt.ylabel('Number of Sentences Selected')
plt.tight_layout()
plt.savefig('results/overall_model_selection_frequency.png')
plt.show()

# 2. Per file counts heatmap
df = pd.DataFrame.from_dict(per_file_counts, orient='index').fillna(0)
df = df[sorted(df.columns, key=lambda x: ["BytedTsinghua-SIA/DAPO-Qwen-32B", "Qwen/QwQ-32B"].index(x) if x in ["BytedTsinghua-SIA/DAPO-Qwen-32B", "Qwen/QwQ-32B"] else x)]
df = df.sort_index()

plt.figure(figsize=(12, 10))
sns.heatmap(df, annot=False, fmt='g', cmap='YlGnBu')
plt.title('Sentence Selection Frequency per Model (Per File)')
plt.xlabel('Model')
plt.ylabel('File Name')
plt.tight_layout()
plt.savefig('results/per_file_model_selection_heatmap.png')
plt.show()
