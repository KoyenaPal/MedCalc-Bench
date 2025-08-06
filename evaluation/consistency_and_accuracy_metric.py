import os
import glob
import json
from itertools import combinations
from collections import defaultdict, Counter
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -------- CONFIG --------
data_dir = "outputs"  # Replace with your actual folder path
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

combo_csv = "model_combination_consistency_accuracy.csv"
individual_csv = "model_individual_accuracy.csv"
row_csv = "consistency_report_with_ground_truth.csv"
# ------------------------

def load_items(filepath):
    with open(filepath, 'r') as f:
        if filepath.endswith('.jsonl'):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)

# Step 1: Load data
row_data = defaultdict(lambda: {"ground_truth": None, "models": {}, "results": {}})
all_models = set()

for filepath in glob.glob(os.path.join(data_dir, "*ensembled_thought.jsonl")):
    try:
        items = load_items(filepath)
        for item in items:
            row = item.get("Row Number")
            model = item.get("LLM Name")
            answer = item.get("LLM Answer")
            ground_truth = item.get("Ground Truth Answer")
            result = item.get("Result")

            if row is not None and model and answer is not None:
                row_data[row]["models"][model] = answer
                row_data[row]["results"][model] = result
                all_models.add(model)

                if ground_truth is not None:
                    if row_data[row]["ground_truth"] is not None and row_data[row]["ground_truth"] != ground_truth:
                        print(f"⚠️ Inconsistent ground truth for Row {row}")
                    row_data[row]["ground_truth"] = ground_truth
    except Exception as e:
        print(f"⚠️ Failed to parse {filepath}: {e}")

all_models = sorted(all_models)

# Step 2: Initialize model accuracy tracking using "Result" field only
model_accuracy = {
    model: {
        "correct_reported": 0,
        "total_reported": 0
    } for model in all_models
}

# Step 3: Prepare row-level consistency report (consistency based on LLM Answer; accuracy from Result)
row_results = []

for row, info in row_data.items():
    models = info["models"]
    results = info["results"]
    ground_truth = info["ground_truth"]

    answers = list(models.values())
    model_names = list(models.keys())
    answer_counts = Counter(answers)
    most_common_answer, freq = answer_counts.most_common(1)[0] if answer_counts else (None, 0)
    consistency_score = freq / len(answers) if answers else 0

    # Determine if majority answer is "Correct" based on Result of models giving that answer
    majority_correct = False
    for model, ans in models.items():
        if ans == most_common_answer and results.get(model) == "Correct":
            majority_correct = True
            break

    # Update individual model accuracy from Result field
    for model, result in results.items():
        if result in {"Correct", "Incorrect"}:
            model_accuracy[model]["total_reported"] += 1
            if result == "Correct":
                model_accuracy[model]["correct_reported"] += 1

    row_results.append({
        "Row Number": row,
        "Answers": answers,
        "LLM Names": model_names,
        "Most Common Answer": most_common_answer,
        "Consistency Score": round(consistency_score, 3),
        "Ground Truth Answer": ground_truth,
        "Is Correct?": majority_correct
    })

with open(row_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "Row Number", "Answers", "LLM Names", "Most Common Answer",
        "Consistency Score", "Ground Truth Answer", "Is Correct?"
    ])
    writer.writeheader()
    for row in row_results:
        writer.writerow(row)

# Step 4: Save individual model accuracy based ONLY on Result field
with open(individual_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "Model", 
        "Accuracy (Reported Result)",
        "Reported Correct", "Reported Total"
    ])
    writer.writeheader()
    for model, stats in model_accuracy.items():
        reported_acc = stats["correct_reported"] / stats["total_reported"] if stats["total_reported"] else 0
        writer.writerow({
            "Model": model,
            "Accuracy (Reported Result)": round(reported_acc, 3),
            "Reported Correct": stats["correct_reported"],
            "Reported Total": stats["total_reported"]
        })

# Step 5: Group consistency + accuracy (majority vs GT) unchanged - consistency uses LLM Answer agreement
combo_results = []
pairwise_matrix = pd.DataFrame(index=all_models, columns=all_models, dtype=float)

for r in range(2, len(all_models) + 1):
    for combo in combinations(all_models, r):
        rows_considered = 0
        consistency_sum = 0
        correct_majority = 0
        with_ground_truth = 0

        for row, data in row_data.items():
            models = data["models"]
            if all(m in models for m in combo):
                answers = [models[m] for m in combo]
                most_common_answer, count = Counter(answers).most_common(1)[0]
                consistency = count / len(combo)
                consistency_sum += consistency
                rows_considered += 1

                ground_truth = data["ground_truth"]
                if ground_truth is not None:
                    with_ground_truth += 1
                    if most_common_answer == str(ground_truth):
                        correct_majority += 1

        if rows_considered > 0:
            avg_consistency = consistency_sum / rows_considered
            accuracy = correct_majority / with_ground_truth if with_ground_truth > 0 else None
            combo_results.append({
                "Model Combination": ", ".join(combo),
                "Num Rows Compared": rows_considered,
                "Average Consistency Score": round(avg_consistency, 3),
                "Accuracy (Match to Ground Truth)": round(accuracy, 3) if accuracy is not None else "N/A"
            })

            if len(combo) == 2:
                m1, m2 = combo
                pairwise_matrix.loc[m1, m2] = round(avg_consistency, 3)
                pairwise_matrix.loc[m2, m1] = round(avg_consistency, 3)

with open(combo_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "Model Combination", "Num Rows Compared",
        "Average Consistency Score", "Accuracy (Match to Ground Truth)"
    ])
    writer.writeheader()
    for row in combo_results:
        writer.writerow(row)

# Step 6: Visualizations
sns.set(style="whitegrid")

# Individual accuracy bar plot (using only Result accuracy)
acc_data = pd.read_csv(individual_csv)
acc_data["Accuracy (Reported Result)"] = pd.to_numeric(acc_data["Accuracy (Reported Result)"], errors="coerce")
acc_reported = acc_data.dropna(subset=["Accuracy (Reported Result)"])

plt.figure(figsize=(8, 5))
sns.barplot(data=acc_reported, x="Model", y="Accuracy (Reported Result)")
plt.title("Individual Model Accuracy (Reported Result)")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "individual_model_accuracy_reported.png"))
plt.close()

# Heatmap for pairwise consistency (based on LLM Answer agreement)
plt.figure(figsize=(8, 6))
sns.heatmap(pairwise_matrix.astype(float), annot=True, cmap="YlGnBu", vmin=0, vmax=1)
plt.title("Pairwise Model Consistency (LLM Answer Agreement)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pairwise_consistency_heatmap.png"))
plt.close()

print(f"CSV and visualizations saved in:\n- {row_csv}\n- {individual_csv}\n- {combo_csv}\n- {output_dir}")
