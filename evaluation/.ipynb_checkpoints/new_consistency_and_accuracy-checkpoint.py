import itertools
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
# ----------------------
# Data loading
# ----------------------
def load_jsonl(file_path):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# ----------------------
# File collection
# ----------------------
def get_jsonl_files_from_dir(directory):
    """Get all .jsonl files from a directory (full paths)."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".jsonl")
    ]

# ----------------------
# Generate model+setting combinations
# ----------------------
# def get_all_model_combinations(jsonl_files, extract_model_name_from_filename):
#     """Return all subsets of models (size >= 2) with all setting combinations."""
#     model_to_files = {}
#     for file in jsonl_files:
#         model_name = extract_model_name_from_filename(file)
#         if model_name is None:
#             print(f"⚠️ Could not extract model name from: {file}")
#         model_to_files.setdefault(model_name, []).append(file)
#     print(model_to_files.keys())
#     model_names = sorted(model_to_files.keys())
#     all_combinations = []

#     for r in range(2, len(model_names) + 1):
#         for model_subset in itertools.combinations(model_names, r):
#             files_per_model = [model_to_files[m] for m in model_subset]
#             for settings_combo in itertools.product(*files_per_model):
#                 all_combinations.append(settings_combo)

#     return all_combinations

def get_all_model_combinations(jsonl_files, extract_model_name_from_filename, extract_setting_from_filename):
    model_to_files = {}
    model_to_settings = {}
    
    for file in jsonl_files:
        model_name = extract_model_name_from_filename(file)
        setting_name = extract_setting_from_filename(file)
        model_to_files.setdefault(model_name, {}).setdefault(setting_name, []).append(file)
        model_to_settings.setdefault(model_name, set()).add(setting_name)
    
    model_names = sorted(model_to_files.keys())
    print(model_to_settings)
    all_combinations = []

    for r in range(2, len(model_names) + 1):
        for model_subset in itertools.combinations(model_names, r):
            # Find settings common to ALL models in this subset
            common_settings = set.intersection(*(model_to_settings[m] for m in model_subset))
            for setting in common_settings:
                # Pick one file per model for this setting
                files_for_setting = [model_to_files[m][setting][0] for m in model_subset]
                all_combinations.append(tuple(files_for_setting))
    
    return all_combinations


# ----------------------
# Consistency & accuracy calculation
# ----------------------
def calculate_consistency_and_accuracy(files, extract_model_name_from_filename):
    all_model_data = {}
    model_accuracies = {}
    
    for file_path in files:
        model_name = extract_model_name_from_filename(file_path)
        data = load_jsonl(file_path)
        all_model_data[model_name] = data
        
        correct_count = sum(1 for item in data if item.get("Result", "").lower() == "correct")
        model_accuracies[model_name] = correct_count / len(data)
    
    num_items = len(next(iter(all_model_data.values())))
    consistent_count = 0
    
    for i in range(num_items):
        answers = [data[i].get("LLM Answer") for data in all_model_data.values()]
        results = [data[i].get("Result", "").lower() for data in all_model_data.values()]
        
        if len(set(answers)) == 1:
            consistent_count += 1
        else:
            if all(res == "correct" for res in results):
                consistent_count += 1
    
    consistency_rate = consistent_count / num_items
    return consistency_rate, model_accuracies

# ----------------------
# Analysis for all subsets with settings
# ----------------------
def analyze_all(jsonl_files, extract_model_name_from_filename, extract_setting_from_filename):
    combinations = get_all_model_combinations(jsonl_files, extract_model_name_from_filename, extract_setting_from_filename)
    
    rows = []
    for combo in combinations:
        consistency, accuracies = calculate_consistency_and_accuracy(combo, extract_model_name_from_filename)
        subset_models = [extract_model_name_from_filename(f) for f in combo]
        subset_settings = [extract_setting_from_filename(f) for f in combo]
        
        row = {
            "subset_models": tuple(subset_models),
            "subset_settings": tuple(subset_settings),
            "files": combo,
            "consistency_rate": consistency
        }
        for m, acc in accuracies.items():
            row[f"accuracy_{m}"] = acc
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

# ----------------------
# Visualization & saving
# ----------------------
def plot_results(df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    subset_labels = [
        " + ".join(f"{m}({s})" for m, s in zip(models, settings))
        for models, settings in zip(df["subset_models"], df["subset_settings"])
    ]

    # # --- Consistency bar plot ---
    # plt.figure(figsize=(12, 6))
    # plt.bar(subset_labels, df["consistency_rate"], color="skyblue", edgecolor="black")
    # plt.xticks(rotation=90)
    # plt.ylabel("Consistency Rate")
    # plt.title("Consistency Rate per Model+Setting Combination")
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "consistency_rate_bar.png"), dpi=300)
    # # plt.show()

    # # --- Accuracy heatmap ---
    # acc_cols = [col for col in df.columns if col.startswith("accuracy_")]
    # if acc_cols:
    #     acc_df = df[acc_cols]
    #     plt.figure(figsize=(12, 6))
    #     plt.imshow(acc_df, cmap="viridis", aspect="auto")
    #     plt.colorbar(label="Accuracy")
    #     plt.xticks(range(len(acc_cols)), acc_cols, rotation=90)
    #     plt.yticks(range(len(df)), subset_labels)
    #     plt.title("Model Accuracy Heatmap")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, "accuracy_heatmap.png"), dpi=300)
    #     # plt.show()

    # Save DataFrame as CSV
    df.to_csv(os.path.join(output_dir, "model_subset_results.csv"), index=False)
    print(f"Results saved to {output_dir}")

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    def extract_model_name_from_filename(filename):
        base = os.path.basename(filename).replace('.jsonl', '')
        transfer_thoughts = False
        if "thoughts_to" in base:
            base = base.split("_thoughts_to_")[-1]
            transfer_thoughts = True
        if "dapo" in base.lower():
            return "BytedTsinghua-SIA/DAPO-Qwen-32B"
        elif "qwq" in base.lower():
            return "Qwen/QwQ-32B"
        elif "openthinker" in base.lower():
            return "open-thoughts/OpenThinker-7B"
        elif "gpt-oss" in base.lower():
            if "medium" in base.lower():
                return "openai/gpt-oss-20b-medium"
            elif "low" in base.lower():
                return "openai/gpt-oss-20b-low"
            elif "high" in base.lower():
                return "openai/gpt-oss-20b-high"
            else:
                return "openai/gpt-oss-20b" if not transfer_thoughts else "openai/gpt-oss-20b-medium"
        else:
            raise ValueError(f"Unknown model for filename: {filename}")
    # def extract_model_name_from_filename(filename):
    #     base = os.path.basename(filename).replace('.jsonl', '')
    #     transfer_thoughts = False
    #     if "thoughts_to" in base:
    #         base = base.split("_thoughts_to_")[-1]
    #         transfer_thoughts = True
            
    #     if "dapo" in base.lower():
    #         return "BytedTsinghua-SIA/DAPO-Qwen-32B"
    #     elif "qwq" in base.lower():
    #         return "Qwen/QwQ-32B"
    #     elif "openthinker" in base.lower():
    #         return "open-thoughts/OpenThinker-7B"
    #     elif "gpt-oss" in base.lower():
    #         if "medium" in base.lower():
    #             return "openai/gpt-oss-20b-medium"
    #         elif "low" in base.lower():
    #             return "openai/gpt-oss-20b-low"
    #         elif "high" in base.lower():
    #             return "openai/gpt-oss-20b-high"
    #         else:
    #             if transfer_thoughts:
    #                 return "openai/gpt-oss-20b-medium"
    #             else:
    #                 return "openai/gpt-oss-20b"
    #     else:
    #         return "unknown_model"

    def extract_setting_from_filename(filename):
        print(filename)
        base = os.path.basename(filename).replace('.jsonl', '').lower()
        if "thoughts_to" in base:
            transfer_without_answer = False
            if "without" in base:
                transfer_without_answer = True
            base = base.split("_thoughts_to_")[0]
            setting = base + " Complete Thought Transfer"
            if transfer_without_answer:
                setting = setting + " Without Answer"
                print(setting)
            return setting
        if "empty" in base:
            return "Empty Thought"
        if "ensembled_thought_without_answer" in base:
            if "alternate" in base:
                return "Ensembled Thought Without Answer V2"
            else:
                return "Ensembled Thought Without Answer V1"
        if ("without_answer" not in base) and ("alternate" in base) and ("ensembled_thought" in base):
            return "Ensembled Thought V2"
        if "ensembled_thought_merged" in base and ("alternate" not in base):
            return "Ensembled Thought V1"

        if "original" in base:
            return "original"
        return "No Setting"

    # Directory containing your JSONL files
    input_dir = "outputs"

    jsonl_files = get_jsonl_files_from_dir(input_dir)
    print(f"Found {len(jsonl_files)} JSONL files.")

    df_results = analyze_all(jsonl_files, extract_model_name_from_filename, extract_setting_from_filename)
    plot_results(df_results, output_dir="plots")
