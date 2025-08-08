import json 
import numpy as np
import os

from collections import defaultdict

def combined_compute_overall_accuracy(output_path, prompt_style, additional_output_file_info=""):
    # Structure: model_name -> category -> list of 0/1
    model_category_accuracy = defaultdict(lambda: defaultdict(list))
    combined_category_accuracy = defaultdict(list)

    with open(f"outputs/{output_path}") as file:
        for line in file:
            data = json.loads(line)
            category = data["Category"]

            model_1 = data.get("LLM Model 1")
            model_2 = data.get("LLM Model 2")
            result_1 = data.get("LLM Result 1", "Incorrect")
            result_2 = data.get("LLM Result 2", "Incorrect")

            # Record for individual models
            if model_1:
                model_category_accuracy[model_1][category].append(1 if result_1 == "Correct" else 0)
            if model_2:
                model_category_accuracy[model_2][category].append(1 if result_2 == "Correct" else 0)

            # Record for ensemble logic (either correct counts as correct)
            combined_result = "Correct" if result_1 == "Correct" or result_2 == "Correct" else "Incorrect"
            combined_category_accuracy[category].append(1 if combined_result == "Correct" else 0)

    # Report per-model accuracy
    for model, cat_results in model_category_accuracy.items():
        print(f"\n--- Accuracy Report for Model: {model} ---")
        total_correct = 0
        total_count = 0
        for category, results in cat_results.items():
            acc = sum(results) / len(results) * 100 if results else 0.0
            print(f"{category}: {acc:.2f}% ({sum(results)}/{len(results)})")
            total_correct += sum(results)
            total_count += len(results)
        overall = (total_correct / total_count * 100) if total_count else 0.0
        print(f"Overall Accuracy: {overall:.2f}% ({total_correct}/{total_count})")

    # Report combined model accuracy
    print(f"\n--- Combined (Ensemble) Accuracy Report ---")
    total_correct = 0
    total_count = 0
    for category, results in combined_category_accuracy.items():
        acc = sum(results) / len(results) * 100 if results else 0.0
        print(f"{category}: {acc:.2f}% ({sum(results)}/{len(results)})")
        total_correct += sum(results)
        total_count += len(results)
    overall = (total_correct / total_count * 100) if total_count else 0.0
    print(f"Overall Accuracy: {overall:.2f}% ({total_correct}/{total_count})\n")

def compute_overall_accuracy(output_path, model_name, prompt_style, is_target_model=False, additional_output_file_info=""): 
    category_accuracy = {}

    with open(f"outputs/{output_path}") as file:
        for line in file:
            data = json.loads(line)
            
            category = data["Category"]

            if category not in category_accuracy:
                category_accuracy[category] = []

            if not is_target_model:
                if data["Result"] == "Correct":
                    category_accuracy[category].append(1)
                else:
                    category_accuracy[category].append(0)

            if is_target_model:
                if data["Target Result"] == "Correct":
                    category_accuracy[category].append(1)
                else:
                    category_accuracy[category].append(0)

    # Compute average and standard deviation for each category
    category_stats = {}
    all_results = []

    for cat, results in category_accuracy.items():
        results_array = np.array(results)
        category_mean = np.mean(results_array)
        category_std = round(np.sqrt(category_mean * (1-category_mean) / len(results_array)), 2)
        category_stats[cat] = {
            "average": round(category_mean * 100, 2),
            "std": category_std
        }
        all_results.extend(results)

    # Compute overall average and standard deviation
    all_results_array = np.array(all_results)
    overall_average = np.mean(all_results_array)
    overall_std =  round(np.sqrt(overall_average * (1-overall_average) / 1047), 2)

    category_stats["overall"] = {
        "average": round(overall_average * 100, 2),
        "std": overall_std
    }

    if not os.path.exists("results"):
        os.makedirs("results")

    if "/" in model_name:
        model_name = model_name.split('/')[1]

    with open(f"results/results_{model_name}_{prompt_style}_{additional_output_file_info}.json", "w") as file:
        json.dump(category_stats, file, indent=4)

    return category_stats

