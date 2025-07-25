import json 
import numpy as np
import os


def compute_overall_accuracy(output_path, model_name, prompt_style, is_target_model=False): 
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

    with open(f"results/results_{model_name}_{prompt_style}.json", "w") as file:
        json.dump(category_stats, file, indent=4)

    return category_stats

