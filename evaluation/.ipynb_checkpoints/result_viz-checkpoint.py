import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os

# Read file
df = pd.read_csv("plots/model_subset_results.csv", sep=",")  # adjust separator

# Parse tuple-like strings
df["subset_models"] = df["subset_models"].apply(ast.literal_eval)
df["subset_settings"] = df["subset_settings"].apply(ast.literal_eval)

# --- Filter: keep only rows where all settings are the same ---
df = df[df["subset_settings"].apply(lambda s: len(set(s)) == 1)].copy()

# All accuracy columns
accuracy_cols = [c for c in df.columns if c.startswith("accuracy_")]

os.makedirs("heatmaps", exist_ok=True)

# Group by the unique setting value (since now all entries in tuple are identical)
df["single_setting"] = df["subset_settings"].apply(lambda s: s[0])


for setting, group in df.groupby("single_setting"):
    setting_str = setting.replace(" ", "_")
    models_in_group = sorted(set(m for tup in group["subset_models"] for m in tup))

    # --- Consistency heatmap ---
    consistency_mat = pd.DataFrame(float("nan"), index=models_in_group, columns=models_in_group)
    for _, row in group.iterrows():
        if len(row["subset_models"]) == 2:
            m1, m2 = row["subset_models"]
            consistency_mat.loc[m1, m2] = row["consistency_rate"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(consistency_mat, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1)
    plt.title(f"Consistency Heatmap\nsetting={setting}")
    plt.tight_layout()
    plt.savefig(f"heatmaps/consistency_heatmap_{setting_str}.png", dpi=300)
    plt.close()

# Consistent sizing parameters
BAR_WIDTH = 6
BAR_HEIGHT_PER_MODEL = 0.5

for setting, group in df.groupby("single_setting"):
    setting_str = setting.replace(" ", "_")
    models_in_group = sorted(set(m for tup in group["subset_models"] for m in tup))

    # --- Accuracy horizontal bar chart ---
    accuracy_scores = []
    for model in models_in_group:
        col = f"accuracy_{model}"
        if col in group:
            vals = group[col].dropna()
            if not vals.empty:
                accuracy_scores.append((model, vals.mean()))

    acc_df = pd.DataFrame(accuracy_scores, columns=["Model", "Accuracy"]).sort_values(
        "Accuracy", ascending=False
    )

    # Fixed figure size: width constant, height based on number of models
    plt.figure(figsize=(BAR_WIDTH, max(2, BAR_HEIGHT_PER_MODEL * len(acc_df))))
    sns.barplot(x="Accuracy", y="Model", data=acc_df, palette="Blues_r")
    plt.xlim(acc_df["Accuracy"].min() - 0.01, acc_df["Accuracy"].max() + 0.01)
    plt.title(f"Model Accuracy\nsetting={setting}")
    for i, v in enumerate(acc_df["Accuracy"]):
        plt.text(v + 0.005, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    plt.savefig(f"heatmaps/accuracy_bar_{setting_str}.png", dpi=300)
    plt.close()

# Prepare grouped accuracy data
grouped_scores = []

for setting, group in df.groupby("single_setting"):
    models_in_group = sorted(set(m for tup in group["subset_models"] for m in tup))
    for model in models_in_group:
        col = f"accuracy_{model}"
        if col in group:
            vals = group[col].dropna()
            if not vals.empty:
                grouped_scores.append((model, setting, vals.mean()))

grouped_df = pd.DataFrame(grouped_scores, columns=["Model", "Setting", "Accuracy"])

# Sort models by their overall average
model_order = grouped_df.groupby("Model")["Accuracy"].mean().sort_values(ascending=False).index

plt.figure(figsize=(14, 6))  # wider chart
sns.barplot(x="Model", y="Accuracy", hue="Setting", data=grouped_df, order=model_order)
plt.ylim(0, 1)
plt.title("Model Accuracy per Setting")
plt.xticks(rotation=45, ha="right")  # angled labels
plt.legend(title="Setting", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("heatmaps/combined_accuracy_grouped.png", dpi=300)
plt.close()



# for setting, group in df.groupby("single_setting"):
#     setting_str = setting.replace(" ", "_")

#     # Models in this group
#     models_in_group = sorted(set(m for tup in group["subset_models"] for m in tup))

#     # --- Accuracy matrix ---
#     accuracy_mat = pd.DataFrame(float("nan"), index=models_in_group, columns=models_in_group)
#     consistency_mat = pd.DataFrame(float("nan"), index=models_in_group, columns=models_in_group)

#     for _, row in group.iterrows():
#         src_models = row["subset_models"]
#         if len(src_models) == 2:
#             m1, m2 = src_models
#             # Fill from accuracy columns if present
#             acc_col_m2 = f"accuracy_{m2}"
#             if acc_col_m2 in row and pd.notna(row[acc_col_m2]):
#                 accuracy_mat.loc[m1, m2] = row[acc_col_m2]
#             # Fill consistency
#             consistency_mat.loc[m1, m2] = row["consistency_rate"]

#     # Plot accuracy heatmap
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(accuracy_mat, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
#     plt.title(f"Accuracy Heatmap\nsetting={setting}")
#     plt.tight_layout()
#     plt.savefig(f"heatmaps/accuracy_heatmap_{setting_str}.png", dpi=300)
#     plt.close()

#     # Plot consistency heatmap
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(consistency_mat, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1)
#     plt.title(f"Consistency Heatmap\nsetting={setting}")
#     plt.tight_layout()
#     plt.savefig(f"heatmaps/consistency_heatmap_{setting_str}.png", dpi=300)
#     plt.close()

#     # Average accuracy bar chart
#     avg_acc = []
#     for model in models_in_group:
#         col = f"accuracy_{model}"
#         if col in group:
#             vals = group[col].dropna()
#             if not vals.empty:
#                 avg_acc.append((model, vals.mean()))
#     avg_acc_df = pd.DataFrame(avg_acc, columns=["Model", "Average Accuracy"])

#     plt.figure(figsize=(8, 3))
#     sns.barplot(x="Model", y="Average Accuracy", data=avg_acc_df)
#     plt.xticks(rotation=45, ha="right")
#     plt.ylim(0, 1)
#     plt.title(f"Average Accuracy per Model\nsetting={setting}")
#     plt.tight_layout()
#     plt.savefig(f"heatmaps/average_accuracy_{setting_str}.png", dpi=300)
#     plt.close()