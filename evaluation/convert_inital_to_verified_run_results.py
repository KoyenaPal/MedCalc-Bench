import os
import tqdm
import pandas as pd
from evaluate import check_correctness
from table_stats import compute_overall_accuracy
import json


if __name__ == "__main__":

    initial_output_dir = "initial_version_outputs-original-full"
    verified_output_dir = "outputs-original-full"
    df = pd.read_csv("../dataset/test_data.csv")
    if not os.path.exists(verified_output_dir):
        os.makedirs(verified_output_dir)
    
    # Normalize df columns once at the start
    df["Calculator ID"] = df["Calculator ID"].astype(str)
    df["Note ID"] = df["Note ID"].astype(str)
    df["Patient Note"] = df["Patient Note"].astype(str)
    df["Question"] = df["Question"].astype(str)
    df["Ground Truth Answer"] = df["Ground Truth Answer"].astype(str)
    df["Ground Truth Explanation"] = df["Ground Truth Explanation"].astype(str)
    
    # for each file in the initial_output_dir, run the following code
    for file in os.listdir(initial_output_dir):
        missing_rows = []
        print(file, flush=True)
        if not file.endswith(".jsonl"):
            continue
        verified_output_filepath = os.path.join(verified_output_dir, file)
        existing = pd.read_json(os.path.join(initial_output_dir, file), lines=True)
        existing["Row Number"] = existing["Row Number"].astype(int)
        existing["Calculator ID"] = existing["Calculator ID"].astype(str)
        existing["Note ID"] = existing["Note ID"].astype(str)
        existing["Patient Note"] = existing["Patient Note"].astype(str)
        existing["Question"] = existing["Question"].astype(str)
        existing["LLM Answer"] = existing["LLM Answer"].astype(str)
        existing["LLM Explanation"] = existing["LLM Explanation"].astype(str)
        existing["Ground Truth Answer"] = existing["Ground Truth Answer"].astype(str)
        existing["Ground Truth Explanation"] = existing["Ground Truth Explanation"].astype(str)
        existing["Result"] = existing["Result"].astype(str)
        
        model_name = ""
        if "LLM Name" in existing.columns:
            model_name = str(existing["LLM Name"].iloc[0])
        if "LLM Model" in existing.columns:
            model_name = str(existing["LLM Model"].iloc[0])
        is_transfer_thoughts = False
        if "Target Model" in existing.columns:
            model_name = str(existing["Target Model"].iloc[0])
            is_transfer_thoughts = True
        if "Target Result" in existing.columns:
            existing["Target Result"] = existing["Target Result"].astype(str)
        if "Target Answer" in existing.columns:
            existing["Target Answer"] = existing["Target Answer"].astype(str)
        if "disk/u/koyena" in model_name.lower():
            model_name = model_name.replace("disk/u/koyena", "disk_u_koyena")

        for idx, existing_row in tqdm.tqdm(existing.iterrows(), total=len(existing)):
            # match by Calculator ID, Patient Note, and Question
            matching_rows = df[
                (df["Calculator ID"] == existing_row["Calculator ID"]) & 
                (df["Note ID"] == existing_row["Note ID"])
            ]
            # if more than one row, raise an error
            if len(matching_rows) > 1:
                raise ValueError(f"Multiple rows found for Row Number: {existing_row['Row Number']}")
            
            if matching_rows.empty:
                missing_rows.append(existing_row["Row Number"])
                continue
                # raise ValueError(f"Row not found for Row Number: {existing_row['Row Number']}")
            
            # Extract the single row as a Series
            row = matching_rows.iloc[0]
            # update Row Number value to the matching row's Row Number
            existing.loc[idx, "Row Number"] = row["Row Number"]
            # update ground truth answer and ground truth explanation
            existing.loc[idx, "Ground Truth Answer"] = row["Ground Truth Answer"]
            existing.loc[idx, "Ground Truth Explanation"] = row["Ground Truth Explanation"]
            answer_value = existing_row["LLM Answer"]
            calculator_id = existing_row["Calculator ID"]
            correctness = False
            try:
                correctness = check_correctness(
                    answer_value, 
                    row["Ground Truth Answer"],
                    calculator_id, 
                    row["Upper Limit"],
                    row["Lower Limit"]
                )
            except Exception as e:
                correctness = False
                existing.loc[idx, "LLM Answer"] = str(e)
                existing.loc[idx, "LLM Explanation"] = str(e)
            
            target_correctness = False
            try:
                if "Target Answer" in existing.columns:
                    target_answer_value = existing_row["Target Answer"]
                    target_calculator_id = existing_row["Calculator ID"]
                    target_correctness = check_correctness(
                        target_answer_value, 
                        row["Ground Truth Answer"], 
                        target_calculator_id, 
                        row["Upper Limit"], 
                        row["Lower Limit"]
                    )
            except Exception as e:
                target_correctness = False
                print(f"error in target correctness: {e}", flush=True)
                existing.loc[idx, "Target Answer"] = str(e)
                existing.loc[idx, "Target Explanation"] = str(e)
            
            existing.loc[idx, "Result"] = "Correct" if correctness else "Incorrect"
            
            if "Target Result" in existing.columns:
                existing.loc[idx, "Target Result"] = "Correct" if target_correctness else "Incorrect"

            
            
            # update to verified output file
            with open(verified_output_filepath, "a") as f:
                f.write(json.dumps(existing.loc[idx].to_dict()) + "\n")

        verified_result_filepath = "results-original-full/results_" + file.split(".")[0] + ".json"
        print("MISSING ROW INFO")
        print(missing_rows, flush=True)
        print(len(missing_rows), flush=True)
        compute_overall_accuracy(file, model_name, "zero_shot", output_dir="outputs-original-full", results_dir="results-original-full", is_target_model=is_transfer_thoughts, custom_path=verified_result_filepath)