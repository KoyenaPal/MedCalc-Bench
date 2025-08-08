#!/bin/bash

# Define arrays of models and prompt styles
models=("Qwen/QwQ-32B" "BytedTsinghua-SIA/DAPO-Qwen-32B" "open-thoughts/OpenThinker-7B")
targetmodels=("open-thoughts/OpenThinker-7B" "BytedTsinghua-SIA/DAPO-Qwen-32B" "Qwen/QwQ-32B")

mkdir -p logs

# Optional delay between runs (in seconds)
delay=5

# Loop through all combinations
for model in "${models[@]}"; do
  for targetmodel in "${targetmodels[@]}"; do
    # Sanitize names for filenames
    safe_model="${model//\//_}"
    safe_targetmodel="${targetmodel//\//_}"
    outputpath="outputs/${safe_model}_zero_shot_original.jsonl"
    timestamp=$(date +"%Y%m%d_%H%M%S")
    logfile="logs/${safe_model}_to_${safe_targetmodel}_$timestamp.log"
    echo "Running: python run_transfer_thoughts.py --model $model --target_model $targetmodel --prompt zero_shot --source_model_output_file $outputpath"
    echo "Logging to: $logfile"
    
    # Run the command
    python run_transfer_thoughts.py --model $model --target_model $targetmodel --prompt zero_shot --source_model_output_file $outputpath > "$logfile" 2>&1

    # Check exit code
    if [ $? -ne 0 ]; then
      echo "❌ Error running: model=$model target_model=$targetmodel  (check $logfile)"
      # Uncomment next line to exit on error:
      # exit 1
    fi

    # Optional delay
    echo "Sleeping $delay seconds..."
    sleep $delay
  done
done

echo "✅ All runs completed."
