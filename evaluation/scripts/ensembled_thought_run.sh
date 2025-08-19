#!/bin/bash

# Define models and their assigned CUDA devices
models=("Qwen/QwQ-32B" "BytedTsinghua-SIA/DAPO-Qwen-32B" "open-thoughts/OpenThinker-7B")
devices=(0 1 2)

# Create logs directory if it doesn't exist
mkdir -p logs

# Timestamp for log uniqueness
timestamp=$(date +"%Y%m%d_%H%M%S")

echo "🚀 Starting model runs at $timestamp"

# Loop through each model/device pair
for i in "${!models[@]}"; do
  model="${models[$i]}"
  device="${devices[$i]}"
  safe_model="${model//\//_}"
  logfile="logs/ensembled_${safe_model}_${timestamp}.log"

  echo "➡️  Launching model: $model on CUDA:$device (Log: $logfile)"

  CUDA_VISIBLE_DEVICES=$device \
  python run_custom_thoughts.py \
    --model "$model" \
    --prompt zero_shot \
    --thought_type ensembled_thought \
    --ensembled_file "ensemble_outputs/merged_output.csv" \
    > "$logfile" 2>&1 &

  # Optional: add sleep if model load is sensitive to startup timing
  # sleep 2
done

# Wait for all background jobs to finish
wait

echo "✅ All model runs completed at $(date +"%Y%m%d_%H%M%S")"
