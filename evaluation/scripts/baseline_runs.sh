#!/bin/bash

# Define arrays of models and prompt styles
#models=("Qwen/QwQ-32B" "BytedTsinghua-SIA/DAPO-Qwen-32B" "open-thoughts/OpenThinker-7B")
#models=("open-thoughts/OpenThinker-7B" "BytedTsinghua-SIA/DAPO-Qwen-32B" "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B" "Qwen/QwQ-32B")
#models=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "~/medcalc-deepseek-r1-distill-qwen-1.5b-inst-grpo")
#models=("/disk/u/koyena/medcalc-llama-3-3b-inst-grpo")
#models=("Qwen/QwQ-32B" "open-thoughts/OpenThinker-7B" "BytedTsinghua-SIA/DAPO-Qwen-32B" "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B")
#models=("/disk/u/koyena/medcalc-deepseek-r1-distill-qwen-1.5b-inst-grpo" "/disk/u/koyena/medcalc-llama-3-3b-inst-grpo" "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "meta-llama/Llama-3.2-3B-Instruct")
models=("meta-llama/Llama-3.2-3B-Instruct")
#models=("/disk/u/koyena/medcalc-qwen-2.5-1.5b-inst-grpo" "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "meta-llama/Llama-3.2-3B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct")
#targetmodels=("nvidia/Nemotron-Research-Reasoning-Qwen-1.5B" "open-thoughts/OpenThinker-7B" "openai/gpt-oss-20b" "BytedTsinghua-SIA/DAPO-Qwen-32B" "Qwen/QwQ-32B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "/medcalc-deepseek-r1-distill-qwen-1.5b-inst-grpo")
targetmodels=("openai/gpt-oss-20b")

# reasoningeffort= ("low" "medium" "high")
mkdir -p logs

# Optional delay between runs (in seconds)
delay=5

# Loop through all combinations
for model in "${models[@]}"; do
  for targetmodel in "${targetmodels[@]}"; do
    # Sanitize names for filenames
    safe_model="${model//\//_}"
    safe_targetmodel="${targetmodel//\//_}"
    outputpath="without_answer/${safe_model}_zero_shot_original_without_answer.jsonl"
    timestamp=$(date +"%Y%m%d_%H%M%S")
    logfile="logs/${safe_model}_to_${safe_targetmodel}_$timestamp.log"
    echo "Running: python run_transfer_thoughts.py --model $model --target_model $targetmodel --prompt zero_shot --source_model_output_file $outputpath --reasoning_effort low"
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
