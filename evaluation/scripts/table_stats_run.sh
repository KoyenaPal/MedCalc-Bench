#!/bin/bash

# Script Name: table_stats_run.sh

# Command 1
echo "Running Nemotron Ones..."
python table_stats.py --model_name nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --output_path nvidia_Nemotron-Research-Reasoning-Qwen-1.5B_zero_shot_empty.jsonl --additional_output_file_info "empty_full" --output_dir outputs-original-full --results_dir results-original-full

python table_stats.py --model_name nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --output_path nvidia_Nemotron-Research-Reasoning-Qwen-1.5B_zero_shot_original.jsonl --additional_output_file_info "original_full" --output_dir outputs-original-full --results_dir results-original-full


# Command 2
echo "Running GPT..."
python table_stats.py --model_name openai/gpt-oss-20b --output_path openai_gpt-oss-20b_zero_shot_empty.jsonl --additional_output_file_info "empty_full" --output_dir outputs-original-full --results_dir results-original-full

python table_stats.py --model_name openai/gpt-oss-20b --output_path openai_gpt-oss-20b_zero_shot_original_high.jsonl --additional_output_file_info "original_full_high" --output_dir outputs-original-full --results_dir results-original-full

python table_stats.py --model_name openai/gpt-oss-20b --output_path openai_gpt-oss-20b_zero_shot_original_medium.jsonl --additional_output_file_info "original_full_medium" --output_dir outputs-original-full --results_dir results-original-full

python table_stats.py --model_name openai/gpt-oss-20b --output_path openai_gpt-oss-20b_zero_shot_original_low.jsonl --additional_output_file_info "original_full_low" --output_dir outputs-original-full --results_dir results-original-full


# Command 3
echo "Running openthinker..."
python table_stats.py --model_name open-thoughts/OpenThinker-7B --output_path open-thoughts_OpenThinker-7B_zero_shot_empty.jsonl --additional_output_file_info "empty_full" --output_dir outputs-original-full --results_dir results-original-full

python table_stats.py --model_name open-thoughts/OpenThinker-7B --output_path open-thoughts_OpenThinker-7B_zero_shot_original.jsonl --additional_output_file_info "original_full" --output_dir outputs-original-full --results_dir results-original-full



# Command 4
echo "Running Qwen/QwQ-32B..."
python table_stats.py --model_name Qwen/QwQ-32B --output_path Qwen_QwQ-32B_zero_shot_empty.jsonl --additional_output_file_info "empty_full" --output_dir outputs-original-full --results_dir results-original-full

python table_stats.py --model_name Qwen/QwQ-32B --output_path Qwen_QwQ-32B_zero_shot_original.jsonl --additional_output_file_info "original_full" --output_dir outputs-original-full --results_dir results-original-full
# Command 5
echo "Running BytedTsinghua-SIA/DAPO-Qwen-32B..."
python table_stats.py --model_name BytedTsinghua-SIA/DAPO-Qwen-32B --output_path BytedTsinghua-SIA_DAPO-Qwen-32B_zero_shot_empty.jsonl --additional_output_file_info "empty_full" --output_dir outputs-original-full --results_dir results-original-full

python table_stats.py --model_name BytedTsinghua-SIA/DAPO-Qwen-32B --output_path BytedTsinghua-SIA_DAPO-Qwen-32B_zero_shot_original.jsonl --additional_output_file_info "original_full" --output_dir outputs-original-full --results_dir results-original-full

echo "All commands executed."