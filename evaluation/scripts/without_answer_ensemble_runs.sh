#!/bin/bash

# Script Name: my_script.sh
# Description: This script runs 7 hard-coded commands sequentially.

# Command 1
echo "Running Nemotron Ones..."
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.csv --model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B  --thought_type ensembled_thought --reasoning_effort low > logs/nemotron-ens
medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.csv --model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/nemotron-medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss_without_answer.txt

# Command 2
echo "Running GPT..."
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.csv --model openai/gpt-oss-20b  --thought_type ensembled_thought --reasoning_effort low > logs/gpt-medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.csv --model openai/gpt-oss-20b  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/gpt-medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss_without_answer.txt

# Command 3
echo "Running openthinker..."
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.csv --model open-thoughts/OpenThinker-7B  --thought_type ensembled_thought --reasoning_effort low > logs/openthinker-ens-medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.csv --model open-thoughts/OpenThinker-7B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/openthinker-medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss_without_answer.txt

# Command 4
echo "Running Qwen/QwQ-32B..."
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.csv --model Qwen/QwQ-32B  --thought_type ensembled_thought --reasoning_effort low > logs/qwq-ens-medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.csv --model Qwen/QwQ-32B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/qwq-medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss_without_answer.txt

# Command 5
echo "Running BytedTsinghua-SIA/DAPO-Qwen-32B..."
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.csv --model BytedTsinghua-SIA/DAPO-Qwen-32B  --thought_type ensembled_thought --reasoning_effort low > logs/dapo-ens-medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss.csv --model BytedTsinghua-SIA/DAPO-Qwen-32B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/dapo-medcalc_ensemble_outputs_gen_qwq_openthinker_eval_gpt_oss_without_answer.txt

echo "All commands executed."