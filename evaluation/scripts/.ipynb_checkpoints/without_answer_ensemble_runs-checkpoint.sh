#!/bin/bash

# Script Name: my_script.sh
# Description: This script runs hard-coded commands sequentially.

# Command 1
echo "Running Nemotron Ones..."

# medcalc_ensemble_gen_qwq_dapo_eval_oss
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_dapo_eval_oss.csv --model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B  --thought_type ensembled_thought --reasoning_effort low > logs/nrr_medcalc_ensemble_gen_qwq_dapo_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_dapo_eval_oss.csv --model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/nrr_without_answer_medcalc_ensemble_gen_qwq_dapo_eval_oss.txt

# medcalc_ensemble_gen_qwq_opent_eval_oss
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_opent_eval_oss.csv --model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B  --thought_type ensembled_thought --reasoning_effort low > logs/nrr_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_opent_eval_oss.csv --model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/nrr_without_answer_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

#medcalc_ensemble_gen_qwq_oss_eval_dapo
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_oss_eval_dapo.csv --model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B  --thought_type ensembled_thought --reasoning_effort low > logs/nrr_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_oss_eval_dapo.csv --model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/nrr_without_answer_medcalc_ensemble_gen_qwq_oss_eval_dapo.txt

# Command 2
echo "Running GPT..."
# medcalc_ensemble_gen_qwq_dapo_eval_oss
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_dapo_eval_oss.csv --model openai/gpt-oss-20b  --thought_type ensembled_thought --reasoning_effort low > logs/oss_medcalc_ensemble_gen_qwq_dapo_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_dapo_eval_oss.csv --model openai/gpt-oss-20b  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/oss_without_answer_medcalc_ensemble_gen_qwq_dapo_eval_oss.txt

# medcalc_ensemble_gen_qwq_opent_eval_oss
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_opent_eval_oss.csv --model openai/gpt-oss-20b  --thought_type ensembled_thought --reasoning_effort low > logs/oss_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_opent_eval_oss.csv --model openai/gpt-oss-20b  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/oss_without_answer_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

#medcalc_ensemble_gen_qwq_oss_eval_dapo
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_oss_eval_dapo.csv --model openai/gpt-oss-20b  --thought_type ensembled_thought --reasoning_effort low > logs/oss_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_oss_eval_dapo.csv --model openai/gpt-oss-20b  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/oss_without_answer_medcalc_ensemble_gen_qwq_oss_eval_dapo.txt

# Command 3
echo "Running openthinker..."

# medcalc_ensemble_gen_qwq_dapo_eval_oss
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_dapo_eval_oss.csv --model open-thoughts/OpenThinker-7B --thought_type ensembled_thought --reasoning_effort low > logs/opent_medcalc_ensemble_gen_qwq_dapo_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_dapo_eval_oss.csv --model open-thoughts/OpenThinker-7B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/opent_without_answer_medcalc_ensemble_gen_qwq_dapo_eval_oss.txt

# medcalc_ensemble_gen_qwq_opent_eval_oss
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_opent_eval_oss.csv --model open-thoughts/OpenThinker-7B  --thought_type ensembled_thought --reasoning_effort low > logs/opent_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_opent_eval_oss.csv --model open-thoughts/OpenThinker-7B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/opent_without_answer_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

#medcalc_ensemble_gen_qwq_oss_eval_dapo
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_oss_eval_dapo.csv --model open-thoughts/OpenThinker-7B  --thought_type ensembled_thought --reasoning_effort low > logs/opent_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_oss_eval_dapo.csv --model open-thoughts/OpenThinker-7B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/opent_without_answer_medcalc_ensemble_gen_qwq_oss_eval_dapo.txt

# Command 4
echo "Running Qwen/QwQ-32B..."

# medcalc_ensemble_gen_qwq_dapo_eval_oss
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_dapo_eval_oss.csv --model Qwen/QwQ-32B --thought_type ensembled_thought --reasoning_effort low > logs/qwq_medcalc_ensemble_gen_qwq_dapo_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_dapo_eval_oss.csv --model Qwen/QwQ-32B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/qwq_without_answer_medcalc_ensemble_gen_qwq_dapo_eval_oss.txt

# medcalc_ensemble_gen_qwq_opent_eval_oss
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_opent_eval_oss.csv --model Qwen/QwQ-32B  --thought_type ensembled_thought --reasoning_effort low > logs/qwq_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_opent_eval_oss.csv --model Qwen/QwQ-32B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/qwq_without_answer_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

#medcalc_ensemble_gen_qwq_oss_eval_dapo
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_oss_eval_dapo.csv --model Qwen/QwQ-32B  --thought_type ensembled_thought --reasoning_effort low > logs/qwq_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_oss_eval_dapo.csv --model Qwen/QwQ-32B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/qwq_without_answer_medcalc_ensemble_gen_qwq_oss_eval_dapo.txt

# Command 5
echo "Running BytedTsinghua-SIA/DAPO-Qwen-32B..."

# medcalc_ensemble_gen_qwq_dapo_eval_oss
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_dapo_eval_oss.csv --model BytedTsinghua-SIA/DAPO-Qwen-32B --thought_type ensembled_thought --reasoning_effort low > logs/dapo_medcalc_ensemble_gen_qwq_dapo_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_dapo_eval_oss.csv --model BytedTsinghua-SIA/DAPO-Qwen-32B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/dapo_without_answer_medcalc_ensemble_gen_qwq_dapo_eval_oss.txt

# medcalc_ensemble_gen_qwq_opent_eval_oss
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_opent_eval_oss.csv --model BytedTsinghua-SIA/DAPO-Qwen-32B  --thought_type ensembled_thought --reasoning_effort low > logs/dapo_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_opent_eval_oss.csv --model BytedTsinghua-SIA/DAPO-Qwen-32B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/dapo_without_answer_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

#medcalc_ensemble_gen_qwq_oss_eval_dapo
python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_oss_eval_dapo.csv --model BytedTsinghua-SIA/DAPO-Qwen-32B  --thought_type ensembled_thought --reasoning_effort low > logs/dapo_medcalc_ensemble_gen_qwq_opent_eval_oss.txt

python run_custom_thoughts.py --prompt zero_shot --ensembled_file without_answer/medcalc_ensemble_gen_qwq_oss_eval_dapo.csv --model BytedTsinghua-SIA/DAPO-Qwen-32B  --thought_type ensembled_thought_without_answer --reasoning_effort low > logs/dapo_without_answer_medcalc_ensemble_gen_qwq_oss_eval_dapo.txt

echo "All commands executed."