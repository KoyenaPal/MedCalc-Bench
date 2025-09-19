#!/bin/bash

# Script Name: sample_original_empty.sh

# Command 1
echo "Running Nemotron Ones..."
python run.py --model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --prompt zero_shot

python run_custom_thoughts.py --model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --prompt zero_shot --thought_type empty


# Command 2
echo "Running GPT..."
python run.py --model openai/gpt-oss-20b --prompt zero_shot --reasoning_effort low

python run_custom_thoughts.py --model openai/gpt-oss-20b --prompt zero_shot --thought_type empty --reasoning_effort low


# Command 3
echo "Running openthinker..."
python run.py --model open-thoughts/OpenThinker-7B --prompt zero_shot --reasoning_effort low

python run_custom_thoughts.py --model open-thoughts/OpenThinker-7B --prompt zero_shot --thought_type empty --reasoning_effort low


# Command 4
echo "Running Qwen/QwQ-32B..."
python run.py --model Qwen/QwQ-32B --prompt zero_shot --reasoning_effort low

python run_custom_thoughts.py --model Qwen/QwQ-32B --prompt zero_shot --thought_type empty --reasoning_effort low

# Command 5
echo "Running BytedTsinghua-SIA/DAPO-Qwen-32B..."
python run.py --model BytedTsinghua-SIA/DAPO-Qwen-32B --prompt zero_shot --reasoning_effort low

python run_custom_thoughts.py --model BytedTsinghua-SIA/DAPO-Qwen-32B --prompt zero_shot --thought_type empty --reasoning_effort low

echo "All commands executed."