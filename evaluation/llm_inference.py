__author__ = "guangzhi"
'''
Adapted from https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/medrag.py
'''

import os
import re
import json
import tqdm
import torch
import time
import argparse
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import openai
import sys
from huggingface_hub import login

login(token=os.getenv("RUNPOD_HF_TOKEN"))

openai.api_key = os.getenv("OPENAI_API_KEY") 

import torch
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class LLMInference:

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo", cache_dir="/disk/u/koyena/hf"):
        self.llm_name = llm_name
        self.cache_dir = cache_dir
        self.thinking_start_tag = ""
        self.thinking_end_tag = ""
        if self.llm_name.split('/')[0].lower() == "openai" and "oss" not in self.llm_name.lower():
            self.model = self.llm_name.split('/')[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 4096
            elif "gpt-4" in self.model:
                self.max_length = 8192
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.type = torch.bfloat16
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir, legacy=False)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
            if "mixtral" in llm_name.lower() or "mistral" in llm_name.lower():
                self.tokenizer.chat_template = open('../templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.type = torch.float16
            elif "llama-3" in llm_name.lower():
                self.max_length = 8192
                self.thinking_end_tag = "</think>"
            elif "meditron-70b" in llm_name.lower():
                self.tokenizer.chat_template = open('../templates/meditron.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = open('../templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
            elif "qwen" in self.llm_name.lower() or "phi" in self.llm_name.lower():
                self.max_length = 32768
                self.thinking_start_tag = "<think>"
                self.thinking_end_tag = "</think>"
            elif "openthinker" in llm_name.lower():
                self.max_length = 32768
                self.thinking_start_tag = "<|begin_of_thought|>"
                self.thinking_end_tag = "<|end_of_thought|>"
            elif "gpt-oss-20b" in llm_name.lower():
                self.max_length = 32768
                #12288 only for transfer from llama-3 to oss
                #self.max_length = 12288
                self.thinking_start_tag = "assistantanalysis"
                self.thinking_end_tag = "assistantfinal"
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                dtype=self.type,
                device_map="auto",
                model_kwargs={"cache_dir":self.cache_dir},
            )

    def answer(self, messages, thinking_message="", do_sample=False):
        # generate answers
        ans = ""
        if thinking_message != "":
            print("CAME TO GENERATE WITH THINKING", flush=True)
            if thinking_message == "<empty>":
                thinking_message = ""
            ans = self.generate_with_thinking(messages, thinking_message=thinking_message, do_sample=do_sample)
        else:
            ans = self.generate(messages, do_sample=do_sample)
        ans = re.sub("\s+", " ", ans)
        
        return ans

    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria
    
    def generate_with_thinking(self, messages, thinking_message="", prompt=None, do_sample=False):
        '''
        generate response given messages and thinking message
        '''
        stopping_criteria = None
        temperature = 0.0
        old_prompt = None
        if do_sample:
            temperature = 1.0
        if prompt is None:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            old_prompt = prompt
            if "gpt-oss" in self.llm_name.lower():
                user_message = [{"role": "user", "content": messages[-1]["content"]}]
                prompt = self.tokenizer.apply_chat_template(user_message,
                                                            model_identity=messages[0]["content"],
                                                            reasoning_effort = messages[-1]["reasoning_effort"],
                                                            tokenize=False, add_generation_prompt=True)
                old_prompt = prompt
                
        if "qwen" in self.llm_name.lower() or "phi" in self.llm_name.lower():
            prompt = f"{prompt}<think>{thinking_message}</think>"
        elif "openthinker" in self.llm_name.lower():
            prompt = f"{prompt}<|begin_of_thought|>{thinking_message}<|end_of_thought|><|begin_of_solution|>"
        elif "gpt-oss" in self.llm_name.lower():
            prompt = f"{prompt}assistantanalysis{thinking_message}assistantfinal"
        # prompt = f"{prompt}<think>{thinking_message}</think>"
        print("FINAL PROMPT", prompt, flush=True)

        
        if "meditron" in self.llm_name.lower():
            stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))
        prompt_token_len = len(self.tokenizer.encode(prompt, add_special_tokens=True))
        old_prompt_token_len = len(self.tokenizer.encode(old_prompt, add_special_tokens=True))
        if "llama-3" in self.llm_name.lower():
            prompt = f"{prompt}<think>{thinking_message}</think>"
            response = self.model(
                prompt,
                do_sample=do_sample,
                eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=min(self.max_length, prompt_token_len + 100 + old_prompt_token_len + 4096),
                max_new_tokens=min(self.max_length, prompt_token_len + 100 + old_prompt_token_len + 4096),
                truncation=True,
                stopping_criteria=stopping_criteria,
                temperature=temperature
            )
            self.thinking_end_tag = "</think>"
        else:
            print("RESPONSE SECTION THINKING NOW", flush=True)
            response = self.model(
                prompt,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=min(self.max_length, prompt_token_len + 100 + old_prompt_token_len + 4096),
                max_new_tokens=min(self.max_length, prompt_token_len + 100 + old_prompt_token_len + 4096),
                truncation=True,
                stopping_criteria=stopping_criteria,
                temperature=temperature
            )
            
        ans = response[0]["generated_text"]
        return ans

    def generate(self, messages, prompt=None, do_sample=False):
        '''
        generate response given messages
        '''
        temperature = 0.0
        if do_sample:
            temperature = 1.0
        if "openai" in self.llm_name.lower() and "oss" not in self.llm_name.lower():
            response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages
            )

            ans = response.choices[0].message.content

        else:
            stopping_criteria = None
            if prompt is None:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if "gpt-oss" in self.llm_name.lower():
                    user_message = [{"role": "user", "content": messages[-1]["content"]}]
                    prompt = self.tokenizer.apply_chat_template(user_message,
                                                                model_identity=messages[0]["content"],
                                                                reasoning_effort = messages[-1]["reasoning_effort"],
                                                                tokenize=False, add_generation_prompt=True)
                if "qwen" in self.llm_name.lower() or "phi" in self.llm_name.lower():
                    if "dapo" in self.llm_name.lower():
                        prompt += "<think>"
                elif "openthinker" in self.llm_name.lower():
                    prompt += "<|begin_of_thought|>"
                elif "llama-3" in self.llm_name.lower():
                    prompt += "<think>"
            if "meditron" in self.llm_name.lower():
                stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))
            if "llama-3" in self.llm_name.lower():
                response = self.model(
                    prompt,
                    do_sample=do_sample,
                    eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=min(self.max_length, len(self.tokenizer.encode(prompt, add_special_tokens=True)) + 4096),
                    max_new_tokens=min(self.max_length, len(self.tokenizer.encode(prompt, add_special_tokens=True)) + 4096),
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    temperature=temperature
                )
                part_ans = response[0]["generated_text"]
                if self.thinking_end_tag not in part_ans:
                    part_ans = part_ans + self.thinking_end_tag
                if "openthinker" in self.llm_name.lower() and "<|begin_of_solution|>" not in part_ans:
                    part_ans = part_ans + "<|begin_of_solution|>"
                response = self.model(
                    part_ans,
                    do_sample=do_sample,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=min(self.max_length, len(self.tokenizer.encode(part_ans, add_special_tokens=True)) + 100),
                    max_new_tokens=min(self.max_length, len(self.tokenizer.encode(part_ans, add_special_tokens=True)) + 100),
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    temperature=temperature
                )
                
            else:
                # SETUP SEED
                print("RESPONSE SECTION NOW", flush=True)
                response = self.model(
                    prompt,
                    do_sample=do_sample,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=min(self.max_length, len(self.tokenizer.encode(prompt, add_special_tokens=True)) + 4096),
                    max_new_tokens=min(self.max_length, len(self.tokenizer.encode(prompt, add_special_tokens=True)) + 4096),
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    temperature=temperature
                )
                part_ans = response[0]["generated_text"]
                if self.thinking_end_tag not in part_ans:
                    part_ans = part_ans + self.thinking_end_tag
                if "openthinker" in self.llm_name.lower() and "<|begin_of_solution|>" not in part_ans:
                    part_ans = part_ans + "<|begin_of_solution|>"
                response = self.model(
                    part_ans,
                    do_sample=do_sample,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=min(self.max_length, len(self.tokenizer.encode(part_ans, add_special_tokens=True)) + 100),
                    max_new_tokens=min(self.max_length, len(self.tokenizer.encode(part_ans, add_special_tokens=True)) + 100),
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    temperature=temperature
                )
                
            ans = response[0]["generated_text"]
            print("ANSWER AFTER EDITING RESPONSE")
            print(ans, flush=True)
        return ans


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)    
