import json
import numpy as np
import random
import pandas as pd
import math
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import argparse
from torch import nn
import os
from transformers import BitsAndBytesConfig

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=False, help="input file path to gen sample")
parser.add_argument("--output_path", type=str, required=False, help="output file path")
parser.add_argument("--model_dir", type=str, required=False, help="model directory to gen sample")
parser.add_argument('--DEVICE_LIST', default='6', type=str, required=False, help="List Device to eval")
parser.add_argument("--max_length", type=int, required=False, default=1024, help="max_length")
parser.add_argument("--temperature", type=float, required=False, default=0.8, help="temperature")
parser.add_argument("--top_k", type=int, required=False, default=5, help="top k")
parser.add_argument("--top_p", type=float, required=False, default=0.9, help="top p")
parser.add_argument("--num_response", type=int, required=False, default=1, help="number response per each prompt")
args = parser.parse_args()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True
)

os.environ["CUDA_VISIBLE_DEVICES"] = args.DEVICE_LIST
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

#model_dir = "T5"
#model_dir = "checkpoint/T5_dpo_positive_lora_2/checkpoint-834"
#model_dir = "checkpoint/T5_dpo_positive_lora/20250402_044103/checkpoint-1500"
#model_dir = "checkpoint/T5_tis_dpo_lora/20250403_173204/checkpoint-1560"
model_dir = args.model_dir

print(f'use checkpoint = {model_dir}')
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir,
                                            #quantization_config=quantization_config,
                                            torch_dtype=torch.bfloat16,
                                            device_map="auto")
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

model = torch.compile(model)

def generate_text(prompt, max_length=args.max_length, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p):
    # t1 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # t2 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            # repetition_penalty=1.2
        )
    # t3 = time.time()
    # print(f'r1 = {t2 - t1}')
    # print(f'r1 = {t3 - t2}')
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


BATCH_SIZE = 50
NUM_RESPONSE_PER_PROMPT = 1

file_path = args.input_path
output_path = args.output_path
data_ = read_jsonl(file_path)
data_list = []

with open(output_path, "w", encoding="utf-8") as file:
    for idx in tqdm(range(len(data_)), desc=file_path):
        ground_truth = data_[idx]['ground_truth']
        full_prompt = f"<context>: {data_[idx]['context']}\n<prompt>: {data_[idx]['prompt']}\n <answer>:"
        prompt = data_[idx]['prompt']
        for idy in range(NUM_RESPONSE_PER_PROMPT):
            text = generate_text(f"{full_prompt}")
            data_list.append({'context': data_[idx]['context'], 'prompt': prompt, 'ground_truth': ground_truth, 'response': text})

            if len(data_list) >= BATCH_SIZE:
                for item in data_list:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
                file.flush()
                data_list = []

    if data_list:
        for item in data_list:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
        file.flush()

print(f"Done with {len(data_)} samples in {output_path}")

