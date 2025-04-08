import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
import json
import os
import argparse
from pynvml import *

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--data_train_path', default='', type=str, required=False, help="Path to training data")
parser.add_argument('--positive_model_dir', default='data/...', type=str, required=False, help="Directory to positive model checkpoints")
parser.add_argument('--negative_model_dir', default='data/...', type=str, required=False, help="Directory to negative model checkpoints")
parser.add_argument('--output_dir', default='data/data_with_weights.jsonl', type=str, required=False, help="Directory to save model checkpoints")
parser.add_argument('--DEVICE_LIST', default='6', type=str, required=False, help="List Device to compute weight")

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.DEVICE_LIST

dataset = load_dataset('json', data_files=args.data_train_path, split='train')
# dataset = dataset.rename_column('question', 'prompt')

def combine_prompt_and_context(example):
    # example['prompt'] = example['context'] + '. ' + example['prompt']
    return example

updated_dataset = dataset.map(combine_prompt_and_context)

columns_to_keep = ["context", "prompt", "chosen", "rejected"]
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
print(dataset.column_names)
print(f'type = {type(dataset)}')

device = f'cuda:{args.DEVICE_LIST[0]}'
model_dir = "T5"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
# Load positive and negative models for weight estimation
positive_model = AutoModelForSeq2SeqLM.from_pretrained(args.positive_model_dir, torch_dtype=torch.float16, device_map="auto")
negative_model = AutoModelForSeq2SeqLM.from_pretrained(args.negative_model_dir, torch_dtype=torch.float16, device_map="auto")


# Function to compute token-level importance weights
def compute_sample_weights(model_positive, model_negative, tokenizer, context, prompt, chosen, rejected, L=-0.5, U=1.5):
    device = model_positive.device
    input_ids = tokenizer('<Context>: ' + context + ' <prompt>: ' + prompt + ' <answer>:', return_tensors='pt').input_ids.to(device)

    # Compute weights for chosen response
    chosen_tokens = tokenizer(chosen, return_tensors='pt').input_ids.to(device)
    decoder_input_ids_chosen = model_positive._shift_right(chosen_tokens)  # Shift for decoder input
    with torch.no_grad():
        outputs_positive_chosen = model_positive(input_ids=input_ids, decoder_input_ids=decoder_input_ids_chosen)
        log_probs_positive_chosen = F.log_softmax(outputs_positive_chosen.logits, dim=-1)
        per_token_logps_positive_chosen = log_probs_positive_chosen.gather(2, chosen_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)[0]

        outputs_negative_chosen = model_negative(input_ids=input_ids, decoder_input_ids=decoder_input_ids_chosen)
        log_probs_negative_chosen = F.log_softmax(outputs_negative_chosen.logits, dim=-1)
        per_token_logps_negative_chosen = log_probs_negative_chosen.gather(2, chosen_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    reward_est_chosen = torch.clamp(per_token_logps_positive_chosen - per_token_logps_negative_chosen, min=L, max=U)
    chosen_weights = torch.exp(reward_est_chosen).cpu().tolist()  # μ = 1 for chosen

    # Compute weights for rejected response
    rejected_tokens = tokenizer(rejected, return_tensors='pt').input_ids.to(device)
    decoder_input_ids_rejected = model_positive._shift_right(rejected_tokens)
    with torch.no_grad():
        outputs_positive_rejected = model_positive(input_ids=input_ids, decoder_input_ids=decoder_input_ids_rejected)
        log_probs_positive_rejected = F.log_softmax(outputs_positive_rejected.logits, dim=-1)
        per_token_logps_positive_rejected = log_probs_positive_rejected.gather(2, rejected_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)[0]

        outputs_negative_rejected = model_negative(input_ids=input_ids, decoder_input_ids=decoder_input_ids_rejected)
        log_probs_negative_rejected = F.log_softmax(outputs_negative_rejected.logits, dim=-1)
        per_token_logps_negative_rejected = log_probs_negative_rejected.gather(2, rejected_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)[0]

    reward_est_rejected = torch.clamp(per_token_logps_positive_rejected - per_token_logps_negative_rejected, min=L, max=U)
    rejected_weights = torch.exp(-reward_est_rejected).cpu().tolist()  # μ = -1 for rejected

    return {'chosen_weights': chosen_weights, 'rejected_weights': rejected_weights}

dataset_with_weights = dataset.map(
    lambda x: compute_sample_weights(positive_model, negative_model, tokenizer, x['context'], x['prompt'], x['chosen'], x['rejected'])
)
print(f"Dataset with weights columns: {dataset_with_weights.column_names}")

with open(args.output_dir, "w", encoding="utf-8") as f:
    for item in dataset_with_weights:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print("✅ Saved to .jsonl")
