# train_dpo.py
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
import os
import datetime

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--type_model', type=str, default='positive', required=False, help="Type of model: positive or negative")
parser.add_argument('--data_train_path', type=str, default='', required=False, help="Path to training data")
parser.add_argument('--output_dir', default='checkpoint/', type=str, required=False, help="Directory to save model checkpoints")
parser.add_argument('--DEVICE_LIST', default='6, 7', type=str, required=False, help="List Device to train")

parser.add_argument('--lora_rank', default=64, type=int, required=False)
parser.add_argument('--lora_alpha', default=512, type=int, required=False)
parser.add_argument('--lora_dropout', default=0.1, type=float, required=False)

parser.add_argument('--batch_size', default=4, type=int, required=False)
parser.add_argument('--learning_rate', default=1e-5, type=float, required=False)
parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False)
parser.add_argument('--num_train_epochs', default=10, type=int, required=False)
parser.add_argument('--logging_steps', default=50, type=int, required=False)
parser.add_argument('--lr_scheduler_type', default='linear', type=str, required=False)

args = parser.parse_args()

# --- Timestamp and CUDA ---
os.environ["CUDA_VISIBLE_DEVICES"] = args.DEVICE_LIST
# now = datetime.datetime.now()
# timestamp_start = now.strftime("%Y%m%d_%H%M%S")

print(f"Train for {args.type_model} model")
dataset = load_dataset('json', data_files=args.data_train_path, split='train')

# --- Dataset preprocessing ---
# dataset = dataset.rename_column('question', 'prompt')
def combine_prompt_and_context(example):
    # print(f"prompt = {example['prompt']}")
    example['prompt'] = '<Context>: ' + example['context'] + ' <prompt>: ' + example['prompt'] + ' <answer>:'
    return example
dataset = dataset.map(combine_prompt_and_context)

columns_to_keep = ["prompt", "chosen", "rejected"]
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
print(dataset.column_names)

# --- Model loading ---
# device = 'cuda:7'
model_dir = "T5"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=torch.float16)

# --- LoRA config ---
lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    target_modules=["q", "v", "k", "o"],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Training arguments ---
training_args = DPOConfig(
    output_dir=os.path.join(args.output_dir),
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs,
    gradient_checkpointing=True,
    lr_scheduler_type=args.lr_scheduler_type,
    logging_steps=args.logging_steps,
    save_strategy='steps',
    learning_rate=args.learning_rate,
    save_total_limit=3,
    save_steps=500
)

trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=dataset)
trainer.train()

