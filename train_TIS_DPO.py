import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
import json
import datetime
import argparse
import os
from trl.trainer.utils import DPODataCollatorWithPadding
from trl.trainer.utils import (
    flush_left,
    selective_log_softmax,
)
from trl.trainer.dpo_trainer import DataCollatorForPreference
from pynvml import *

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--data_train_path', default='', type=str, required=False, help="Path to training data")
parser.add_argument('--output_dir', default='checkpoint/', type=str, required=False, help="Directory to save model checkpoints")
parser.add_argument('--DEVICE_LIST', default='7', type=str, required=False, help="List Device to train")

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

# --- Dataset preprocessing ---
dataset = load_dataset('json', data_files=args.data_train_path, split='train')
# dataset = dataset.rename_column('question', 'prompt')
def combine_prompt_and_context(example):
    # print(f"prompt = {example['prompt']}")
    example['prompt'] = '<Context>: ' + example['context'] + ' <prompt>: ' + example['prompt'] + ' <answer>:'
    return example
dataset = dataset.map(combine_prompt_and_context)
print(dataset.column_names)

model_dir = "T5"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto") # = model to train

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

base_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")  # = reference model

class TISDPODataCollatorForPreference(DataCollatorForPreference):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        #print(f"features = {examples[0].keys()}")
        max_chosen_len = max(len(f['chosen_weights']) for f in examples)
        max_rejected_len = max(len(f['rejected_weights']) for f in examples)
        for f in examples:
            f['chosen_weights'] += [0.0] * (max_chosen_len - len(f['chosen_weights']))
            f['rejected_weights'] += [0.0] * (max_rejected_len - len(f['rejected_weights']))
        batch['chosen_weights'] = torch.tensor([f['chosen_weights'] for f in examples], dtype=torch.float)
        batch['rejected_weights'] = torch.tensor([f['rejected_weights'] for f in examples], dtype=torch.float)
        return batch

class TISDPOTrainer(DPOTrainer):
    def concatenated_forward(self, model, batch):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        # Add for token weights of chosen and rejected
        chosen_weights = batch["chosen_weights"]
        rejected_weights = batch["rejected_weights"]
        max_weight_len = max(chosen_weights.shape[1], rejected_weights.shape[1])
        chosen_weights_padded = torch.nn.functional.pad(chosen_weights, (0, max_weight_len - chosen_weights.shape[1]), value=0.0)
        rejected_weights_padded = torch.nn.functional.pad(rejected_weights, (0, max_weight_len - rejected_weights.shape[1]), value=0.0)
        concatenated_weights = torch.cat([chosen_weights_padded, rejected_weights_padded], dim=0)
        ###########
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        if self.is_encoder_decoder:
            labels = completion_input_ids.clone()
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:

            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            # Truncate right
            if self.max_length is not None:
                if self.truncation_mode == "keep_end":
                    input_ids = input_ids[:, -self.max_length:]
                    attention_mask = attention_mask[:, -self.max_length:]
                    loss_mask = loss_mask[:, -self.max_length:]
                elif self.truncation_mode == "keep_start":
                    input_ids = input_ids[:, :self.max_length]
                    attention_mask = attention_mask[:, :self.max_length]
                    loss_mask = loss_mask[:, :self.max_length]
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', 'keep_start']."
                    )
            if self.use_logits_to_keep:
                # Compute logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
                model_kwargs["logits_to_keep"] = logits_to_keep
            if self.padding_free:
                # Flatten the input_ids, position_ids, and loss_mask
                # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
                #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask
            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            # Offset the logits by one to align with the labels
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()
            if self.use_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels = labels.clone()
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        ######## ADD #######
        if concatenated_weights.shape[1] != per_token_logps.shape[1]:
            if concatenated_weights.shape[1] > per_token_logps.shape[1]:
                concatenated_weights = concatenated_weights[:, :per_token_logps.shape[1]]
            else:
                concatenated_weights = torch.nn.functional.pad(concatenated_weights, (0, per_token_logps.shape[1] - concatenated_weights.shape[1]), value=0.0)

        weighted_per_token_logps = per_token_logps * concatenated_weights
        all_logps = weighted_per_token_logps.sum(dim=1)
        ######## END ADD #######

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1) # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights_adj = all_weights[:num_examples]
                rejected_weights_adj = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights_adj + rejected_weights_adj), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=0,
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        return output

padding_value = 0
training_args = DPOConfig(
    output_dir=os.path.join(args.output_dir),
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs,
    gradient_checkpointing=True,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    logging_steps=args.logging_steps,
    save_strategy='steps',
    save_steps=600,
    beta=0.1,
    padding_value=0,
    remove_unused_columns=False,
)

trainer = TISDPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    #processing_class=processing,
    train_dataset=dataset,
    data_collator=TISDPODataCollatorForPreference(pad_token_id=padding_value),
)

trainer.train()
