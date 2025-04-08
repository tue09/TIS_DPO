#!/bin/bash

DATA_TRAIN_PATH="data/Vivi_true_train.jsonl"
OUTPUT_DIR_BASE="checkpoint"
DEVICE_LIST="6"

LORA_RANK=64
LORA_ALPHA=512
LORA_DROPOUT=0.1

BATCH_SIZE=4
LEARNING_RATE=1e-5
GRAD_ACC_STEPS=2
NUM_EPOCHS=8
LOGGING_STEPS=25
SCHEDULER_TYPE="linear"

POS_TRAIN_DATA_DIR="data/Vivi_true_train.jsonl"
NEG_TRAIN_DATA_DIR="data/Vivi_reverse_train.jsonl"
POS_MODEL_DIR="${OUTPUT_DIR_BASE}/T5_dpo_positive_lora_new"
CHECKPOINT_POS_MODEL="/checkpoint-3000"
NEG_MODEL_DIR="${OUTPUT_DIR_BASE}/T5_dpo_negative_lora_new"
CHECKPOINT_NEG_MODEL="/checkpoint-3000"
WEIGHTED_DATA_PATH="data/data_with_weights.jsonl"

TISDPO_MODEL_DIR="${OUTPUT_DIR_BASE}/T5_tisdpo_lora_new"

# ------------------------------
# STEP 1: Train initial DPO model
# ------------------------------
echo "Training positive model ..."
python train_DPO_hf.py \
  --type_model "positive" \
  --data_train_path "$POS_TRAIN_DATA_DIR" \
  --output_dir "$POS_MODEL_DIR" \
  --DEVICE_LIST "$DEVICE_LIST" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
  --num_train_epochs "$NUM_EPOCHS" \
  --logging_steps "$LOGGING_STEPS" \
  --lr_scheduler_type "$SCHEDULER_TYPE"

echo "Training negative model ..."
python train_DPO_hf.py \
  --type_model "negative" \
  --data_train_path "$NEG_TRAIN_DATA_DIR" \
  --output_dir "$NEG_MODEL_DIR" \
  --DEVICE_LIST "$DEVICE_LIST" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
  --num_train_epochs "$NUM_EPOCHS" \
  --logging_steps "$LOGGING_STEPS" \
  --lr_scheduler_type "$SCHEDULER_TYPE"

# ------------------------------
# STEP 2: Compute weights
# ------------------------------
echo "Running compute_weight.py..."
python compute_weight.py \
  --data_train_path "$DATA_TRAIN_PATH" \
  --positive_model_dir "$POS_MODEL_DIR$CHECKPOINT_POS_MODEL" \
  --negative_model_dir "$NEG_MODEL_DIR$CHECKPOINT_NEG_MODEL" \
  --output_dir "$WEIGHTED_DATA_PATH" \
  --DEVICE "$DEVICE_LIST"

# ------------------------------
# STEP 3: Train final TIS-DPO model
# ------------------------------
echo "Running train_TIS_DPO.py..."
python train_TIS_DPO.py \
  --data_train_path "$WEIGHTED_DATA_PATH" \
  --output_dir "$TISDPO_MODEL_DIR" \
  --DEVICE_LIST "$DEVICE_LIST" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
  --num_train_epochs "$NUM_EPOCHS" \
  --logging_steps "$LOGGING_STEPS" \
  --lr_scheduler_type "$SCHEDULER_TYPE"

echo "Training completed!"
