#!/bin/bash

# For step 1
INPUT_GEN_PATH="data/Vivi_true_test.jsonl"
OUTPUT_GEN_PATH="gen2eval/T5_TISDPO.jsonl"
MODEL_DIR="checkpoint/T5_tisdpo_lora_new/checkpoint-3000"
DEVICE_LIST="6"
MAX_LENGTH=1024
TEMPERATURE=0.8
TOP_K=5
TOP_P=0.8
NUM_RESPONSE=1

# For step 2
QWEN_ENDPOINT="http://localhost:8005/v1/chat/completions"
QWEN_MODEL="Qwen/Qwen2.5-32B-Instruct"
EMBEDDING_MODEL="bkai-foundation-models/vietnamese-bi-encoder"
TEMPERATURE=0.6
MAX_TOKENS=2048
MAX_RETRIES=5
MAX_WAIT=30
TIMEOUT=600
MAX_WORKERS=4

# ------------------------------
# STEP 1: Generate sample
# ------------------------------
echo "Generate sample ..."
python gen_sample.py \
  --input_path "$INPUT_GEN_PATH" \
  --output_path "$OUTPUT_GEN_PATH" \
  --model_dir "$MODEL_DIR" \
  --DEVICE_LIST "$DEVICE_LIST" \
  --max_length "$MAX_LENGTH" \
  --temperature "$TEMPERATURE" \
  --top_k "$TOP_K" \
  --top_p "$TOP_P" \
  --num_response "$NUM_RESPONSE" 

# ------------------------------
# STEP 2: Evaluation
# ------------------------------
echo "Evaluate Ragas ..."
python eval_ragas.py \
  --file_eval_path "$OUTPUT_GEN_PATH" \
  --qwen_endpoint "$QWEN_ENDPOINT" \
  --qwen_model "$QWEN_MODEL" \
  --embedding_model_name "$EMBEDDING_MODEL" \
  --temperature "$TEMPERATURE" \
  --max_tokens "$MAX_TOKENS" \
  --max_retries "$MAX_RETRIES" \
  --max_wait "$MAX_WAIT" \
  --timeout "$TIMEOUT" \
  --max_workers "$MAX_WORKERS"

echo "Evaluate completed!"
