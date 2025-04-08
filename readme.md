# TIS-DPO: Training & Evaluation Pipeline

This repository provides a complete pipeline for training and evaluating a **Token Importance Scaled Direct Preference Optimization (TIS-DPO)** model (following [TIS-DPO paper](https://arxiv.org/pdf/2410.04350)). The process includes training two baseline DPO models (positive and negative), computing token-level importance weights based on both models, and training the final TIS-DPO model using the computed weights. The evaluation phase involves generating samples using the final model and evaluating the output with the Ragas framework.

---

## Installation

Install all dependencies via `pip`:

```bash
pip install -r requirements.txt
```

---

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Training Pipeline](#training-pipeline)
  - [Step 1: Train Positive & Negative DPO Models](#step-1-train-positive--negative-dpo-models)
  - [Step 2: Compute Token Weights](#step-2-compute-token-weights)
  - [Step 3: Train Final TIS-DPO Model](#step-3-train-final-tis-dpo-model)
- [Evaluation Pipeline](#evaluation-pipeline)
  - [Step 1: Generate Samples](#step-1-generate-samples)
  - [Step 2: Evaluate with Ragas](#step-2-evaluate-with-ragas)
- [How to Run](#how-to-run)
- [Configurable Parameters](#configurable-parameters)
- [Contact & Support](#contact--support)

---

## Overview

This project implements the TIS-DPO training paradigm, which enhances preference modeling by computing token-level importance using both **positive** and **negative** DPO models:

- **Positive DPO Model:** Trained with (chosen, rejected) pairs.
- **Negative DPO Model:** Trained with (rejected, chosen) pairs.
- **Token Weights Computation:** Token-level importance is derived from the logit differences of the two models.
- **TIS-DPO Model:** Trained with token-weighted loss to better align with preference signals.
- **Evaluation:** Final model is tested on `[faithfulness, answer_correctness, context_precision, answer_relevancy context_recall]` via Ragas.

---

## Data Preparation

Make sure the following files exist in the `data/` directory:

- **Training Data:**
  - `Vivi_true_train.jsonl` — Contains data for positive DPO training.
  - `Vivi_reverse_train.jsonl` — Contains data pairs for negative DPO training.
- **Test Data:**
  - `Vivi_true_test.jsonl` — Contains data for evaluate model.
  
  _Note:_ Each line in these JSONL files must follow the standard format with the fields:  
  ```json
  {
    "context": "...",
    "prompt": "...",
    "ground_truth": "...",
    "chosen": "...",
    "rejected": "..."
  }

## Training Pipeline

Training is performed via `bash_train.sh` and includes three major steps:

### Step 1: Train Positive & Negative DPO Models

Trains two separate models:

- **Positive model** on (chosen, rejected) pairs
- **Negative model** on (rejected, chosen) pairs

Each uses LoRA fine-tuning with configurable hyperparameters.

### Step 2: Compute Token Weights

Using both trained models, token-level importance weights are computed and saved to a new training file. This step follows the TIS-DPO methodology, relying on per-token logit score differences.

### Step 3: Train Final TIS-DPO Model

The final model is trained using the weighted data. 

---

## Evaluation Pipeline

Evaluation is defined in `bash_eval.sh` and involves two key steps:

### Step 1: Generate Samples

The trained TIS-DPO model generates responses based on test prompts. The output is stored in a JSONL file for evaluation.

### Step 2: Evaluate with Ragas

The generated outputs are passed through a Ragas-based evaluation script.

---

## How to Run

### Training

To run the full training pipeline:

```bash
bash bash_train.sh
```

### Evaluation

To run the full evaluation pipeline:

```bash
bash bash_eval.sh
```
