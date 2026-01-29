#!/bin/bash

# Example usage scripts for the updated train.py

# ========================================
# Example 1: Regular training with config overrides
# ========================================
echo "Example 1: Training with config overrides"
python train.py \
    --config config.yaml \
    --train_data data/exp1/train_split.jsonl \
    --val_data data/exp1/val_split.jsonl \
    --extra_args learning_rate=0.001 train_batch_size=256 dropout_prob=0.3

# ========================================
# Example 2: Training with test set evaluation
# ========================================
echo "Example 2: Training with test set evaluation"
python train.py \
    --config config.yaml \
    --train_data data/exp1/train_split.jsonl \
    --val_data data/exp1/val_split.jsonl \
    --test_data data/exp1/test.jsonl

# ========================================
# Example 3: Evaluation only (no training)
# ========================================
echo "Example 3: Evaluation only on test set"
python train.py \
    --config config.yaml \
    --train_data data/exp1/train_split.jsonl \
    --val_data data/exp1/val_split.jsonl \
    --test_data data/exp1/test.jsonl \
    --checkpoint output/20260112_212022/best-mrr5-epoch=150-val_mrr@5=0.4500.ckpt \
    --eval_only

# ========================================
# Example 4: Resume training from checkpoint
# ========================================
echo "Example 4: Resume training from checkpoint"
python train.py \
    --config config.yaml \
    --train_data data/exp1/train_split.jsonl \
    --val_data data/exp1/val_split.jsonl \
    --checkpoint output/20260112_212022/last.ckpt

# ========================================
# Example 5: Advanced config overrides with nested keys
# ========================================
echo "Example 5: Multiple config overrides including nested keys"
python train.py \
    --config config.yaml \
    --train_data data/exp1/train_split.jsonl \
    --val_data data/exp1/val_split.jsonl \
    --extra_args \
        learning_rate=0.0005 \
        hidden_size=128 \
        num_layers=2 \
        epochs=100 \
        train_batch_size=1024 \
        eval_batch_size=2048 \
        weight_decay=0.01

# ========================================
# Example 6: All features combined
# ========================================
echo "Example 6: All features - config overrides, test set, checkpoint, eval only"
python train.py \
    --config config.yaml \
    --train_data data/exp1/train_split.jsonl \
    --val_data data/exp1/val_split.jsonl \
    --test_data data/exp1/test.jsonl \
    --checkpoint output/20260112_212022/best-mrr5-epoch=150-val_mrr@5=0.4500.ckpt \
    --eval_only \
    --extra_args eval_batch_size=512
