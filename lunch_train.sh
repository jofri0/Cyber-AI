#!/usr/bin/env bash
accelerate launch train_hf.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --train_file train.jsonl \
  --validation_file val.jsonl \
  --output_dir ./out \
  --batch_size 1 \
  --num_train_epochs 3