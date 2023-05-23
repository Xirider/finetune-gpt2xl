#!/bin/bash

# Define the list of dataset names
datasets=("dataset1" "dataset2" "dataset3")

# Loop through each dataset name
for idx in "${!datasets[@]}"; do
  dataset_name="${datasets[$idx]}"

  # Run the command with the dataset name
  deepspeed --num_gpus=1 run_clm.py \
  --deepspeed ds_config.json \
  --model_name_or_path gpt2-large \
  --dataset_name "$dataset_name" \
  --do_train \
  --do_eval \
  --fp16 \
  --overwrite_cache \
  --evaluation_strategy="steps" \
  --output_dir "gpt2-large_task$idx" \
  --eval_steps 200 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 2 \
  --per_device_train_batch_size 8

done
