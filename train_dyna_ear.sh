#!/bin/bash

if test $# -ne 3; then
	echo "Usage: train_dyna_ear.sh <src_model_path> <output_dir> <training_dataset>"
	exit -1
fi

src_model_path=$1
output_dir=$2
training_dataset=$3

echo "Running with: ${src_model_path} ${output_dir} ${training_dataset}"

for ((i = 0 ; i < 10 ; i++)); do
    
    echo "Seed: $i"
    
    python ear_with_dyna.py \
        --src_model_path "${src_model_path}/entropybert-gab25k-${i}-0.01" \
        --output_dir ${output_dir} \
        --training_dataset ${training_dataset} \
        --max_epochs 20 \
        --batch_size 32 \
        --max_seq_length 128 \
        --gpus 1 \
        --num_workers 8 \
        --learning_rate 2e-5 \
        --early_stop_epochs 5 \
        --seed $i \
        --regularization entropy \
        --reg_strength 0.01 \
        --warmup_train_perc 0.1 \
        --weight_decay 0.01 \
        --save_transformers_model \
        --balanced_loss

done
