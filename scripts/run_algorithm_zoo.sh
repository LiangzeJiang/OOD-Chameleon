#!/bin/bash

source "configs/$DATASET.sh"

CLASSIFIERS=("linear" "mlp")

for CLASSIFIER in "${CLASSIFIERS[@]}"; do
    for m in "${!MODEL[@]}"; do
        model="${MODEL[$m]}"

        python train.py \
        --data_size "${DATA_SIZE[@]}" \
        --sc "${SC[@]}" \
        --ci "${CI[@]}" \
        --ai "${AI[@]}" \
        --task_y "${TASK_Y[@]}" \
        --task_a "${TASK_A[@]}" \
        --algorithm "${ALGORITHMS[@]}" \
        --model "$model" \
        --data_name "$DATA_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --classifier "$CLASSIFIER" \
        --num_epochs "$EPOCHS" \
        --seed 0  # better run multiple seeds

    done
done
