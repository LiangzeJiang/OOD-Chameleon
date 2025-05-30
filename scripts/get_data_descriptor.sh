#!/bin/bash

source "configs/$DATASET.sh"

for m in "${!MODEL[@]}"; do
    model="${MODEL[$m]}"

    python data_descriptor.py \
    --data_size "${DATA_SIZE[@]}" \
    --sc "${SC[@]}" \
    --ci "${CI[@]}" \
    --ai "${AI[@]}" \
    --task_y "${TASK_Y[@]}" \
    --task_a "${TASK_A[@]}" \
    --model "$model" \
    --data_name "$DATA_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --split "tr"

    python data_descriptor.py \
    --data_size "${DATA_SIZE[@]}" \
    --sc "${SC[@]}" \
    --ci "${CI[@]}" \
    --ai "${AI[@]}" \
    --task_y "${TASK_Y[@]}" \
    --task_a "${TASK_A[@]}" \
    --model "$model" \
    --data_name "$DATA_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --split "te"
done

