#!/bin/bash

source "configs/$DATASET.sh"

# Some parameters are ignored if the dataset is officehome, see shifts_generator.py for details
if [ "$DATA_NAME" == "OfficeHome" ]; then
    python shifts_generator.py \
    --task_y "${TASK_Y[@]}" \
    --task_a "${TASK_A[@]}" \
    --data_name "$DATA_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"
else
    python shifts_generator.py \
    --data_size "${DATA_SIZE[@]}" \
    --sc "${SC[@]}" \
    --ci "${CI[@]}" \
    --ai "${AI[@]}" \
    --task_y "${TASK_Y[@]}" \
    --task_a "${TASK_A[@]}" \
    --data_name "$DATA_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"
fi


