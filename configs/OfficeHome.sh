#!/bin/bash
# https://www.hemanthdv.org/officeHomeDataset.html


source "configs/DATA_PATH.sh"

DATA_NAME="OfficeHome"

# Models
MODEL=("resnet" "clip")

# Training epochs
EPOCHS=1000

# Algorithms
ALGORITHMS=("ERM" "GroupDRO" "remax-margin" "oversample" "undersample")

# Task Y
TASK_Y=(-1)

# Task A
TASK_A=(-1)

DIR="$OUTPUT_DIR/officehome/tasks_y-1_a-1"
echo "DIR: $DIR"

# Initialize arrays
DATA_SIZE=()
SC=()  # Spurious correlation values
CI=()  # Label shift values
AI=()  # Covariate shift values

# Iterate over files in the directory
for n in $(ls "$DIR"); do
    # Split the filename by underscore
    IFS='_' read -r -a info <<< "$n"

    # Extract values
    data_size=${info[1]}
    sc=${info[2]:2}
    ci=${info[3]:2}
    ai=${info[4]:2}

    # Append to arrays
    DATA_SIZE+=("$data_size")
    SC+=("$sc")
    CI+=("$ci")
    AI+=("$ai")
done

echo "DATA_SIZE: ${DATA_SIZE[@]}"
echo "SC: ${SC[@]}"
echo "CI: ${CI[@]}"
echo "AI: ${AI[@]}"



