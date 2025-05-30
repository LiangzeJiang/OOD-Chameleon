#!/bin/bash
# https://metashift.readthedocs.io/en/latest/sub_pages/download_MetaShift.html#appendix-d-constructing-metashift-from-coco-dataset
# Loadable shell file for MetaShift dataset

DATA_NAME="MetaShift"

# Models
MODEL=("resnet" "clip")

# Training epochs
EPOCHS=1000

# Algorithms
ALGORITHMS=("ERM" "GroupDRO" "remax-margin" "oversample" "undersample")

# Data sizes
DATA_SIZE=(200 500 1000)

# Task Y
TASK_Y=(-1)

# Task A
TASK_A=(-1)

# Spurious correlation values
SC=(
    # 3 shifts
    0.79 0.34 0.52 0.54 0.12 0.26 0.62 0.53 0.09 0.51
    0.64 0.31 0.28 0.40 0.30 0.80 0.45 0.12 0.17 0.71
    0.51 0.26 0.28 0.55 0.40 0.37 0.12 0.36 0.57 0.54
    # 1 shift
    0.42 0.77 0.66 0.85 0.19 0.32 0.21 0.02 0.94 0.58
    0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50
    0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50
)

# Label shift values
CI=(
    # 3 shifts
    0.46 0.12 0.50 0.10 0.65 0.46 0.78 0.29 0.20 0.43
    0.48 0.56 0.50 0.47 0.48 0.91 0.45 0.47 0.78 0.80
    0.55 0.49 0.67 0.50 0.13 0.26 0.91 0.72 0.10 0.67
    # 1 shift
    0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50
    0.39 0.70 0.55 0.95 0.72 0.85 0.11 0.29 0.47 0.09
    0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50
)

# Covariate shift values
AI=(
    # 3 shifts
    0.36 0.76 0.39 0.54 0.36 0.52 0.74 0.32 0.87 0.82
    0.71 0.58 0.36 0.84 0.67 0.79 0.20 0.47 0.22 0.59
    0.27 0.66 0.37 0.57 0.65 0.65 0.13 0.28 0.43 0.62
    # 1 shift
    0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50
    0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50
    0.57 0.63 0.81 0.06 0.34 0.12 0.30 0.45 0.97 0.76
)
