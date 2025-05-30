#!/bin/bash

source "configs/DATA_PATH.sh"
export DATA_DIR
export OUTPUT_DIR

DATASETS=("CelebA" "MetaShift" "CMNIST" "OfficeHome" "MultiNLI" "CivilComments")
for DATASET in "${DATASETS[@]}"; do
    export DATASET

    # 1. generate datasets with diverse distribution shifts.
    if [ "$DATASET" != "CMNIST" ]; then # skip step 1 for cmnist since it is handled on the fly
        bash scripts/generate_data_pool.sh
    fi
    # 2. get the data descriptor for each generated dataset.
    bash scripts/get_data_descriptor.sh
    # 3. get the performance of each algorithm on each dataset.
    bash scripts/run_algorithm_zoo.sh
    # 4. assemble the meta-dataset and train the algorithm selector.
    # (see ../train_chameleon.ipynb)
done