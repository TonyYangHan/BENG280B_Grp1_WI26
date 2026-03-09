#!/usr/bin/env bash
set -euo pipefail


DATASET_DIR="../ISLES-2022-npz-multimodal_clean/"
OUTPUT_DIR="result/"
CONFIG_FILE="./config.yaml"


python train_advanced.py  "${DATASET_DIR}" "${OUTPUT_DIR}"
