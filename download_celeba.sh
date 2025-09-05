#!/bin/bash

# ==============================================================================
# Script to download and prepare the CelebA dataset for PyTorch/Torchvision
#
# This script avoids the gdown rate-limit error by downloading from a
# reliable public mirror using wget. It creates the exact directory
# structure that torchvision expects.
#
# Usage:
#   1. Save this script as download_celeba.sh
#   2. Make it executable: chmod +x download_celeba.sh
#   3. Run it: ./download_celeba.sh
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# The root directory where your Python script expects the 'data' folder.
# This matches the "./data" used in your code.
DATA_ROOT="./data"
CELEBA_DIR="$DATA_ROOT/celeba"

# A reliable public mirror for the CelebA dataset files.
BASE_URL="https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba"

# List of files required by torchvision for the CelebA dataset.
FILES_TO_DOWNLOAD=(
  "img_align_celeba.zip"
  "list_attr_celeba.txt"
  "identity_CelebA.txt"
  "list_bbox_celeba.txt"
  "list_landmarks_align_celeba.txt"
)
# The evaluation partitions file is often in a different location or structure.
# We will download it separately and place it correctly.
EVAL_PARTITION_URL="https://raw.githubusercontent.com/pytorch/vision/main/references/public_data/celeba/list_eval_partition.txt"
EVAL_DIR="$CELEBA_DIR/eval"


# --- Script Execution ---
echo "======================================="
echo "  CelebA Dataset Setup Script"
echo "======================================="

# 1. Create the necessary directory structure
echo ""
echo ">>> Step 1: Creating directory structure at $CELEBA_DIR..."
mkdir -p "$CELEBA_DIR"
mkdir -p "$EVAL_DIR"
echo "    Done."

# 2. Navigate into the target directory for downloading
cd "$CELEBA_DIR"

# 3. Download the main dataset files
echo ""
echo ">>> Step 2: Downloading main CelebA dataset files..."
for file in "${FILES_TO_DOWNLOAD[@]}"; do
  if [ -f "$file" ]; then
    echo "    - $file already exists, skipping."
  else
    echo "    - Downloading $file..."
    # Use wget with a progress bar for a better user experience
    wget -q --show-progress "$BASE_URL/$file"
  fi
done
echo "    All main files are present."


# 4. Download the evaluation partition file
echo ""
echo ">>> Step 3: Downloading evaluation partition file..."
if [ -f "eval/list_eval_partition.txt" ]; then
    echo "    - eval/list_eval_partition.txt already exists, skipping."
else
    echo "    - Downloading list_eval_partition.txt..."
    wget -q --show-progress -O "eval/list_eval_partition.txt" "$EVAL_PARTITION_URL"
fi
echo "    Evaluation file is present."

# Return to the original directory
cd - > /dev/null

echo ""
echo "======================================="
echo -e "\e[32mSUCCESS: CelebA dataset is ready.\e[0m"
echo "The files are located in the '$CELEBA_DIR' directory."
echo "You can now run your Python script."
echo "Torchvision will detect these files and skip the automatic download."
echo "======================================="