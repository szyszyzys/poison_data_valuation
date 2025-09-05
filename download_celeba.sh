#!/bin/bash

# ==============================================================================
# Script to download and prepare the CelebA dataset for PyTorch/Torchvision
#
# v2.0 - Updated with working links to fix 404 Not Found errors.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
DATA_ROOT="./data"
CELEBA_DIR="$DATA_ROOT/celeba"

# Correct, working URLs for the dataset files.
ZIP_URL="https://www.dropbox.com/s/d1kjpkqklf0uw77/img_align_celeba.zip?dl=1"
TXT_BASE_URL="https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/celeba_helpers"

FILES_TO_DOWNLOAD=(
  "img_align_celeba.zip"
  "list_attr_celeba.txt"
  "identity_CelebA.txt"
  "list_bbox_celeba.txt"
  "list_landmarks_align_celeba.txt"
)
EVAL_PARTITION_URL="https://raw.githubusercontent.com/pytorch/vision/main/references/public_data/celeba/list_eval_partition.txt"
EVAL_DIR="$CELEBA_DIR/eval"


# --- Script Execution ---
echo "======================================="
echo "  CelebA Dataset Setup Script (v2.0)"
echo "======================================="

# 1. Create directory structure
echo ""
echo ">>> Step 1: Creating directory structure at $CELEBA_DIR..."
mkdir -p "$CELEBA_DIR"
mkdir -p "$EVAL_DIR"
echo "    Done."

# 2. Navigate into the target directory
cd "$CELEBA_DIR"

# 3. Download the dataset files
echo ""
echo ">>> Step 2: Downloading main CelebA dataset files..."
for file in "${FILES_TO_DOWNLOAD[@]}"; do
  if [ -f "$file" ]; then
    echo "    - $file already exists, skipping."
  else
    echo "    - Downloading $file..."
    if [[ "$file" == *.zip ]]; then
      # Use the Dropbox URL for the zip file
      # Use -O to save the file with the correct name
      wget --show-progress -O "$file" "$ZIP_URL"
    else
      # Use the GitHub URL for the text files
      wget --show-progress "$TXT_BASE_URL/$file"
    fi
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
    wget --show-progress -O "eval/list_eval_partition.txt" "$EVAL_PARTITION_URL"
fi
echo "    Evaluation file is present."

# Return to the original directory
cd - > /dev/null

echo ""
echo "======================================="
echo -e "\e[32mSUCCESS: CelebA dataset is ready.\e[0m"
echo "You can now run your Python script."
echo "======================================="