#!/bin/bash
# Usage: bash export_full_env.sh myenv

ENV_NAME=${1:-$(conda info --envs | awk '/\*/ {print $1}')}
EXPORT_FILE="environment_full.yml"

echo "🔹 Exporting full environment for: $ENV_NAME"

# Step 1: Export conda packages (no builds)
conda env export -n "$ENV_NAME" --no-builds | grep -v "prefix:" > tmp_conda.yml

# Step 2: Export pip packages
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
pip freeze | sort > tmp_pip.txt

# Step 3: Append pip packages under the correct pip section
echo "name: $ENV_NAME" > "$EXPORT_FILE"
echo "dependencies:" >> "$EXPORT_FILE"

# Extract all non-pip dependencies
awk '/- pip:/{exit} {print}' tmp_conda.yml | tail -n +3 >> "$EXPORT_FILE"

# Add pip section
echo "  - pip:" >> "$EXPORT_FILE"
awk '{print "    - " $0}' tmp_pip.txt >> "$EXPORT_FILE"

echo "✅ Environment exported to: $EXPORT_FILE"

# Cleanup
rm tmp_conda.yml tmp_pip.txt
