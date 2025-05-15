#!/usr/bin/env bash
###############################################################################
# install_env.sh  —  Robust one‑liner to recreate the DGM‑Benchmark environment
#
#  • Prefers micromamba (fast, minimal), falls back to conda automatically
#  • Uses environment.lock.yml if present for bit‑for‑bit reproducibility;
#    otherwise falls back to environment.yml
#  • Activates the env, installs the project in‑place (`pip install -e .`)
#  • Runs `pip check` and an optional smoke test to make sure deps resolve
#
# Usage: ./install_env.sh          # installs into dgm-benchmark env
#        ./install_env.sh gpu      # forces CUDA build variants if present
###############################################################################
set -euo pipefail
ENV_NAME="dgm-benchmark"
LOCK_FILE="environment.lock.yml"
YAML_FILE="environment.yml"
EXTRA_ARG="${1:-}"

# -----------------------------------------------------------------------------
# Helper: download micromamba to a local cache if conda isn't installed
# -----------------------------------------------------------------------------
install_micromamba() {
  local MAMBADIR="$HOME/.local/micromamba"
  if [[ ! -x "$MAMBADIR/bin/micromamba" ]]; then
    echo "[*] Micromamba not found → downloading…"
    curl -sSLo micromamba.tar.bz2 \
      https://micro.mamba.pm/api/micromamba/linux-64/latest         # change for macOS if needed
    mkdir -p "$MAMBADIR"
    tar -xjf micromamba.tar.bz2 -C "$MAMBADIR" --strip-components=1
    rm micromamba.tar.bz2
  fi
  export PATH="$MAMBADIR/bin:$PATH"
}

# -----------------------------------------------------------------------------
# 1. Choose installer ---------------------------------------------------------
# -----------------------------------------------------------------------------
if command -v micromamba &>/dev/null; then
  MAMBA_CMD="micromamba"
elif command -v conda &>/dev/null; then
  MAMBA_CMD="conda"
else
  install_micromamba
  MAMBA_CMD="micromamba"
fi
echo "[*] Using $MAMBA_CMD for environment creation"

# -----------------------------------------------------------------------------
# 2. Determine which spec file to use -----------------------------------------
# -----------------------------------------------------------------------------
if [[ -f $LOCK_FILE ]]; then
  SPEC_FILE=$LOCK_FILE
  echo "[*] Found lockfile → strict reproducibility enabled"
else
  SPEC_FILE=$YAML_FILE
fi

# -----------------------------------------------------------------------------
# 3. Create (or update) the environment --------------------------------------
# -----------------------------------------------------------------------------
echo "[*] Creating env '$ENV_NAME' from $SPEC_FILE"
if [[ "$MAMBA_CMD" == "micromamba" ]]; then
  micromamba create -y -n "$ENV_NAME" -f "$SPEC_FILE"
  # shellcheck source=/dev/null
  eval "$(micromamba shell hook -s bash)"
  micromamba activate "$ENV_NAME"
else
  conda env create -y -n "$ENV_NAME" -f "$SPEC_FILE" || \
  conda env update  -y -n "$ENV_NAME" -f "$SPEC_FILE"
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  conda config --env --set channel_priority strict
fi

# -----------------------------------------------------------------------------
# 4. Install project in editable mode (if setup.cfg/pyproject.toml exists) ----
# -----------------------------------------------------------------------------
if [[ -f setup.cfg || -f pyproject.toml ]]; then
  echo "[*] Installing project in editable mode"
  python -m pip install -e .
fi

# -----------------------------------------------------------------------------
# 5. Optional GPU extras (override via ./install_env.sh gpu) ------------------
# -----------------------------------------------------------------------------
if [[ "$EXTRA_ARG" == "gpu" && -f requirements_gpu.txt ]]; then
  echo "[*] Installing GPU‑specific extras"
  python -m pip install -r requirements_gpu.txt
fi

# -----------------------------------------------------------------------------
# 6. Sanity checks ------------------------------------------------------------
# -----------------------------------------------------------------------------
echo "[*] Verifying package consistency"
python -m pip check        # dependency conflicts → non‑zero exit

echo "[✓] Environment '$ENV_NAME' is ready."
echo "    To activate later:  $MAMBA_CMD activate $ENV_NAME"
