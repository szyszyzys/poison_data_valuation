import json
import re
import sys
import torch
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

from entry.gradient_market.run_all_exp import setup_data_and_model

# --- IMPORTS (Adjust to match your structure) ---
try:
    from common.gradient_market_configs import AppConfig
    from entry.gradient_market.automate_exp.base_configs import get_base_image_config
except ImportError:
    print("⚠️  Error: Run this script from the project root.")
    sys.exit(1)


def parse_config_string(config_str: str, key: str, type_cast=str):
    """
    Extracts a value from the string representation in config_snapshot.json.
    Example: "DataConfig(..., buyer_ratio=0.1, ...)" -> extracts 0.1
    """
    # Regex to find key=value or key='value'
    pattern = f"{key}=(['\"]?)([^,'\"]+)(['\"]?)"
    match = re.search(pattern, config_str)
    if match:
        val = match.group(2)
        if type_cast == bool:
            return val == "True"
        return type_cast(val)
    return None


def inspect_run(run_dir: str, label: str):
    """Loads config from disk, recreates data, and analyzes buyer distribution."""
    print(f"\n{'=' * 60}")
    print(f"🔍 INSPECTING: {label}")
    print(f"Path: {run_dir}")
    print(f"{'=' * 60}")

    path = Path(run_dir) / "config_snapshot.json"
    if not path.exists():
        print("❌ Config snapshot not found!")
        return

    with open(path, 'r') as f:
        snapshot = json.load(f)

    # 1. Extract relevant strings
    data_str = snapshot.get("data", "")
    exp_str = snapshot.get("experiment", "")

    print(f"📄 Raw Data Config String: {data_str[:150]}...")

    # 2. Parse Key Parameters
    dataset = parse_config_string(exp_str, "dataset_name")
    strategy = parse_config_string(data_str, "strategy")
    alpha = parse_config_string(data_str, "dirichlet_alpha", float)

    # Buyer specific
    buyer_ratio = parse_config_string(data_str, "buyer_ratio", float)
    buyer_strategy = parse_config_string(data_str, "buyer_strategy")
    buyer_alpha = parse_config_string(data_str, "buyer_dirichlet_alpha", float)

    print(f"\n⚙️  Parsed Configuration:")
    print(f"   - Dataset: {dataset}")
    print(f"   - Buyer Ratio: {buyer_ratio}")
    print(f"   - Buyer Strategy: {buyer_strategy}")
    print(f"   - Buyer Alpha: {buyer_alpha}")

    # 3. Reconstruct AppConfig for Loading
    # We start with a base config and override with parsed values
    cfg = get_base_image_config()
    cfg.experiment.dataset_name = dataset
    # We assume image modality based on your previous context
    cfg.data.image.buyer_ratio = buyer_ratio
    cfg.data.image.buyer_strategy = buyer_strategy
    cfg.data.image.buyer_dirichlet_alpha = buyer_alpha
    cfg.data.image.strategy = strategy
    cfg.data.image.dirichlet_alpha = alpha

    # Important: Ensure cache is DISABLED to see the real generation logic
    cfg.use_cache = False

    # 4. Load Data
    print("\n🔄 Re-generating Data Split (No Cache)...")
    device = "cpu"
    try:
        # Calling your main setup function
        (buyer_loader, _, _, _, _, _, _, num_classes) = setup_data_and_model(cfg, device)
    except Exception as e:
        print(f"❌ Data Loading Failed: {e}")
        return

    # 5. Analyze Statistics
    if buyer_loader is None:
        print("❌ Buyer Loader is None!")
        return

    labels = []
    for _, y in buyer_loader:
        labels.extend(y.tolist())

    counts = Counter(labels)
    sorted_classes = sorted(counts.keys())
    missing_classes = set(range(num_classes)) - set(sorted_classes)

    values = list(counts.values())
    mean_count = np.mean(values) if values else 0
    std_dev = np.std(values) if values else 0

    print(f"\n📊 BUYER DATA STATISTICS:")
    print(f"   - Total Samples: {len(labels)}")
    print(f"   - Classes Present: {len(sorted_classes)} / {num_classes}")
    print(f"   - Missing Classes: {len(missing_classes)}")
    if len(missing_classes) > 0:
        print(f"     ⚠️  Missing: {list(missing_classes)[:10]}...")

    print(f"   - Samples per Class (Avg): {mean_count:.2f}")
    print(f"   - Imbalance (Std Dev): {std_dev:.2f}")

    # 6. Heuristic Verdict
    if len(missing_classes) > 0:
        print("\n🔴 VERDICT: ROOT DATA BROKEN (Missing Classes)")
        print("   FLTrust will assign score=0 to sellers who update missing classes.")
    elif std_dev > mean_count:
        print("\n🟠 VERDICT: HIGHLY UNBALANCED")
        print("   FLTrust might be unstable due to high variance in root gradient.")
    else:
        print("\n🟢 VERDICT: DATA LOOKS HEALTHY")


# --- RUNNER ---
if __name__ == "__main__":
    # REPLACE THESE PATHS WITH YOUR ACTUAL PATHS

    # 1. A path that worked (Step 2.5 or similar)
    GOOD_RUN = "results/step2.5_find_hps_fltrust_image_CIFAR100/ds-cifar100_model-cnn_agg-fltrust_k-0_clip-5_no_attack_seed-42/aggregation.fltrust.clip_norm_5.0/ds-cifar100_model-cnn_agg-fltrust_k-0_clip-5_no_attack_seed-42/run_0_seed_42"

    # 2. A path that failed (Step 11 Seller Only)
    BAD_RUN = "results/step11_seller_only_fltrust_CIFAR100/ds-cifar100_model-cnn_agg-fltrust_k-0_clip-5_no_attack_seed-42/aggregation.fltrust.clip_norm_5.0/ds-cifar100_model-cnn_agg-fltrust_k-0_clip-5_no_attack_seed-42/run_0_seed_42"

    # Check if paths exist before running (optional, just for safety)
    if Path(GOOD_RUN).exists():
        inspect_run(GOOD_RUN, "Step 2.5 (Baseline - Should be IID)")
    else:
        print(f"Skipping Good Run (Path not found: {GOOD_RUN})")

    if Path(BAD_RUN).exists():
        inspect_run(BAD_RUN, "Step 11 (Seller Only - Suspected Dirichlet Issue)")
    else:
        print(f"Skipping Bad Run (Path not found: {BAD_RUN})")