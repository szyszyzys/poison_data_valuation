import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from entry.gradient_market.run_all_exp import setup_data_and_model

# --- IMPORTS ---
try:
    # Add project root to sys.path to ensure imports work
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

    from marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig
    from entry.gradient_market.automate_exp.base_configs import get_base_image_config
except ImportError:
    print("⚠️  Error: Could not import project modules.")
    print(f"   Ensure you are running from project root.")
    sys.exit(1)


def parse_config_string(config_str: str, key: str, type_cast=str):
    """Extracts value from string representation."""
    pattern = f"{key}=(['\"]?)([^,'\"]+)(['\"]?)"
    match = re.search(pattern, config_str)
    if match:
        val = match.group(2)
        if type_cast == bool: return val == "True"
        return type_cast(val)
    return None


def find_and_inspect(base_scenario_dir: str, alpha_folder: str):
    """
    Locates the config_snapshot.json inside the nested folder structure
    and runs the inspection.
    """
    print(f"\n{'=' * 80}")
    print(f"🔍 SEARCHING IN: {base_scenario_dir} / {alpha_folder}")

    start_path = Path(base_scenario_dir) / alpha_folder

    if not start_path.exists():
        print(f"❌ Path not found: {start_path}")
        print(f"   Check if 'results/step11_...' exists.")
        return

    # Find the config file recursively
    config_files = list(start_path.rglob("config_snapshot.json"))

    if not config_files:
        print("❌ No config_snapshot.json found in this directory!")
        return

    # Use the first one found
    target_config = config_files[0]

    # --- FIX: Use resolve() to handle absolute/relative mismatch ---
    try:
        display_path = target_config.resolve().relative_to(Path.cwd())
    except ValueError:
        # Fallback if paths don't overlap or other issues
        display_path = target_config

    print(f"✅ Found config: {display_path}")

    inspect_file(target_config)


def inspect_file(config_path: Path):
    with open(config_path, 'r') as f:
        snapshot = json.load(f)

    data_str = snapshot.get("data", "")
    exp_str = snapshot.get("experiment", "")

    # Parse
    dataset = parse_config_string(exp_str, "dataset_name")
    strategy = parse_config_string(data_str, "strategy")
    alpha = parse_config_string(data_str, "dirichlet_alpha", float)

    # Buyer specific
    buyer_ratio = parse_config_string(data_str, "buyer_ratio", float)
    buyer_strategy = parse_config_string(data_str, "buyer_strategy")
    buyer_alpha = parse_config_string(data_str, "buyer_dirichlet_alpha", float)

    print(f"\n⚙️  Parsed Settings:")
    print(f"   - Dataset: {dataset}")
    print(f"   - Seller Strategy: {strategy} (Alpha={alpha})")
    print(f"   - Buyer Strategy:  {buyer_strategy} (Alpha={buyer_alpha})")
    print(f"   - Buyer Ratio:     {buyer_ratio}")

    # Reconstruct Config
    cfg = get_base_image_config()
    cfg.experiment.dataset_name = dataset
    cfg.data.image.buyer_ratio = buyer_ratio
    cfg.data.image.buyer_strategy = buyer_strategy
    cfg.data.image.buyer_dirichlet_alpha = buyer_alpha
    cfg.data.image.strategy = strategy
    cfg.data.image.dirichlet_alpha = alpha
    cfg.use_cache = False  # Force regen to test logic

    # Run Setup
    print("\n🔄 Simulating Data Partitioning (No Cache)...")
    try:
        device = "cpu"
        # Mute logging if possible, or just ignore it
        (buyer_loader, _, _, _, _, _, _, num_classes) = setup_data_and_model(cfg, device)

        if buyer_loader:
            analyze_buyer_stats(buyer_loader, num_classes)
        else:
            print("❌ Buyer Loader is None (Partition failed entirely)")

    except Exception as e:
        print(f"❌ Crash during setup: {e}")


def analyze_buyer_stats(loader, num_classes):
    labels = []
    for _, y in loader:
        labels.extend(y.tolist())

    counts = Counter(labels)
    unique_present = len(counts)
    missing = num_classes - unique_present
    avg_samples = np.mean(list(counts.values())) if counts else 0

    print(f"\n📊 RESULTS:")
    print(f"   - Total Samples: {len(labels)}")
    print(f"   - Classes: {unique_present} / {num_classes} (Missing: {missing})")
    print(f"   - Avg Samples/Class: {avg_samples:.2f}")

    if missing > 10 or avg_samples < 5:
        print("\n🔴 VERDICT: BROKEN ROOT DATASET")
        print("   FLTrust fails because the root dataset is too sparse/biased.")
        print("   -> Fix: Use 'iid' for uniform cases OR increase buyer_ratio.")
    else:
        print("\n🟢 VERDICT: HEALTHY ROOT DATASET")


if __name__ == "__main__":
    # === CONFIGURATION ===
    # We check two specific scenarios to verify the hypothesis

    # 1. SELLER ONLY (Alpha=0.1) -> Should have UNIFORM Buyer (Alpha=100)
    # If this is 'dirichlet', it might be broken.
    SCENARIO_BAD = "results/step11_seller_only_fltrust_CIFAR100"
    ALPHA_BAD = "alpha_0.1"

    # 2. BUYER ONLY (Alpha=0.1) -> Buyer IS Non-IID (Alpha=0.1)
    # This is SUPPOSED to be broken/biased.
    SCENARIO_EXPECTED_BAD = "results/step11_buyer_only_fltrust_CIFAR100"
    ALPHA_EXPECTED = "alpha_0.1"

    print("--- TEST 1: Seller-Only Bias (Buyer should be Uniform) ---")
    find_and_inspect(SCENARIO_BAD, ALPHA_BAD)

    print("\n\n--- TEST 2: Buyer-Only Bias (Buyer should be Biased) ---")
    find_and_inspect(SCENARIO_EXPECTED_BAD, ALPHA_EXPECTED)
