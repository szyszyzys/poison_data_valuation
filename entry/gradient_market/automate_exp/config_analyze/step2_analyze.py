import pandas as pd
import glob
import json
import re
import os
import pathlib

# --- Configuration ---
# Base directory where your results are saved
BASE_RESULTS_DIR = "./results"
# The experiment name pattern from your step 2 generator
EXPERIMENT_PATTERN = "step2_validate_fedavg_*"
# ---

def parse_path_info(filepath):
    """
    Parses the experiment parameters from a given file path.

    Expected path structure:
    ./results/{SCENARIO_NAME}/{HP_SUFFIX}/{RUN_DIR}/final_metrics.json
    e.g.:
    ./results/step2_validate_fedavg_image_cifar10_flexiblecnn/adv_0.3_poison_1.0/run_0_seed_42/final_metrics.json
    """
    try:
        path = pathlib.Path(filepath)
        parts = path.parts

        # 1. Parse Scenario Name (e.g., "step2_validate_fedavg_image_cifar10_flexiblecnn")
        scenario_name = parts[-4]
        scenario_match = re.match(r'step2_validate_(\w+)_(\w+)_([a-zA-Z0-9]+)_(\w+)', scenario_name)
        if not scenario_match:
            print(f"Warning: Skipping path with unexpected scenario format: {scenario_name}")
            return None

        defense, modality, dataset, model = scenario_match.groups()

        # 2. Parse HP Suffix (e.g., "adv_0.3_poison_1.0")
        hp_suffix = parts[-3]
        hp_match = re.match(r'adv_([\d\.]+)_poison_([\d\.]+)', hp_suffix)
        if not hp_match:
            print(f"Warning: Skipping path with unexpected HP format: {hp_suffix}")
            return None

        adv_rate, poison_rate = hp_match.groups()

        # 3. Parse Run Info (e.g., "run_0_seed_42")
        run_info = parts[-2]
        run_match = re.match(r'run_(\d+)_seed_(\d+)', run_info)
        if not run_match:
            print(f"Warning: Skipping path with unexpected run format: {run_info}")
            return None

        run_id, seed = run_match.groups()

        return {
            "dataset": dataset,
            "model": model,
            "defense": defense,
            "adv_rate": float(adv_rate),
            "poison_rate": float(poison_rate),
            "seed": int(seed),
            "filepath": filepath
        }

    except Exception as e:
        print(f"Error parsing path {filepath}: {e}")
        return None

def load_metrics(filepath):
    """Loads key metrics from the final_metrics.json file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Your run_final_evaluation_and_logging prefixes metrics
        # We also grab val_acc as a secondary check
        return {
            "test_acc": data.get("test_acc"),
            "test_asr": data.get("test_asr"),
            "val_acc": data.get("val_acc"),
        }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    """Main analysis function."""

    # 1. Find all result files
    search_path = os.path.join(
        BASE_RESULTS_DIR,
        EXPERIMENT_PATTERN,
        "adv_*_poison_*",
        "run_*_seed_*",
        "final_metrics.json"
    )

    print(f"🔍 Searching for results in: {search_path}\n")
    all_files = glob.glob(search_path)

    if not all_files:
        print("❌ No 'final_metrics.json' files found.")
        print("   Please check that your `BASE_RESULTS_DIR` is correct and that experiments have finished.")
        return

    print(f"✅ Found {len(all_files)} result files.")

    # 2. Parse all files and load metrics
    all_results = []
    for f in all_files:
        params = parse_path_info(f)
        if not params:
            continue

        metrics = load_metrics(f)
        if not metrics:
            continue

        params.update(metrics)
        all_results.append(params)

    if not all_results:
        print("❌ No valid results could be parsed.")
        return

    # 3. Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Define key metrics and grouping columns
    key_metrics = ['test_acc', 'test_asr', 'val_acc']
    group_by_cols = ['dataset', 'model', 'adv_rate', 'poison_rate']

    # Ensure metrics are numeric
    for col in key_metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Aggregate results (mean, std, count across seeds)
    df_agg = df.groupby(group_by_cols)[key_metrics].agg(['mean', 'std', 'count'])

    # Sort for readability
    df_agg = df_agg.sort_values(by=['dataset', 'model', 'adv_rate'], ascending=[True, True, True])

    # 5. Print the final summary table
    print("\n" + "="*80)
    print("--- Aggregated Attack Validation (Step 2) Results ---")
    print("="*80)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_agg)

    print("\n" + "="*80)
    print("--- 🔬 ANALYSIS ---")
    print("="*80)
    print("A successful attack validation has two parts:")
    print("\n1. BENIGN CASE (adv_rate = 0.0):")
    print("   - ('test_acc', 'mean') should be HIGH (e.g., > 0.60 for tabular, > 0.80 for CIFAR10).")
    print("   - ('test_asr', 'mean') should be LOW (near 0.01 for 100-class, near 0.10 for 10-class).")

    print("\n2. ATTACKED CASE (adv_rate > 0.0):")
    print("   - ('test_acc', 'mean') should STILL BE HIGH (close to the benign 'test_acc').")
    print("   - ('test_asr', 'mean') should be VERY HIGH (e.g., > 0.95 or 95%).")
    print("\nIf you see this pattern for all your models, your attacks are working. You can now proceed to 'Step 3: Defense Tuning'.")
    print("If 'test_asr' is low even when 'adv_rate' > 0.0, your attack setup has a bug.")


if __name__ == "__main__":
    main()