import pandas as pd
import glob
import json
import re
import os
import pathlib

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
EXPERIMENT_PATTERN = "step2_validate_fedavg_*"


# ---

def parse_path_info(filepath):
    """
    Parses the experiment parameters from a given file path.
    """
    try:
        path = pathlib.Path(filepath)
        parts = path.parts

        scenario_name = parts[-4]
        scenario_match = re.match(r'step2_validate_(\w+)_(\w+)_([a-zA-Z0-9]+)_(\w+)', scenario_name)
        if not scenario_match:
            print(f"Warning: Skipping path with unexpected scenario format: {scenario_name}")
            return None

        defense, modality, dataset, model = scenario_match.groups()

        hp_suffix = parts[-3]

        # Use re.search to find the patterns anywhere in the complex string
        adv_match = re.search(r'adv-([\d_p]+)', hp_suffix)
        poison_match = re.search(r'poison-([\d_p]+)', hp_suffix)

        if not adv_match or not poison_match:
            # Handle the 0.0 case which might not have 'poison'
            if adv_match:
                adv_rate_str = adv_match.group(1).replace('p', '.')
                adv_rate = float(adv_rate_str)
                if adv_rate == 0.0:
                    poison_rate = 0.0  # Assume 0.0 if not found and adv_rate is 0
                else:
                    print(f"Warning: Skipping path, could not parse poison rate from: {hp_suffix}")
                    return None
            else:
                print(f"Warning: Skipping path, could not parse adv rate from: {hp_suffix}")
                return None
        else:
            adv_rate_str = adv_match.group(1).replace('p', '.')
            poison_rate_str = poison_match.group(1).replace('p', '.')
            adv_rate = float(adv_rate_str)
            poison_rate = float(poison_rate_str)

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
            "adv_rate": adv_rate,
            "poison_rate": poison_rate,
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

        # --- THIS IS THE FIX ---
        # Look for "acc" and "asr" (from your JSON)
        # NOT "test_acc" and "test_asr"
        return {
            "test_acc": data.get("acc"),  # <-- CHANGED
            "test_asr": data.get("asr"),  # <-- CHANGED
            "val_acc": data.get("B-Acc"),  # Use B-Acc as val_acc
        }
        # --- END FIX ---

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def main():
    """Main analysis function."""

    # Use a more flexible wildcard "*" to match the complex HP string
    search_path = os.path.join(
        BASE_RESULTS_DIR,
        EXPERIMENT_PATTERN,
        "*",  # Flexible pattern
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

    all_results = []
    for f in all_files:
        params = parse_path_info(f)
        if not params:
            continue

        metrics = load_metrics(f)
        if not metrics or metrics.get('test_acc') is None:
            print(f"Warning: Skipping {f}, missing 'acc' or 'asr' in JSON.")
            continue

        params.update(metrics)
        all_results.append(params)

    if not all_results:
        print("❌ No valid results could be parsed.")
        return

    df = pd.DataFrame(all_results)

    key_metrics = ['test_acc', 'test_asr', 'val_acc']
    group_by_cols = ['dataset', 'model', 'adv_rate', 'poison_rate']

    for col in key_metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_agg = df.groupby(group_by_cols)[key_metrics].agg(['mean', 'std', 'count'])

    df_agg = df_agg.sort_values(by=['dataset', 'model', 'adv_rate'], ascending=[True, True, True])

    print("\n" + "=" * 80)
    print("--- Aggregated Attack Validation (Step 2) Results ---")
    print("=" * 80)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_agg)

    print("\n" + "=" * 80)
    print("--- 🔬 ANALYSIS ---")
    print("=" * 80)
    print("A successful attack validation has two parts:")
    print("\n1. BENIGN CASE (adv_rate = 0.0):")
    print("   - ('test_acc', 'mean') should be HIGH (e.g., > 0.60 for tabular, > 0.80 for CIFAR10).")
    print("   - ('test_asr', 'mean') should be LOW (near 0.01 for 100-class, near 0.10 for 10-class).")

    print("\n2. ATTACKED CASE (adv_rate > 0.0):")
    print("   - ('test_acc', 'mean') should STILL BE HIGH (close to the benign 'test_acc').")
    print("   - ('test_asr', 'mean') should be VERY HIGH (e.g., > 0.95 or 95%).")
    print(
        "\nIf you see this pattern for all your models, your attacks are working. You can now proceed to 'Step 3: Defense Tuning'.")
    print("If 'test_asr' is low even when 'adv_rate' > 0.0, your attack setup has a bug.")


if __name__ == "__main__":
    main()