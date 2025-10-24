import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Regex to parse the hyperparameter folder name, e.g., "opt_Adam_lr_0.001_epochs_5"
HPARAM_REGEX = re.compile(r"opt_(?P<optimizer>\w+)_lr_(?P<lr>[\d\.]+)_epochs_(?P<epochs>\d+)")

def parse_scenario_name(name: str) -> Dict[str, str]:
    """
    Parses 'step1_tune_fedavg_image_CIFAR10_cnn'
    into {'dataset': 'CIFAR10', 'model': 'cnn'}
    """
    try:
        parts = name.split('_')
        return {
            "dataset": parts[-2],
            "model": parts[-1]
        }
    except Exception:
        print(f"Warning: Could not parse scenario name: {name}")
        return {"dataset": "unknown", "model": "unknown"}

def parse_hparam_name(name: str) -> Dict[str, Any]:
    """
    Parses 'opt_Adam_lr_0.001_epochs_5'
    into {'optimizer': 'Adam', 'lr': 0.001, 'epochs': 5}
    """
    match = HPARAM_REGEX.match(name)
    if not match:
        print(f"Warning: Could not parse hparam name: {name}")
        return {}

    data = match.groupdict()
    try:
        data['lr'] = float(data['lr'])
        data['epochs'] = int(data['epochs'])
        return data
    except ValueError:
        print(f"Warning: Could not cast types in hparam name: {name}")
        return {}


def find_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Finds all final_metrics.json files and parses their experiment context."""

    print(f"🔍 Scanning for results in: {results_dir}...")

    # Find all 'final_metrics.json' files from successful runs
    # The path is: <results_dir> / <scenario_name> / <hparam_name> / <seed_name> / final_metrics.json
    metrics_files = list(results_dir.glob("step1_tune_fedavg_*/*/*/final_metrics.json"))

    if not metrics_files:
        print("❌ ERROR: No 'final_metrics.json' files found.")
        print("Please make sure you are pointing to the correct root results directory.")
        return []

    print(f"✅ Found {len(metrics_files)} individual run results.")

    all_results = []

    for metrics_file in metrics_files:
        seed_dir = metrics_file.parent
        hparam_dir = seed_dir.parent
        scenario_dir = hparam_dir.parent

        # 1. Check if the run was successful
        if not (seed_dir / ".success").exists():
            continue # Skip this run, it failed.

        # 2. Parse context from folder names
        scenario_info = parse_scenario_name(scenario_dir.name)
        hparam_info = parse_hparam_name(hparam_dir.name)

        if not scenario_info or not hparam_info:
            continue # Skip if parsing failed

        try:
            seed = int(seed_dir.name.split('_')[-1])
        except Exception:
            seed = -1

        # 3. Load the metrics
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read or parse {metrics_file}: {e}")
            continue

        # 4. Store the combined record
        record = {
            **scenario_info,
            **hparam_info,
            "seed": seed,
            "scenario_name": scenario_dir.name,
            "status": "success",
            "test_acc": metrics.get("test_acc"),
            "test_loss": metrics.get("test_loss"),
            "completed_rounds": metrics.get("completed_rounds")
        }
        all_results.append(record)

    return all_results

def analyze_results(results: List[Dict[str, Any]]):
    """Aggregates results and prints summary tables."""

    if not results:
        print("No successful results to analyze.")
        return

    df = pd.DataFrame(results)

    # Define the core config for each experiment
    config_cols = ["scenario_name", "dataset", "model", "optimizer", "lr", "epochs"]

    # Aggregate results across seeds (e.g., mean, std)
    agg_df = df.groupby(config_cols).agg(
        mean_test_acc=pd.NamedAgg(column="test_acc", aggfunc="mean"),
        std_test_acc=pd.NamedAgg(column="test_acc", aggfunc="std"),
        mean_test_loss=pd.NamedAgg(column="test_loss", aggfunc="mean"),
        mean_rounds=pd.NamedAgg(column="completed_rounds", aggfunc="mean"),
        num_success_runs=pd.NamedAgg(column="seed", aggfunc="count")
    ).reset_index()

    # Fill NaN std (for single-seed runs) with 0 for cleaner sorting
    agg_df['std_test_acc'] = agg_df['std_test_acc'].fillna(0)

    # Sort by best accuracy
    agg_df = agg_df.sort_values(by="mean_test_acc", ascending=False)

    # --- 1. Show Top 3 Configs for Each Scenario ---
    print("\n" + "="*80)
    print("📈 Top 3 Performing Configs per Scenario (by mean_test_acc)")
    print("="*80)

    top_3 = agg_df.groupby("scenario_name").apply(
        lambda x: x.nlargest(3, "mean_test_acc")
    ).reset_index(drop=True)

    # Re-order columns for clarity
    display_cols_top3 = [
        "dataset", "model", "optimizer", "lr", "epochs",
        "mean_test_acc", "std_test_acc", "num_success_runs", "mean_rounds"
    ]
    print(top_3[display_cols_top3].to_string(index=False, float_format="%.4f"))


    # --- 2. Show the Single Best Config for Each Scenario ---
    print("\n" + "="*80)
    print("🏆 Best Baseline Config per Scenario (by max mean_test_acc)")
    print("="*80)

    # Find the index of the row with max 'mean_test_acc' for each group
    best_idx = agg_df.groupby("scenario_name")["mean_test_acc"].idxmax()
    best_configs_df = agg_df.loc[best_idx]

    # Sort for final display
    best_configs_df = best_configs_df.sort_values(by=["dataset", "model"])

    display_cols_best = [
        "dataset", "model", "optimizer", "lr", "epochs", "mean_test_acc", "std_test_acc"
    ]
    print(best_configs_df[display_cols_best].to_string(index=False, float_format="%.4f"))

    print("\n" + "="*80)
    print("Analysis complete. Use the 'Best Baseline Config' table to update your base configs.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FedAvg tuning results to find the best baseline configs."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        nargs="?",
        default="./results",
        help="The root directory where all experiment results are stored (default: ./results)"
    )

    args = parser.parse_args()
    results_path = Path(args.results_dir)

    # Set pandas display options for nice console output
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    try:
        results_list = find_all_results(results_path)
        analyze_results(results_list)
    except FileNotFoundError:
        print(f"❌ ERROR: The directory '{results_path}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()