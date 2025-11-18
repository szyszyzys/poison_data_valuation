import pandas as pd
import json
import re
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
OUTPUT_DIR = "./analysis_outputs/step7_tables"  # Where to save CSVs/Tables

# Robust regex to match BOTH adaptive and baseline folders
SCENARIO_PATTERN = re.compile(r'step7_([a-z_]+)_([a-zA-Z0-9_]+)_(.*)')


# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses the experiment scenario name from the folder name.
    Handles both 'step7_adaptive_*' and 'step7_baseline_no_attack_*'.
    """
    try:
        # 1. Check for Baseline
        if scenario_name.startswith('step7_baseline_no_attack_'):
            match = re.search(r'step7_baseline_no_attack_([a-zA-Z0-9_]+)_(.*)', scenario_name)
            if match:
                return {
                    "threat_model": "baseline",
                    "adaptive_mode": "N/A",
                    "defense": match.group(1),
                    "dataset": match.group(2),
                    "threat_label": "0. Baseline (No Attack)"
                }

        # 2. Check for Adaptive Attack
        elif scenario_name.startswith('step7_adaptive_'):
            match = re.search(r'step7_adaptive_([a-z_]+)_([a-z_]+)_([a-zA-Z0-9_]+)_(.*)', scenario_name)
            if match:
                threat_model = match.group(1)
                adaptive_mode = match.group(2)
                defense = match.group(3)
                dataset = match.group(4)

                threat_model_map = {
                    'black_box': '1. Black-Box',
                    'gradient_inversion': '2. Grad-Inversion',
                    'oracle': '3. Oracle'
                }
                threat_label = threat_model_map.get(threat_model, threat_model)

                return {
                    "threat_model": threat_model,
                    "adaptive_mode": adaptive_mode,
                    "defense": defense,
                    "dataset": dataset,
                    "threat_label": threat_label
                }

        # 3. Fallback
        return {"defense": "unknown"}
    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"defense": "unknown"}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """
    Walks the directory, parses folders, and aggregates final metrics.
    """
    all_rows = []
    base_path = Path(base_dir)
    print(f"🔍 Scanning for 'step7_*' results in: {base_path.resolve()}...")

    # --- FIX: Glob for ALL step7 folders (baseline + adaptive) ---
    scenario_folders = list(base_path.glob("step7_*"))

    if not scenario_folders:
        print("❌ Error: No 'step7_*' directories found.")
        return pd.DataFrame()

    print(f"✅ Found {len(scenario_folders)} base directories.")

    for scenario_path in scenario_folders:
        scenario_params = parse_scenario_name(scenario_path.name)

        # Filter out unparseable folders
        if scenario_params.get("defense") == "unknown":
            continue

        # Find all runs (seeds)
        metrics_files = list(scenario_path.rglob('final_metrics.json'))

        for metrics_file in metrics_files:
            try:
                # Load Global Metrics
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                # Load Seller Selection Rates
                report_file = metrics_file.parent / "marketplace_report.json"
                adv_sel_rate = np.nan
                ben_sel_rate = np.nan

                if report_file.exists():
                    with open(report_file, 'r') as f:
                        report = json.load(f)
                    sellers = report.get('seller_summaries', {}).values()

                    adv_sellers = [s['selection_rate'] for s in sellers if s.get('type') == 'adversary']
                    ben_sellers = [s['selection_rate'] for s in sellers if s.get('type') == 'benign']

                    if adv_sellers:
                        adv_sel_rate = np.mean(adv_sellers)
                    if ben_sellers:
                        ben_sel_rate = np.mean(ben_sellers)

                # Combine Data
                row = {
                    **scenario_params,
                    "acc": metrics.get('acc', 0) * 100,  # Convert to %
                    "asr": metrics.get('asr', 0) * 100,
                    "rounds": metrics.get('completed_rounds', 0),
                    "adv_sel_rate": adv_sel_rate * 100 if pd.notna(adv_sel_rate) else 0.0,
                    "ben_sel_rate": ben_sel_rate * 100 if pd.notna(ben_sel_rate) else 0.0,
                    "seed_path": str(metrics_file.parent)
                }
                all_rows.append(row)

            except Exception as e:
                print(f"Error processing {metrics_file}: {e}")

    return pd.DataFrame(all_rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Collect Data
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("❌ No valid results found.")
        return

    # 2. Filter for MartFL Only (as requested)
    print("\n--- Filtering for MartFL results ---")
    df_martfl = df[df['defense'] == 'martfl'].copy()

    if df_martfl.empty:
        print("❌ No MartFL results found.")
        return

    # 3. Aggregate (Mean/Std over seeds)
    # Group by the key experiment parameters
    group_cols = ['threat_label', 'adaptive_mode', 'defense', 'dataset']
    metric_cols = ['acc', 'asr', 'adv_sel_rate', 'ben_sel_rate']

    df_agg = df_martfl.groupby(group_cols)[metric_cols].agg(['mean', 'std']).reset_index()

    # Flatten columns
    df_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_agg.columns.values]

    # 4. Save Summary CSV
    csv_path = Path(OUTPUT_DIR) / "step7_martfl_summary_table.csv"
    df_agg.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"✅ Saved summary CSV to: {csv_path}")

    # 5. Print nice summary to console
    print("\n=== MartFL Adaptive Attack Summary ===")
    # Select and reorder relevant columns for display
    display_cols = ['threat_label', 'adaptive_mode', 'adv_sel_rate_mean', 'ben_sel_rate_mean', 'acc_mean']
    print(df_agg[display_cols].to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()