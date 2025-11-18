import pandas as pd
import json
import re
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
OUTPUT_DIR = "./analysis_outputs/step7_tables"

# Robust regex to match BOTH adaptive and baseline folders
SCENARIO_PATTERN = re.compile(r'step7_([a-z_]+)_([a-zA-Z0-9_]+)_(.*)')

# Defined Adversary Rate for Proxy Calculation (0.3 = 30%)
PROXY_ADV_RATE = 0.3


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

                # Initialize
                adv_sel_rate = 0.0
                ben_sel_rate = 0.0

                if report_file.exists():
                    with open(report_file, 'r') as f:
                        report = json.load(f)
                    sellers = report.get('seller_summaries', {})

                    # 1. Calculate Benign Selection Rate
                    # (Average of ALL benign sellers, always)
                    benign_rates = [
                        s['selection_rate'] for s in sellers.values()
                        if s.get('type') == 'benign'
                    ]
                    if benign_rates:
                        ben_sel_rate = np.mean(benign_rates)

                    # 2. Calculate Adversary Selection Rate (Conditional Logic)
                    if scenario_params['threat_model'] == 'baseline':
                        # --- FIX: PROXY BASELINE CALCULATION ---
                        # We want the rate for bn_0, bn_1, bn_2 (assuming sorted IDs)
                        # explicitly to match the 30% adv rate slots.

                        # Get all benign items (id, data)
                        benign_items = [
                            (sid, s) for sid, s in sellers.items()
                            if s.get('type') == 'benign'
                        ]

                        # Sort by ID (e.g., 'bn_0', 'bn_1'...)
                        # We use a natural sort or just string sort
                        benign_items.sort(key=lambda x: x[0])

                        # Determine how many to take (30% of total sellers)
                        total_sellers = len(sellers)
                        proxy_count = int(total_sellers * PROXY_ADV_RATE)
                        # Ensure at least 1 if rate > 0, default to 3 if 10 sellers
                        if proxy_count == 0 and PROXY_ADV_RATE > 0:
                            proxy_count = 3

                        # Extract the rates for the first 'proxy_count' sellers
                        proxy_rates = [item[1]['selection_rate'] for item in benign_items[:proxy_count]]

                        if proxy_rates:
                            adv_sel_rate = np.mean(proxy_rates)
                        else:
                            adv_sel_rate = 0.0

                    else:
                        # --- For Attacks, use ACTUAL ADVERSARIES ---
                        adv_rates = [
                            s['selection_rate'] for s in sellers.values()
                            if s.get('type') == 'adversary'
                        ]
                        if adv_rates:
                            adv_sel_rate = np.mean(adv_rates)
                        else:
                            adv_sel_rate = 0.0

                # Combine Data
                row = {
                    **scenario_params,
                    "acc": metrics.get('acc', 0) * 100,  # Convert to %
                    "asr": metrics.get('asr', 0) * 100,
                    "rounds": metrics.get('completed_rounds', 0),
                    "adv_sel_rate": adv_sel_rate * 100,
                    "ben_sel_rate": ben_sel_rate * 100,
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
    # Ensure columns exist before printing
    display_cols = [c for c in display_cols if c in df_agg.columns]
    print(df_agg[display_cols].to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()