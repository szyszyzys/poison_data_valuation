import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step7_martfl_debug"
EXPLORATION_ROUNDS = 30
TARGET_DEFENSE = "martfl"

# --- PARSING LOGIC ---
def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    try:
        if 'baseline_no_attack' in scenario_name:
            parts = scenario_name.replace('step7_baseline_no_attack_', '').split('_')
            return {
                "threat_model": "baseline",
                "adaptive_mode": "N/A",
                "defense": parts[0],
                "dataset": parts[1] if len(parts) > 1 else "CIFAR100",
                "threat_label": "0. Baseline"
            }

        if not scenario_name.startswith('step7_adaptive_'):
            return {"defense": "unknown"}

        rest = scenario_name.replace('step7_adaptive_', '')

        # Threat Model Extraction
        threat_model = "unknown"
        if rest.startswith('black_box_'):
            threat_model = 'black_box'
            rest = rest.replace('black_box_', '')
        elif rest.startswith('gradient_inversion_'):
            threat_model = 'gradient_inversion'
            rest = rest.replace('gradient_inversion_', '')
        elif rest.startswith('oracle_'):
            threat_model = 'oracle'
            rest = rest.replace('oracle_', '')

        # Adaptive Mode Extraction
        adaptive_mode = "unknown"
        if rest.startswith('data_poisoning_'):
            adaptive_mode = 'data_poisoning'
            rest = rest.replace('data_poisoning_', '')
        elif rest.startswith('gradient_manipulation_'):
            adaptive_mode = 'gradient_manipulation'
            rest = rest.replace('gradient_manipulation_', '')

        parts = rest.split('_')
        defense = parts[0]
        dataset = parts[1] if len(parts) > 1 else "CIFAR100"

        threat_labels = {
            'black_box': '1. Black-Box',
            'gradient_inversion': '2. Grad-Inversion',
            'oracle': '3. Oracle'
        }

        return {
            "threat_model": threat_model,
            "adaptive_mode": adaptive_mode,
            "defense": defense,
            "dataset": dataset,
            "threat_label": threat_labels.get(threat_model, threat_model)
        }

    except Exception as e:
        print(f"!!! Error parsing '{scenario_name}': {e}")
        return {"defense": "unknown"}


# --- DATA COLLECTION WITH DEBUGGING ---
def collect_all_results(base_dir: str, target_defense: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_seller_dfs = []
    all_summary_rows = []
    base_path = Path(base_dir)

    # 1. Debug: List all candidate folders
    scenario_folders = list(base_path.glob("step7_*"))
    print(f"DEBUG: Found {len(scenario_folders)} total step7 folders.")

    for scenario_path in scenario_folders:
        print(f"Checking: {scenario_path.name}...")

        # 2. Debug: Check Parser
        scenario_params = parse_scenario_name(scenario_path.name)
        parsed_defense = scenario_params.get("defense")

        if parsed_defense != target_defense:
            print(f"  -> Skipped. (Parsed defense '{parsed_defense}' != target '{target_defense}')")
            continue

        print(f"  -> MATCHED! Threat: {scenario_params.get('threat_model')}")

        # 3. Debug: Check Files
        marker_files = list(scenario_path.rglob('final_metrics.json'))
        if not marker_files:
             print(f"  -> WARNING: No 'final_metrics.json' found inside {scenario_path.name}")

        for final_metrics_file in marker_files:
            try:
                run_dir = final_metrics_file.parent
                seed_id = f"{scenario_path.name}/{run_dir.name}"

                seller_file = run_dir / 'seller_metrics.csv'
                if seller_file.exists():
                    df_seller = pd.read_csv(seller_file, on_bad_lines='skip')
                    df_seller['seed_id'] = seed_id
                    df_seller = df_seller.assign(**scenario_params)

                    df_seller['seller_type'] = df_seller['seller_id'].apply(
                        lambda x: 'Adversary' if str(x).startswith('adv_') else 'Benign')

                    all_seller_dfs.append(df_seller)
                else:
                    print(f"     -> Missing seller_metrics.csv in {run_dir.name}")

                with open(final_metrics_file, 'r') as f:
                    metrics = json.load(f)

                # Validation Logic
                if not df_seller.empty:
                    valid = df_seller[df_seller['round'] > EXPLORATION_ROUNDS]
                    if valid.empty: valid = df_seller
                    adv_sel = valid[valid['seller_type'] == 'Adversary']['selected'].mean()
                else:
                    adv_sel = 0.0

                all_summary_rows.append({
                    **scenario_params,
                    'seed_id': seed_id,
                    'acc': metrics.get('acc', 0),
                    'adv_sel_rate': adv_sel
                })

            except Exception as e:
                print(f"  -> ERROR reading {run_dir}: {e}")

    df_s = pd.concat(all_seller_dfs, ignore_index=True) if all_seller_dfs else pd.DataFrame()
    df_sum = pd.DataFrame(all_summary_rows)

    return df_s, df_sum


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Debugging Step 7 Data Loading ---")
    df_s, df_sum = collect_all_results(BASE_RESULTS_DIR, target_defense=TARGET_DEFENSE)

    # --- SAVE DEBUG CSVS ---
    print("\n--- SAVING CSVs ---")
    if not df_sum.empty:
        sum_path = output_dir / "debug_summary_metrics.csv"
        df_sum.to_csv(sum_path, index=False)
        print(f"Saved summary data to: {sum_path}")
        print("Threat Labels found in Summary:")
        print(df_sum['threat_label'].value_counts())
    else:
        print("Summary DataFrame is empty!")

    if not df_s.empty:
        sel_path = output_dir / "debug_seller_metrics.csv"
        df_s.to_csv(sel_path, index=False)
        print(f"Saved seller data to: {sel_path}")
        print("Threat Labels found in Sellers:")
        print(df_s['threat_label'].value_counts())
    else:
        print("Seller DataFrame is empty!")

if __name__ == "__main__":
    main()
    main()