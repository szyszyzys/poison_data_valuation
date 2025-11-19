# FILE: step12_visual_analysis.py

import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_main_summary"


# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    Parses Step 12 folder names to extract experimental variables.
    Expected format: step12_main_summary_[DEFENSE]_[MODALITY]_[DATASET]_[MODEL]
    Example: step12_main_summary_martfl_image_CIFAR100_cnn
    """
    try:
        parts = scenario_name.split('_')
        # Heuristic parsing based on standard naming convention
        if 'step12' in parts and 'main' in parts:
            # Find the index of 'summary' to anchor the rest
            try:
                idx = parts.index('summary')
                defense = parts[idx + 1]
                modality = parts[idx + 2]
                dataset = parts[idx + 3]
                return {
                    "scenario": scenario_name,
                    "defense": defense,
                    "modality": modality,
                    "dataset": dataset
                }
            except IndexError:
                pass

        # Fallback regex if split fails
        match = re.search(r'step12_main_summary_([^_]+)_([^_]+)_([^_]+)', scenario_name)
        if match:
            return {
                "scenario": scenario_name,
                "defense": match.group(1),
                "modality": match.group(2),
                "dataset": match.group(3)
            }

        return {"scenario": scenario_name, "defense": "unknown", "dataset": "unknown"}
    except Exception as e:
        print(f"Error parsing scenario '{scenario_name}': {e}")
        return {"scenario": scenario_name, "defense": "unknown", "dataset": "unknown"}


def load_metrics_from_csv(run_dir: Path) -> pd.DataFrame:
    """
    Scans seller_metrics.csv for:
    1. Valuation metrics (influence, shapley, loo, similarity)
    2. Selection rates (from 'selected' column)

    Returns a DataFrame row with columns like:
      Benign_influence_score, Adversary_influence_score, 
      Benign_selected, Adversary_selected, etc.
    """
    csv_path = run_dir / "seller_metrics.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        # FIX: on_bad_lines='skip' handles race-condition write errors in CSVs
        df = pd.read_csv(csv_path, on_bad_lines='skip')

        if df.empty or 'seller_id' not in df.columns:
            return pd.DataFrame()

        # 1. Identify Seller Type
        df['type'] = df['seller_id'].apply(
            lambda x: 'Adversary' if str(x).startswith('adv') else 'Benign'
        )

        # 2. Identify Columns of Interest
        # We want 'selected' plus anything looking like a valuation metric
        target_keywords = ['influence', 'shap', 'loo', 'sim_']

        # Find valuation columns
        val_cols = [c for c in df.columns if any(x in c.lower() for x in target_keywords)]

        # Add 'selected' if present (convert boolean to int for averaging)
        if 'selected' in df.columns:
            df['selected'] = df['selected'].astype(int)
            val_cols.append('selected')

        if not val_cols:
            return pd.DataFrame()

        # 3. Aggregate Average Score per Type
        # groupby().mean() automatically ignores NaNs
        summary = df.groupby('type')[val_cols].mean().reset_index()
        return summary

    except Exception as e:
        print(f"⚠️ Warning: Could not read CSV {csv_path}: {e}")
        return pd.DataFrame()


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """
    Walks directory, combines JSON global metrics with CSV detailed metrics.
    """
    all_runs = []
    base_path = Path(base_dir)

    # Find all Step 12 folders
    scenario_folders = [f for f in base_path.glob("step12_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenarios to process.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)

        # Look for final_metrics.json to identify completed runs
        for metrics_file in scenario_path.rglob("final_metrics.json"):
            run_dir = metrics_file.parent

            # 1. Load Standard Metrics (JSON)
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                acc = metrics.get('acc', 0)
                rounds = metrics.get('completed_rounds', 0)
            except:
                acc = 0
                rounds = 0

            # 2. Load Valuation & Selection Data (CSV)
            df_val = load_metrics_from_csv(run_dir)

            # Flatten the summary dataframe into a single dictionary row
            flat_record = {
                **run_scenario,
                "acc": acc,
                "rounds": rounds
            }

            if not df_val.empty:
                for _, row in df_val.iterrows():
                    s_type = row['type']  # Benign or Adversary
                    for col in df_val.columns:
                        if col != 'type':
                            # Example key: Benign_influence_score
                            flat_record[f"{s_type}_{col}"] = row[col]

            all_runs.append(flat_record)

    return pd.DataFrame(all_runs)


# --- PLOTTING HELPERS ---

def _clean_filename(s: str) -> str:
    """Sanitizes strings for filenames."""
    s = re.sub(r'\([^)]*\)', '', s)  # Remove parens
    s = re.sub(r'[^\w]', '_', s)  # Replace non-alphanumeric
    s = re.sub(r'_+', '_', s)  # Dedup underscores
    return s.strip('_')


def plot_platform_usability(df: pd.DataFrame, output_dir: Path):
    """
    Plots Global Accuracy, Rounds, and Selection Rates.
    """
    print("\n--- Plotting Performance & Selection ---")

    # List of (Display Name, Column Key, Is Percentage)
    # Note: 'Benign_selected' comes from the CSV aggregation
    metrics_config = [
        ('Global Accuracy', 'acc', True),
        ('Rounds to Converge', 'rounds', False),
        ('Benign Selection Rate', 'Benign_selected', True),
        ('Adversary Selection Rate', 'Adversary_selected', True)
    ]

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset].copy()
        if subset.empty: continue

        for disp_name, col_name, is_pct in metrics_config:
            # Skip if column doesn't exist (e.g. no 'Adversary_selected' data)
            if col_name not in subset.columns:
                continue

            # Skip if all NaN
            if subset[col_name].isna().all():
                continue

            plt.figure(figsize=(7, 5))

            # Aggregate across seeds
            plot_data = subset.groupby('defense')[col_name].mean().reset_index()

            if is_pct:
                # Convert 0.5 -> 50.0, but check if already percent
                if plot_data[col_name].max() <= 1.0:
                    plot_data[col_name] *= 100
                ylabel = f"{disp_name} (%)"
            else:
                ylabel = disp_name

            sns.barplot(
                data=plot_data,
                x='defense',
                y=col_name,
                order=[d for d in defense_order if d in plot_data['defense'].unique()],
                palette='viridis'
            )

            plt.title(f"{disp_name}\nDataset: {dataset}", fontsize=14)
            plt.ylabel(ylabel, fontsize=12)
            plt.xlabel("Defense", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.5)

            # Save
            clean_metric = _clean_filename(disp_name)
            fname = output_dir / f"Step12_Perf_{dataset}_{clean_metric}.pdf"
            plt.savefig(fname, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {fname.name}")


def plot_valuation_fairness(df: pd.DataFrame, output_dir: Path):
    """
    Plots Valuation and Similarity scores (Benign vs Adversary).
    """
    print("\n--- Plotting Valuation & Similarity ---")

    # Identify all metric roots (e.g., "influence_score" from "Benign_influence_score")
    metric_roots = set()
    for col in df.columns:
        if col.startswith('Adversary_') or col.startswith('Benign_'):
            root = col.replace('Adversary_', '').replace('Benign_', '')
            # Skip 'selected' as we plotted it in the performance section
            if root != 'selected':
                metric_roots.add(root)

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        if subset.empty: continue

        for root_metric in metric_roots:
            adv_col = f"Adversary_{root_metric}"
            ben_col = f"Benign_{root_metric}"

            # Skip if columns missing
            if adv_col not in subset.columns or ben_col not in subset.columns:
                continue

            # Skip if data is effectively empty (all NaNs or zeros if that's invalid)
            if subset[adv_col].isna().all() and subset[ben_col].isna().all():
                continue

            # Melt for Side-by-Side Bar Plot
            melted = subset.melt(
                id_vars=['defense'],
                value_vars=[adv_col, ben_col],
                var_name='Seller Type',
                value_name='Score'
            )

            # Rename Legend Labels
            melted['Seller Type'] = melted['Seller Type'].map({
                adv_col: 'Adversary',
                ben_col: 'Benign'
            })

            plt.figure(figsize=(8, 5))

            sns.barplot(
                data=melted,
                x='defense',
                y='Score',
                hue='Seller Type',
                order=[d for d in defense_order if d in melted['defense'].unique()],
                palette={'Benign': '#2ecc71', 'Adversary': '#e74c3c'},  # Green vs Red
                errorbar=None
            )

            # Aesthetics
            clean_title = root_metric.replace('_', ' ').title()
            plt.title(f"{clean_title}\nDataset: {dataset}", fontsize=14)
            plt.ylabel(f"Average Score", fontsize=12)
            plt.xlabel("Defense", fontsize=12)
            plt.axhline(0, color='black', linewidth=1)
            plt.legend(title=None)
            plt.grid(axis='y', linestyle='--', alpha=0.3)

            # Save
            clean_metric_name = _clean_filename(root_metric)
            fname = output_dir / f"Step12_Valuation_{dataset}_{clean_metric_name}.pdf"
            plt.savefig(fname, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {fname.name}")


# --- MAIN EXECUTION ---

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # 1. Collect Data
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data found in Step 12 folders.")
        return

    # 2. Save Summary CSV
    csv_path = output_dir / "step12_full_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary data to: {csv_path.name}")

    # 3. Generate Plots
    plot_platform_usability(df, output_dir)
    plot_valuation_fairness(df, output_dir)

    print("\n✅ Analysis complete.")


if __name__ == "__main__":
    main()