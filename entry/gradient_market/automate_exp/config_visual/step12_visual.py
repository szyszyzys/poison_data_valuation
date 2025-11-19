import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step2.5_figures"
RELATIVE_ACC_THRESHOLD = 0.90
# ---------------------

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    hps = {}
    pattern = r'opt_(\w+)_lr_([0-9\.]+)_epochs_([0-9]+)'
    match = re.search(pattern, hp_folder_name)
    if match:
        hps['optimizer'] = match.group(1)
        hps['learning_rate'] = float(match.group(2))
        hps['local_epochs'] = int(match.group(3))
    return hps

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        pattern = r'step2\.5_find_hps_(fedavg|martfl|fltrust|skymask)_(image|text|tabular)_(.+)'
        match = re.search(pattern, scenario_name)
        if match:
            return {
                "scenario": scenario_name,
                "defense": match.group(1),
                "modality": match.group(2),
                "dataset": match.group(3),
            }
        else:
            return {"scenario": scenario_name, "defense": "unknown", "dataset": "unknown"}
    except Exception as e:
        return {"scenario": scenario_name}

def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """
    Loads metrics AND Valuation scores (KernelSHAP).
    """
    run_data = {}
    try:
        # 1. Load basic metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        # 2. Load Marketplace Report for Selection & Valuation
        report_file = metrics_file.parent / "marketplace_report.json"
        if not report_file.exists():
            return run_data

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = list(report.get('seller_summaries', {}).values())

        # Filter lists
        adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
        ben_sellers = [s for s in sellers if s.get('type') == 'benign']

        # --- A. Selection Rates ---
        run_data['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv_sellers]) if adv_sellers else 0.0
        run_data['benign_selection_rate'] = np.mean([s['selection_rate'] for s in ben_sellers]) if ben_sellers else 0.0

        # --- B. Valuation Scores (KernelSHAP) ---
        # We look for: seller -> valuation -> kernelshap

        def get_kshap(seller_list):
            scores = []
            for s in seller_list:
                val_data = s.get('valuation', {})
                # Some runs might not have valuation enabled, so use .get()
                # Assuming structure is {'kernelshap': 0.123, ...}
                score = val_data.get('kernelshap', np.nan)
                if pd.notna(score):
                    scores.append(score)
            return np.mean(scores) if scores else np.nan

        run_data['adv_kshap'] = get_kshap(adv_sellers)
        run_data['benign_kshap'] = get_kshap(ben_sellers)

        return run_data
    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}

def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)

    # Filter out nolocalclip
    scenario_folders = [
        f for f in base_path.glob("step2.5_find_hps_*")
        if f.is_dir() and not f.name.endswith("_nolocalclip")
    ]

    if not scenario_folders:
        print(f"Error: No directories found in {base_path}.")
        return pd.DataFrame()

    print(f"Processing {len(scenario_folders)} scenarios...")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                run_hps = parse_hp_suffix(relative_parts[0])
                run_metrics = load_run_data(metrics_file)

                if run_metrics:
                    all_runs.append({**run_scenario, **run_hps, **run_metrics})
            except Exception:
                continue

    if not all_runs: return pd.DataFrame()

    df = pd.DataFrame(all_runs)

    # Calculate Thresholds
    dataset_max_acc = df.groupby('dataset')['acc'].max().to_dict()
    df['platform_usable_threshold'] = df['dataset'].map(dataset_max_acc) * RELATIVE_ACC_THRESHOLD
    df['platform_usable'] = (df['acc'] >= df['platform_usable_threshold'])

    return df

def plot_platform_usability_with_selection(df: pd.DataFrame, output_dir: Path):
    """Plots Acc, Rounds, and Selection Rates."""
    print("\n--- Plotting Usability & Selection ---")

    # Prepare Metrics
    metrics = [
        ('Usability Rate (%)', 'platform_usable', True), # Metric Name, Col Name, is_bool
        ('Avg. Usable Accuracy (%)', 'acc', False),
        ('Avg. Usable Rounds', 'rounds', False),
        ('Avg. Benign Selection (%)', 'benign_selection_rate', False),
        ('Avg. Adversary Selection (%)', 'adv_selection_rate', False)
    ]

    # Filter data for 'Usable Accuracy/Rounds' to only include usable runs
    df_usable = df[df['platform_usable'] == True]

    for dataset in df['dataset'].unique():
        defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

        for metric_name, col, is_bool in metrics:
            # Choose source DF based on metric
            source_df = df_usable if col in ['acc', 'rounds'] else df
            source_df = source_df[source_df['dataset'] == dataset]

            if source_df.empty: continue

            # Aggregate
            agg_df = source_df.groupby('defense')[col].mean().reset_index()
            if col in ['acc', 'benign_selection_rate', 'adv_selection_rate'] or is_bool:
                agg_df[col] *= 100 # Convert to %

            plt.figure(figsize=(6, 4))
            sns.barplot(data=agg_df, x='defense', y=col, order=defense_order, palette='viridis')
            plt.title(f"{metric_name}\nDataset: {dataset}")
            plt.ylabel(metric_name)

            safe_name = metric_name.split('(')[0].strip().replace(' ', '_').replace('.', '')
            plt.savefig(output_dir / f"plot_{dataset}_{safe_name}.pdf", bbox_inches='tight')
            plt.close()

def plot_valuation_results(df: pd.DataFrame, output_dir: Path):
    """
    (NEW) Plots Benign vs Adversary KernelSHAP scores side-by-side.
    """
    print("\n--- Plotting Valuation (KernelSHAP) ---")

    # Check if we actually have valuation data
    if 'benign_kshap' not in df.columns or df['benign_kshap'].isna().all():
        print("⚠️ No KernelSHAP data found. Did you enable valuation in the config?")
        return

    # Melt the dataframe to put Benign/Adv SHAP in one column for plotting
    melt_cols = ['defense', 'dataset', 'benign_kshap', 'adv_kshap']
    df_val = df[melt_cols].melt(
        id_vars=['defense', 'dataset'],
        value_vars=['benign_kshap', 'adv_kshap'],
        var_name='Seller Type',
        value_name='KernelSHAP Score'
    )

    # Rename for legend
    df_val['Seller Type'] = df_val['Seller Type'].replace({
        'benign_kshap': 'Benign Sellers',
        'adv_kshap': 'Adversary Sellers'
    })

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    for dataset in df['dataset'].unique():
        plot_data = df_val[df_val['dataset'] == dataset]
        if plot_data.empty: continue

        plt.figure(figsize=(8, 5))

        sns.barplot(
            data=plot_data,
            x='defense',
            y='KernelSHAP Score',
            hue='Seller Type',
            order=defense_order,
            palette={'Benign Sellers': 'blue', 'Adversary Sellers': 'red'},
            errorbar=None # Cleaner look
        )

        plt.title(f"Valuation Fairness (KernelSHAP)\nDataset: {dataset}")
        plt.ylabel("Avg. Shapley Value (Higher is Better)")
        plt.axhline(0, color='black', linewidth=0.8) # Zero line

        plt.savefig(output_dir / f"plot_{dataset}_KernelSHAP.pdf", bbox_inches='tight')
        plt.close()

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty: return

    # 1. Plot Standard Metrics
    plot_platform_usability_with_selection(df, output_dir)

    # 2. Plot Valuation Metrics (NEW)
    plot_valuation_results(df, output_dir)

    # Save CSV
    df.to_csv(output_dir / "step2.5_full_summary.csv", index=False)
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()