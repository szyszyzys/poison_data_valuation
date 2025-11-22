import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step11_figures_visuals"

# --- Constants for Heterogeneity Plot ---
ALPHAS_IN_TEST = [100.0, 1.0, 0.5, 0.1]
ALPHA_LABELS = ["IID", "1.0", "0.5", "0.1"]

# --- Constants for Scarcity Plot ---
RATIOS_IN_TEST = [0.01, 0.05, 0.1, 0.2]

# Consistent Styling
DEFENSE_PALETTE = {
    "fedavg": "#3498db",  # Blue
    "fltrust": "#e74c3c",  # Red
    "martfl": "#2ecc71",  # Green
    "skymask": "#9b59b6"  # Purple
}

# Added Markers for B/W readability
CUSTOM_MARKERS = {
    "fedavg": "o",
    "fltrust": "X",
    "martfl": "s",
    "skymask": "P"
}


def set_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 9


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses folder suffixes.
    Handles: 'iid', 'alpha_X', and 'ratio_sweep_X'.
    """
    hps = {}

    # Case 0: Explicit IID folder
    if hp_folder_name == "iid":
        hps['experiment_type'] = 'heterogeneity'
        hps['x_val'] = 100.0
        return hps

    # Case 1: Heterogeneity Sweep
    match_alpha = re.search(r'alpha_([0-9\.]+)', hp_folder_name)
    if match_alpha:
        hps['experiment_type'] = 'heterogeneity'
        hps['x_val'] = float(match_alpha.group(1))
        return hps

    # Case 2: Buyer Ratio Sweep
    match_ratio = re.search(r'ratio_sweep_([0-9\.]+)', hp_folder_name)
    if match_ratio:
        hps['experiment_type'] = 'scarcity'
        hps['x_val'] = float(match_ratio.group(1))
        return hps

    return {}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """Loads metrics from JSONs."""
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0  # Normalize %

        run_data['acc'] = acc
        run_data['asr'] = metrics.get('asr', 0)

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
            sellers = list(report.get('seller_summaries', {}).values())
            adv = [s for s in sellers if s.get('type') == 'adversary']
            ben = [s for s in sellers if s.get('type') == 'benign']

            run_data['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv]) if adv else 0.0
            run_data['benign_selection_rate'] = np.mean([s['selection_rate'] for s in ben]) if ben else 0.0
        else:
            run_data['adv_selection_rate'] = 0.0
            run_data['benign_selection_rate'] = 0.0

        return run_data
    except:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """
    Robust data collector handling both OLD (flat) and NEW (nested) directory structures.
    """
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step11_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} step11 folders.")

    for folder in scenario_folders:
        folder_name = folder.name

        # --- STRATEGY 1: Try Parse Old Structure (step11_BIAS_DEFENSE_DATASET) ---
        match_old = re.match(r'step11_(market_wide|buyer_only|seller_only)_(fedavg|martfl|fltrust|skymask)_(.*)', folder_name)

        # --- STRATEGY 2: Try Parse New Structure (step11_DEFENSE_DATASET) ---
        match_new = re.match(r'step11_(fedavg|martfl|fltrust|skymask)_(.*)', folder_name)

        # Default parsing info
        defense = None
        dataset = None
        bias_from_folder = None
        is_nested = False

        if match_old:
            bias_raw = match_old.group(1)
            bias_from_folder = bias_raw.replace('_', '-').title() + " Bias"
            defense = match_old.group(2)
            dataset = match_old.group(3)
            is_nested = False
        elif match_new:
            defense = match_new.group(1)
            dataset = match_new.group(2)
            is_nested = True # Bias info is inside subdirectories
        else:
            continue

        # Walk files
        for metrics_file in folder.rglob("final_metrics.json"):
            try:
                # Skip stale results if necessary (optional filter)
                # if "_alpha-100_" in metrics_file.parent.name: continue

                # Get path parts relative to the scenario folder
                parts = metrics_file.parent.relative_to(folder).parts

                bias_source = "unknown"
                hp_folder = None

                if is_nested:
                    # Expected: vary_buyer/alpha_0.1/...
                    if len(parts) < 2: continue

                    subdir_bias = parts[0] # e.g., vary_buyer
                    if subdir_bias == "vary_buyer": bias_source = "Buyer-Only Bias"
                    elif subdir_bias == "vary_seller": bias_source = "Seller-Only Bias"
                    elif subdir_bias == "vary_market": bias_source = "Market-Wide Bias"
                    else: continue

                    hp_folder = parts[1] # e.g., alpha_0.1
                else:
                    # Expected: alpha_0.1/...
                    if len(parts) < 1: continue
                    bias_source = bias_from_folder
                    hp_folder = parts[0]

                # Parse HP
                run_hps = parse_hp_suffix(hp_folder)
                if 'experiment_type' not in run_hps: continue

                # Load Metrics
                metrics = load_run_data(metrics_file)
                if metrics:
                    all_runs.append({
                        "defense": defense,
                        "dataset": dataset,
                        "bias_source": bias_source,
                        **run_hps,
                        **metrics
                    })
            except Exception as e:
                continue

    return pd.DataFrame(all_runs)


def plot_heterogeneity_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    print(f"\n--- Generating Heterogeneity Plots for {dataset} ---")

    dataset_df = df[(df['dataset'] == dataset) & (df['experiment_type'] == 'heterogeneity')].copy()
    if dataset_df.empty:
        print(f"  No heterogeneity data found.")
        return

    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        dataset_df[col] = dataset_df[col] * 100

    valid_biases = ['Market-Wide Bias', 'Buyer-Only Bias', 'Seller-Only Bias']
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    metrics_order = [('acc', 'Accuracy'), ('asr', 'ASR'), ('benign_selection_rate', 'Benign Select'), ('adv_selection_rate', 'Attacker Select')]

    for bias in valid_biases:
        bias_df = dataset_df[dataset_df['bias_source'] == bias]
        if bias_df.empty: continue

        fig, axes = plt.subplots(1, 4, figsize=(24, 4.5), constrained_layout=True)
        set_plot_style()

        for i, (col_name, display_name) in enumerate(metrics_order):
            ax = axes[i]
            sns.lineplot(
                ax=ax, data=bias_df, x='x_val', y=col_name,
                hue='defense', style='defense', hue_order=defense_order, style_order=defense_order,
                palette=DEFENSE_PALETTE, markers=CUSTOM_MARKERS, dashes=False, markersize=10, linewidth=3.0, errorbar=('ci', 95)
            )
            ax.set_title(f"{display_name}", fontweight='bold', fontsize=16)
            ax.set_xlabel("Heterogeneity", fontsize=14)
            if i == 0: ax.set_ylabel("Rate / Score (%)", fontsize=14)
            else: ax.set_ylabel("")

            # IID Logic (100 on Left)
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(FixedLocator(ALPHAS_IN_TEST))
            ax.xaxis.set_major_formatter(FixedFormatter(ALPHA_LABELS))
            ax.set_xlim(max(ALPHAS_IN_TEST) * 1.4, min(ALPHAS_IN_TEST) * 0.8)
            ax.grid(True, which='major', linestyle='--', alpha=0.6)
            ax.get_legend().remove()

        handles, labels = axes[0].get_legend_handles_labels()
        labels = [l.capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask", "SkyMask").replace("Martfl", "MARTFL") for l in labels]
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(defense_order), frameon=True, fontsize=14, title="Defense Methods")

        safe_bias = bias.replace(' ', '').replace('-', '')
        fname = output_dir / f"Step11_Heterogeneity_{dataset}_{safe_bias}.pdf"
        plt.savefig(fname, bbox_inches='tight', format='pdf')
        print(f"  Saved: {fname.name}")
        plt.close()


def plot_scarcity_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    print(f"\n--- Generating Data Scarcity Plots for {dataset} ---")

    dataset_df = df[(df['dataset'] == dataset) & (df['experiment_type'] == 'scarcity')].copy()
    if dataset_df.empty:
        print("  No Scarcity data found.")
        return

    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        dataset_df[col] = dataset_df[col] * 100

    bias = 'Seller-Only Bias'
    bias_df = dataset_df[dataset_df['bias_source'] == bias]
    if bias_df.empty: return

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    metrics_order = [('acc', 'Accuracy'), ('asr', 'ASR'), ('benign_selection_rate', 'Benign Select'), ('adv_selection_rate', 'Attacker Select')]

    fig, axes = plt.subplots(1, 4, figsize=(24, 4.5), constrained_layout=True)
    set_plot_style()

    for i, (col_name, display_name) in enumerate(metrics_order):
        ax = axes[i]
        sns.lineplot(
            ax=ax, data=bias_df, x='x_val', y=col_name,
            hue='defense', style='defense', hue_order=defense_order, style_order=defense_order,
            palette=DEFENSE_PALETTE, markers=CUSTOM_MARKERS, dashes=False, markersize=10, linewidth=3.0, errorbar=('ci', 95)
        )
        ax.set_title(f"{display_name}", fontweight='bold', fontsize=16)
        ax.set_xlabel("Buyer Data Ratio", fontsize=14)
        if i == 0: ax.set_ylabel("Rate / Score (%)", fontsize=14)
        else: ax.set_ylabel("")

        ax.xaxis.set_major_locator(FixedLocator(RATIOS_IN_TEST))
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
        ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    labels = [l.capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask", "SkyMask").replace("Martfl", "MARTFL") for l in labels]
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(defense_order), frameon=True, fontsize=14, title="Defense Methods")

    fig.suptitle(f"Impact of Root Data Scarcity (Seller-Only Bias, Alpha=0.5)", fontsize=16, y=1.05)
    fname = output_dir / f"Step11_Scarcity_{dataset}.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"  Saved: {fname.name}")
    plt.close()


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty:
        print("No valid data found.")
        return

    df.to_csv(output_dir / "step11_full_summary.csv", index=False)

    for dataset in df['dataset'].unique():
        if dataset != 'unknown':
            plot_heterogeneity_row(df, dataset, output_dir)
            plot_scarcity_row(df, dataset, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()