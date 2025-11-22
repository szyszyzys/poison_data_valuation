import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any
from matplotlib.ticker import FixedLocator, FixedFormatter

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
    "fedavg": "o",   # Circle
    "fltrust": "X",  # X
    "martfl": "s",   # Square
    "skymask": "P"   # Plus
}

def set_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 9


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses directory names to find Bias Source."""
    try:
        pattern = r'step11_(market_wide|buyer_only|seller_only)_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)
        if match:
            raw_bias = match.group(1)
            bias_formatted = raw_bias.replace('_', '-').title() + " Bias"
            return {
                "bias_source": bias_formatted,
                "defense": match.group(2),
                "dataset": match.group(3)
            }
        return {"bias_source": "IGNORE", "dataset": "unknown"}
    except:
        return {"bias_source": "IGNORE"}


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses folder suffixes.
    Handles: 'iid', 'alpha_X', and 'ratio_sweep_X'.
    """
    hps = {}

    # --- FIX IS HERE ---
    # Explicitly handle the "iid" folder name from the new generator
    if hp_folder_name == "iid":
        hps['experiment_type'] = 'heterogeneity'
        hps['x_val'] = 100.0 # Map string "iid" to numeric 100.0 for plotting
        return hps
    # -------------------

    # Case 1: Heterogeneity Sweep (e.g., alpha_1.0)
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
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step11_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenario folders.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if run_scenario.get("bias_source") == "IGNORE": continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                # --- FILTER OLD STALE RESULTS ---
                leaf_name = metrics_file.parent.name
                if "_alpha-100_" in leaf_name:
                    continue # Skip the old, incorrect IID runs
                # --------------------------------

                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                # Parse folder name (e.g., "iid", "alpha_1.0")
                hp_folder = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder)

                if 'experiment_type' not in run_hps:
                    # Debug print if things are being skipped unexpectedly
                    # print(f"Skipping unknown folder: {hp_folder}")
                    continue

                metrics = load_run_data(metrics_file)
                if metrics:
                    all_runs.append({**run_scenario, **run_hps, **metrics})
            except Exception as e:
                print(f"Error processing {metrics_file}: {e}")
                continue

    return pd.DataFrame(all_runs)


def plot_heterogeneity_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Plots: X-Axis = Alpha (Heterogeneity)
    """
    print(f"\n--- Generating Heterogeneity Plots for {dataset} ---")

    dataset_df = df[(df['dataset'] == dataset) & (df['experiment_type'] == 'heterogeneity')].copy()
    if dataset_df.empty:
        print(f"  No heterogeneity data found for {dataset}.")
        return

    # Convert to percentages
    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        dataset_df[col] = dataset_df[col] * 100

    valid_biases = ['Market-Wide Bias', 'Buyer-Only Bias', 'Seller-Only Bias']
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    metrics_order = [
        ('acc', 'Accuracy'),
        ('asr', 'ASR'),
        ('benign_selection_rate', 'Benign Select'),
        ('adv_selection_rate', 'Attacker Select')
    ]

    for bias in valid_biases:
        bias_df = dataset_df[dataset_df['bias_source'] == bias]
        if bias_df.empty: continue

        fig, axes = plt.subplots(1, 4, figsize=(24, 4.5), constrained_layout=True)
        set_plot_style()

        for i, (col_name, display_name) in enumerate(metrics_order):
            ax = axes[i]
            sns.lineplot(
                ax=ax,
                data=bias_df,
                x='x_val',
                y=col_name,
                hue='defense',
                style='defense',
                hue_order=defense_order,
                style_order=defense_order,
                palette=DEFENSE_PALETTE,
                markers=CUSTOM_MARKERS,
                dashes=False,
                markersize=10,
                linewidth=3.0,
                errorbar=('ci', 95)
            )

            ax.set_title(f"{display_name}", fontweight='bold', fontsize=16)
            ax.set_xlabel("Heterogeneity", fontsize=14)
            if i == 0: ax.set_ylabel("Rate / Score (%)", fontsize=14)
            else: ax.set_ylabel("")

            # --- AXIS LOGIC: IID (100) on LEFT ---
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(FixedLocator(ALPHAS_IN_TEST))
            ax.xaxis.set_major_formatter(FixedFormatter(ALPHA_LABELS))
            ax.set_xlim(max(ALPHAS_IN_TEST) * 1.4, min(ALPHAS_IN_TEST) * 0.8)

            ax.grid(True, which='major', linestyle='--', alpha=0.6)
            ax.get_legend().remove()

        handles, labels = axes[0].get_legend_handles_labels()
        labels = [l.capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask", "SkyMask").replace("Martfl", "MARTFL") for l in labels]

        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15),
                   ncol=len(defense_order), frameon=True, fontsize=14, title="Defense Methods")

        safe_bias = bias.replace(' ', '').replace('-', '')
        fname = output_dir / f"Step11_Heterogeneity_{dataset}_{safe_bias}.pdf"
        plt.savefig(fname, bbox_inches='tight', format='pdf')
        print(f"  Saved: {fname.name}")
        plt.close()


def plot_scarcity_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Plots: X-Axis = Buyer Ratio (Data Scarcity)
    """
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
    metrics_order = [
        ('acc', 'Accuracy'),
        ('asr', 'ASR'),
        ('benign_selection_rate', 'Benign Select'),
        ('adv_selection_rate', 'Attacker Select')
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 4.5), constrained_layout=True)
    set_plot_style()

    for i, (col_name, display_name) in enumerate(metrics_order):
        ax = axes[i]
        sns.lineplot(
            ax=ax,
            data=bias_df,
            x='x_val',
            y=col_name,
            hue='defense',
            style='defense',
            hue_order=defense_order,
            style_order=defense_order,
            palette=DEFENSE_PALETTE,
            markers=CUSTOM_MARKERS,
            dashes=False,
            markersize=10,
            linewidth=3.0,
            errorbar=('ci', 95)
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

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15),
               ncol=len(defense_order), frameon=True, fontsize=14, title="Defense Methods")

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

    # Save debug CSV
    df.to_csv(output_dir / "step11_full_summary.csv", index=False)

    for dataset in df['dataset'].unique():
        if dataset != 'unknown':
            plot_heterogeneity_row(df, dataset, output_dir)
            plot_scarcity_row(df, dataset, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()