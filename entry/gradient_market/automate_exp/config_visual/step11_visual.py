import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator, FixedFormatter

# ==========================================
# 1. GLOBAL CONFIGURATION & STYLING
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step11_figures_visuals"

# --- Naming Standards ---
PRETTY_NAMES = {
    "fedavg": "FedAvg",
    "fltrust": "FLTrust",
    "martfl": "MARTFL",
    "skymask": "SkyMask",
    "skymask_small": "SkyMask",
}

# --- Color Standards ---
DEFENSE_COLORS = {
    "FedAvg": "#7f8c8d",   # Grey
    "FLTrust": "#3498db",  # Blue
    "MARTFL": "#2ecc71",   # Green
    "SkyMask": "#e74c3c",  # Red (Highlighted)
}

DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

# --- Markers for Line Plots ---
CUSTOM_MARKERS = {
    "FedAvg": "o",
    "FLTrust": "X",
    "MARTFL": "s",
    "SkyMask": "P"
}

# --- Constants for Heterogeneity Plot ---
ALPHAS_IN_TEST = [100.0, 1.0, 0.5, 0.1]
ALPHA_LABELS = ["IID", "1.0", "0.5", "0.1"]

# --- Constants for Scarcity Plot ---
RATIOS_IN_TEST = [0.01, 0.05, 0.1, 0.2]

def format_label(label: str) -> str:
    """Standardizes names."""
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_publication_style():
    """Sets the 'Big & Bold' professional style globally."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.8) # Consistent Scale

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'legend.title_fontsize': 20,
        'axes.linewidth': 2.0,
        'axes.edgecolor': '#333333',
        'lines.linewidth': 3.5,
        'lines.markersize': 11,
    })

# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses folder suffixes."""
    hps = {}
    if hp_folder_name == "iid":
        hps['experiment_type'] = 'heterogeneity'
        hps['x_val'] = 100.0
        return hps

    match_alpha = re.search(r'alpha_([0-9\.]+)', hp_folder_name)
    if match_alpha:
        hps['experiment_type'] = 'heterogeneity'
        hps['x_val'] = float(match_alpha.group(1))
        return hps

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
        if acc > 1.0: acc /= 100.0

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
    """Robust data collector handling directory structures."""
    all_runs = []
    base_path = Path(base_dir)

    scenario_folders = [f for f in base_path.glob("step11_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} step11 folders.")

    for folder in scenario_folders:
        folder_name = folder.name

        match_old = re.match(r'step11_(market_wide|buyer_only|seller_only)_(fedavg|martfl|fltrust|skymask)_(.*)', folder_name)
        match_new = re.match(r'step11_(fedavg|martfl|fltrust|skymask)_(.*)', folder_name)

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
            is_nested = True
        else:
            continue

        for metrics_file in folder.rglob("final_metrics.json"):
            try:
                parts = metrics_file.parent.relative_to(folder).parts
                bias_source = "unknown"
                hp_folder = None

                if is_nested:
                    if len(parts) < 2: continue
                    subdir_bias = parts[0]
                    if subdir_bias == "vary_buyer": bias_source = "Buyer-Only Bias"
                    elif subdir_bias == "vary_seller": bias_source = "Seller-Only Bias"
                    elif subdir_bias == "vary_market": bias_source = "Market-Wide Bias"
                    else: continue
                    hp_folder = parts[1]
                else:
                    if len(parts) < 1: continue
                    bias_source = bias_from_folder
                    hp_folder = parts[0]

                run_hps = parse_hp_suffix(hp_folder)
                if 'experiment_type' not in run_hps: continue

                metrics = load_run_data(metrics_file)
                if metrics:
                    all_runs.append({
                        "defense": defense,
                        "dataset": dataset,
                        "bias_source": bias_source,
                        **run_hps,
                        **metrics
                    })
            except Exception:
                continue

    df = pd.DataFrame(all_runs)

    # --- Standardization ---
    if not df.empty:
        df['defense'] = df['defense'].apply(format_label)

    return df

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_heterogeneity_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    print(f"\n--- Generating Heterogeneity Plots for {dataset} ---")

    dataset_df = df[(df['dataset'] == dataset) & (df['experiment_type'] == 'heterogeneity')].copy()
    if dataset_df.empty: return

    # Convert to percentages
    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        dataset_df[col] = dataset_df[col] * 100

    valid_biases = ['Market-Wide Bias', 'Buyer-Only Bias', 'Seller-Only Bias']
    metrics_order = [('acc', 'Accuracy'), ('asr', 'ASR'), ('benign_selection_rate', 'Benign Select'), ('adv_selection_rate', 'Attacker Select')]

    # Only plot active defenses
    active_defenses = [d for d in DEFENSE_ORDER if d in dataset_df['defense'].unique()]

    for bias in valid_biases:
        bias_df = dataset_df[dataset_df['bias_source'] == bias]
        if bias_df.empty: continue

        fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

        for i, (col_name, display_name) in enumerate(metrics_order):
            ax = axes[i]
            sns.lineplot(
                ax=ax, data=bias_df, x='x_val', y=col_name,
                hue='defense', style='defense',
                hue_order=active_defenses, style_order=active_defenses,
                palette=DEFENSE_COLORS, markers=CUSTOM_MARKERS,
                dashes=False, errorbar=('ci', 95)
            )
            ax.set_title(f"{display_name}", pad=15)
            ax.set_xlabel("Heterogeneity", labelpad=10)

            if i == 0: ax.set_ylabel("Rate / Score (%)", labelpad=10)
            else: ax.set_ylabel("")

            # IID Logic (100 on Left)
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(FixedLocator(ALPHAS_IN_TEST))
            ax.xaxis.set_major_formatter(FixedFormatter(ALPHA_LABELS))
            ax.set_xlim(max(ALPHAS_IN_TEST) * 1.4, min(ALPHAS_IN_TEST) * 0.8)

            if ax.get_legend(): ax.get_legend().remove()

        # Global Legend
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.16),
                       ncol=len(active_defenses), frameon=True, title="Defense Methods")

        safe_bias = bias.replace(' ', '').replace('-', '')
        fname = output_dir / f"Step11_Heterogeneity_{dataset}_{safe_bias}.pdf"
        plt.savefig(fname, bbox_inches='tight', format='pdf')
        print(f"  Saved: {fname.name}")
        plt.close()


def plot_scarcity_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    print(f"\n--- Generating Data Scarcity Plots for {dataset} ---")

    dataset_df = df[(df['dataset'] == dataset) & (df['experiment_type'] == 'scarcity')].copy()
    if dataset_df.empty: return

    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        dataset_df[col] = dataset_df[col] * 100

    bias = 'Seller-Only Bias' # Default view
    bias_df = dataset_df[dataset_df['bias_source'] == bias]
    if bias_df.empty: return

    metrics_order = [('acc', 'Accuracy'), ('asr', 'ASR'), ('benign_selection_rate', 'Benign Select'), ('adv_selection_rate', 'Attacker Select')]
    active_defenses = [d for d in DEFENSE_ORDER if d in dataset_df['defense'].unique()]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

    for i, (col_name, display_name) in enumerate(metrics_order):
        ax = axes[i]
        sns.lineplot(
            ax=ax, data=bias_df, x='x_val', y=col_name,
            hue='defense', style='defense',
            hue_order=active_defenses, style_order=active_defenses,
            palette=DEFENSE_COLORS, markers=CUSTOM_MARKERS,
            dashes=False, errorbar=('ci', 95)
        )
        ax.set_title(f"{display_name}", pad=15)
        ax.set_xlabel("Buyer Data Ratio", labelpad=10)

        if i == 0: ax.set_ylabel("Rate / Score (%)", labelpad=10)
        else: ax.set_ylabel("")

        ax.xaxis.set_major_locator(FixedLocator(RATIOS_IN_TEST))
        if ax.get_legend(): ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.16),
                   ncol=len(active_defenses), frameon=True, title="Defense Methods")

    # Main Title
    # plt.suptitle(f"Impact of Root Data Scarcity (Seller-Only Bias, Alpha=0.5)", fontsize=24, fontweight='bold', y=1.05)

    fname = output_dir / f"Step11_Scarcity_{dataset}.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"  Saved: {fname.name}")
    plt.close()


def main():
    # 1. Apply Global Style
    set_publication_style()

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