import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ==========================================
# 1. GLOBAL CONFIGURATION & STYLING
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step10_figures"

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

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name."""
    try:
        pattern = r'step10_scalability_(fedavg|martfl|fltrust|skymask|skymask_small)_(.*)'
        match = re.search(pattern, scenario_name)
        if match:
            return {
                "scenario_base": scenario_name,
                "defense": match.group(1),
                "dataset": match.group(2)
            }
        else:
            return {}
    except Exception:
        return {}


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses 'n_sellers_10'"""
    hps = {}
    pattern = r'n_sellers_([0-9]+)'
    match = re.search(pattern, hp_folder_name)
    if match: hps['n_sellers'] = int(match.group(1))
    return hps


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """Loads key data from final_metrics.json and marketplace_report.json"""
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['asr'] = metrics.get('asr', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
            sellers = report.get('seller_summaries', {}).values()
            adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
            ben_sellers = [s for s in sellers if s.get('type') == 'benign']

            run_data['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv_sellers]) if adv_sellers else np.nan
            run_data['benign_selection_rate'] = np.mean([s['selection_rate'] for s in ben_sellers]) if ben_sellers else np.nan

            if not adv_sellers and ben_sellers:
                run_data['adv_selection_rate'] = 0.0 # Fix for clean runs if any

        return run_data
    except Exception:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)

    scenario_folders = [f for f in base_path.glob("step10_scalability_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if not run_scenario: continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                hp_folder_name = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder_name)
                run_metrics = load_run_data(metrics_file)

                if run_metrics:
                    all_runs.append({
                        **run_scenario,
                        **run_hps,
                        **run_metrics
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

def plot_scalability_composite_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Generates a SINGLE wide figure (1x4) for Scalability Analysis.
    """
    print(f"\n--- Plotting Composite Scalability Row: {dataset} ---")

    subset = df[df['dataset'] == dataset].copy()
    if subset.empty: return

    # Convert to percentages
    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        if col in subset.columns: subset[col] = subset[col] * 100

    # Only plot active defenses
    active_defenses = [d for d in DEFENSE_ORDER if d in subset['defense'].unique()]

    # Initialize Figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

    # Define metrics
    metrics_map = [
        ('acc', 'Accuracy (%)'),
        ('asr', 'ASR (%)'),
        ('benign_selection_rate', 'Benign Select (%)'),
        ('adv_selection_rate', 'Adv. Select (%)')
    ]

    for i, (metric, label) in enumerate(metrics_map):
        ax = axes[i]
        if metric in subset.columns:
            sns.lineplot(
                ax=ax,
                data=subset,
                x='n_sellers',
                y=metric,
                hue='defense',
                style='defense',
                hue_order=active_defenses,
                style_order=active_defenses,
                palette=DEFENSE_COLORS, # Use standard colors
                markers=True,
                dashes=False,
            )
            ax.set_title(f"{label.replace(' (%)', '')}", pad=15)
            ax.set_xlabel("Number of Sellers", labelpad=10)
            ax.set_ylabel(label, labelpad=10)
            ax.set_ylim(-5, 105)

            # Force integer ticks on X-axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Remove individual legends
            if ax.get_legend(): ax.get_legend().remove()
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')

    # --- Global Legend ---
    # Use handles from first plot (assuming it has data)
    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.16),
            ncol=len(active_defenses),
            frameon=True,
            title="Defense Methods"
        )

    # Save
    safe_dataset = re.sub(r'[^\w]', '', dataset)
    filename = output_dir / f"plot_scalability_composite_{safe_dataset}.pdf"
    plt.savefig(filename, bbox_inches='tight', format='pdf', dpi=300)
    print(f"  -> Saved plot to: {filename}")
    plt.close('all')


def main():
    # 1. Apply Global Style
    set_publication_style()

    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # 2. Collect Data
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    # 3. Plot
    for dataset in df['dataset'].unique():
        if pd.notna(dataset):
            plot_scalability_composite_row(df, dataset, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()