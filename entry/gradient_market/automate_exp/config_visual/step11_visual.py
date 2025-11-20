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

# We treat 100.0 as the "IID" case based on your experiment generation
ALPHAS_IN_TEST = [100.0, 1.0, 0.5, 0.1]

# Custom Labels for the X-Axis (Must match ALPHAS_IN_TEST order)
ALPHA_LABELS = ["IID", "1.0", "0.5", "0.1"]

# Consistent Styling
DEFENSE_PALETTE = {
    "fedavg": "#3498db",  # Blue
    "fltrust": "#e74c3c",  # Red
    "martfl": "#2ecc71",  # Green
    "skymask": "#9b59b6"  # Purple
}


# ---------------------

def set_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
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
    """Parses alpha value from folder name."""
    hps = {}
    match = re.search(r'alpha_([0-9\.]+)', hp_folder_name)
    if match:
        hps['dirichlet_alpha'] = float(match.group(1))
    return hps


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
    print(f"Searching in {base_path}...")

    scenario_folders = [f for f in base_path.glob("step11_*") if f.is_dir()]

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if run_scenario.get("bias_source") == "IGNORE": continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                run_hps = parse_hp_suffix(relative_parts[0])
                if 'dirichlet_alpha' not in run_hps: continue

                metrics = load_run_data(metrics_file)
                if metrics:
                    all_runs.append({**run_scenario, **run_hps, **metrics})
            except:
                continue

    return pd.DataFrame(all_runs)


def plot_four_in_a_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Generates the composite figure.
    CRITICAL UPDATE: X-Axis now shows "IID" on the left.
    """
    print(f"\n--- Generating Composite Row Figures for {dataset} ---")
    set_plot_style()

    dataset_df = df[df['dataset'] == dataset].copy()
    if dataset_df.empty: return

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

        for i, (col_name, display_name) in enumerate(metrics_order):
            ax = axes[i]

            sns.lineplot(
                ax=ax,
                data=bias_df,
                x='dirichlet_alpha',
                y=col_name,
                hue='defense',
                style='defense',
                hue_order=defense_order,
                style_order=defense_order,
                palette=DEFENSE_PALETTE,
                markers=True,
                dashes=False,
                markersize=9,
                linewidth=2.5,
                errorbar=('ci', 95)
            )

            ax.set_title(f"{display_name}", fontweight='bold', fontsize=16)
            ax.set_xlabel("Heterogeneity", fontsize=14)
            if i == 0:
                ax.set_ylabel("Rate / Score (%)", fontsize=14)
            else:
                ax.set_ylabel("")

            # --- KEY UPDATE: CUSTOM X-AXIS LABELS ---
            ax.set_xscale('log')

            # 1. Set Ticks exactly where your data points are
            ax.xaxis.set_major_locator(FixedLocator(ALPHAS_IN_TEST))

            # 2. Replace numbers with ["IID", "1.0", "0.5", "0.1"]
            ax.xaxis.set_major_formatter(FixedFormatter(ALPHA_LABELS))

            # 3. Reverse the axis limit so 100 (IID) is on the LEFT
            # Log axis: Higher number usually on right. We want 100 on left.
            # So we set max limit > 100 and min limit < 0.1, but ordered (max, min)
            ax.set_xlim(max(ALPHAS_IN_TEST) * 1.4, min(ALPHAS_IN_TEST) * 0.8)

            ax.grid(True, which='major', linestyle='--', alpha=0.6)
            ax.get_legend().remove()

        # Legend and Title
        handles, labels = axes[0].get_legend_handles_labels()
        labels = [l.capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask",
                                                                                                   "SkyMask").replace(
            "Martfl", "MARTFL") for l in labels]

        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15),
                   ncol=len(defense_order), frameon=True, fontsize=14, title="Defense Methods")

        # fig.suptitle(f"Impact of {bias} on {dataset}", fontsize=18, fontweight='bold', y=1.05)

        safe_bias = bias.replace(' ', '').replace('-', '')
        fname = output_dir / f"Step11_Composite_Row_{dataset}_{safe_bias}.pdf"
        plt.savefig(fname, bbox_inches='tight', format='pdf')
        print(f"  Saved Composite: {fname.name}")
        plt.close()


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty:
        print("No valid data found.")
        return

    for dataset in df['dataset'].unique():
        if dataset != 'unknown':
            plot_four_in_a_row(df, dataset, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()