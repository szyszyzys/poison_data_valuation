# FILE: step11_visual_individual.py

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
FIGURE_OUTPUT_DIR = "./figures/step11_figures_individual"
ALPHAS_IN_TEST = [100.0, 1.0, 0.5, 0.1]

# Hardcoded colors to ensure consistent legend across different files
DEFENSE_PALETTE = {
    "fedavg": "#3498db",  # Blue
    "fltrust": "#e74c3c",  # Red
    "martfl": "#2ecc71",  # Green
    "skymask": "#9b59b6"  # Purple
}


# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses directory names to find Bias Source."""
    try:
        # Pattern: step11_[bias]_[defense]_[dataset]
        pattern = r'step11_(market_wide|buyer_only|seller_only)_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)

        if match:
            raw_bias = match.group(1)
            # Format: "buyer_only" -> "Buyer-Only Bias"
            bias_formatted = raw_bias.replace('_', '-').title() + " Bias"
            return {
                "scenario_base": scenario_name,
                "bias_source": bias_formatted,
                "defense": match.group(2),
                "dataset": match.group(3)
            }
        else:
            # Filter out the generic 'heterogeneity' folder or unknown folders
            return {"bias_source": "IGNORE", "dataset": "unknown"}

    except Exception:
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

        # Load Marketplace Report for selection rates
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
    scenario_folders = [f for f in base_path.glob("step11_*") if f.is_dir()]

    print(f"Found {len(scenario_folders)} scenario directories.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)

        # Skip folders that don't match specific bias types
        if run_scenario.get("bias_source") == "IGNORE":
            continue

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


# =============================================================================
#  PLOTTER: SPLIT EVERYTHING
# =============================================================================

def plot_completely_separate(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Loops through Bias Source -> Loops through Metric -> Saves Individual PDF.
    """
    print(f"\n--- Generating Split Figures for {dataset} ---")

    dataset_df = df[df['dataset'] == dataset].copy()
    if dataset_df.empty: return

    # Define Metrics to Plot
    metrics_map = {
        'acc': 'Model Accuracy',
        'asr': 'Attack Success Rate',
        'benign_selection_rate': 'Benign Selection Rate',
        'adv_selection_rate': 'Attacker Selection Rate',
    }

    valid_biases = ['Market-Wide Bias', 'Buyer-Only Bias', 'Seller-Only Bias']
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # --- OUTER LOOP: BIAS SOURCE ---
    for bias in valid_biases:
        bias_df = dataset_df[dataset_df['bias_source'] == bias]
        if bias_df.empty: continue

        # --- INNER LOOP: METRIC ---
        for col_name, display_name in metrics_map.items():
            if col_name not in bias_df.columns: continue

            # Create a dedicated figure
            plt.figure(figsize=(6, 4))

            sns.lineplot(
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
                markersize=8,
                linewidth=2,
                errorbar=('ci', 95)
            )

            # Titles
            plt.title(f"{display_name}\nCondition: {bias} ({dataset})", fontsize=12)
            plt.ylabel(display_name, fontsize=11)
            plt.xlabel("Heterogeneity (Dirichlet Alpha)", fontsize=11)

            # Log Scale & Formatting
            ax = plt.gca()
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(FixedLocator(ALPHAS_IN_TEST))
            ax.xaxis.set_major_formatter(FixedFormatter([str(a) for a in ALPHAS_IN_TEST]))

            # Reverse Axis: 100 (Easy) -> 0.1 (Hard)
            ax.set_xlim(max(ALPHAS_IN_TEST) * 1.3, min(ALPHAS_IN_TEST) * 0.8)
            ax.grid(True, which='major', linestyle='--', alpha=0.5)

            plt.legend(title="Defense", bbox_to_anchor=(1.02, 1), loc='upper left')

            # Filename Generation
            safe_bias = bias.replace(' ', '').replace('-', '')
            safe_metric = display_name.replace(' ', '')

            fname = output_dir / f"Step11_{dataset}_{safe_bias}_{safe_metric}.pdf"

            plt.savefig(fname, bbox_inches='tight')
            plt.close()  # Close figure to free memory
            print(f"  Saved: {fname.name}")


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No valid data found.")
        return

    # Save summary CSV
    df.to_csv(output_dir / "step11_full_summary.csv", index=False)

    # Generate Plots
    for dataset in df['dataset'].unique():
        if dataset != 'unknown':
            plot_completely_separate(df, dataset, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()