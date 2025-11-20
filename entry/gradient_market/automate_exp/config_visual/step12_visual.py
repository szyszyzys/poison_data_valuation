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

# --- VISUAL CONSISTENCY ---
CUSTOM_PALETTE = {
    "fedavg": "#4c72b0",  # Deep Blue
    "fltrust": "#dd8452",  # Deep Orange
    "martfl": "#55a868",  # Deep Green
    "skymask": "#c44e52"  # Deep Red
}

TYPE_PALETTE = {
    "Benign": "#2ca02c",  # Green
    "Adversary": "#d62728"  # Red
}


def set_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['lines.linewidth'] = 2.5


# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses Step 12 folder names."""
    try:
        parts = scenario_name.split('_')
        if 'step12' in parts and 'main' in parts:
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
        return {"scenario": scenario_name, "defense": "unknown", "dataset": "unknown"}
    except Exception:
        return {"scenario": scenario_name, "defense": "unknown", "dataset": "unknown"}


def load_metrics_from_csv(run_dir: Path) -> pd.DataFrame:
    """Scans seller_metrics.csv for valuation and selection rates."""
    csv_path = run_dir / "seller_metrics.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        if df.empty or 'seller_id' not in df.columns:
            return pd.DataFrame()

        df['type'] = df['seller_id'].apply(lambda x: 'Adversary' if str(x).startswith('adv') else 'Benign')

        target_keywords = ['influence', 'shap', 'loo', 'sim_']
        val_cols = [c for c in df.columns if any(x in c.lower() for x in target_keywords)]

        if 'selected' in df.columns:
            df['selected'] = df['selected'].astype(int)
            val_cols.append('selected')

        if not val_cols:
            return pd.DataFrame()

        summary = df.groupby('type')[val_cols].mean().reset_index()
        return summary

    except Exception:
        return pd.DataFrame()


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks directory, combines JSON global metrics with CSV detailed metrics."""
    all_runs = []
    base_path = Path(base_dir)
    scenario_folders = [f for f in base_path.glob("step12_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenarios to process.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            run_dir = metrics_file.parent
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                acc = metrics.get('acc', 0)
                rounds = metrics.get('completed_rounds', 0)
            except:
                acc = 0;
                rounds = 0

            df_val = load_metrics_from_csv(run_dir)
            flat_record = {**run_scenario, "acc": acc, "rounds": rounds}

            if not df_val.empty:
                for _, row in df_val.iterrows():
                    s_type = row['type']
                    for col in df_val.columns:
                        if col != 'type':
                            flat_record[f"{s_type}_{col}"] = row[col]

            all_runs.append(flat_record)

    return pd.DataFrame(all_runs)


# --- COMPOSITE PLOTTING FUNCTIONS ---

def plot_performance_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Generates a 1x3 Row: [Accuracy, Rounds, Selection Gap]
    """
    print(f"\n--- Plotting Performance Row: {dataset} ---")
    set_plot_style()

    subset = df[df['dataset'] == dataset].copy()
    if subset.empty: return

    # Calculate Selection Gap (Benign - Adversary) if columns exist
    if 'Benign_selected' in subset.columns and 'Adversary_selected' in subset.columns:
        subset['Selection Gap'] = (subset['Benign_selected'] - subset['Adversary_selected']) * 100
        subset['Benign Selection'] = subset['Benign_selected'] * 100
    else:
        subset['Selection Gap'] = np.nan

    # Ensure Accuracy is %
    if subset['acc'].max() <= 1.0:
        subset['acc'] *= 100

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    # Filter order to what exists
    defense_order = [d for d in defense_order if d in subset['defense'].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # 1. Global Accuracy
    sns.barplot(ax=axes[0], data=subset, x='defense', y='acc', order=defense_order, palette=CUSTOM_PALETTE)
    axes[0].set_title("Global Accuracy (%)", fontweight='bold')
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 100)

    # 2. Rounds
    sns.barplot(ax=axes[1], data=subset, x='defense', y='rounds', order=defense_order, palette=CUSTOM_PALETTE)
    axes[1].set_title("Rounds to Converge", fontweight='bold')
    axes[1].set_ylabel("Rounds")
    axes[1].set_xlabel("")

    # 3. Selection Gap (Benign - Adversary)
    # Positive is good (filtering bad guys), Negative is bad (filtering good guys)
    sns.barplot(ax=axes[2], data=subset, x='defense', y='Selection Gap', order=defense_order, palette=CUSTOM_PALETTE)
    axes[2].set_title("Selection Advantage\n(Benign - Adversary)", fontweight='bold')
    axes[2].set_ylabel("Percentage Points")
    axes[2].set_xlabel("")
    axes[2].axhline(0, color='black', linewidth=1.5)

    # Format X Labels
    for ax in axes:
        labels = [l.get_text().capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask",
                                                                                                              "SkyMask").replace(
            "Martfl", "MARTFL") for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fname = output_dir / f"Step12_Performance_Row_{dataset}.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"  Saved: {fname.name}")
    plt.close()


def plot_valuation_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Generates a Multi-Panel Figure for Valuation Metrics (Influence, Shapley, Similarity).
    """
    print(f"\n--- Plotting Valuation Row: {dataset} ---")
    set_plot_style()

    subset = df[df['dataset'] == dataset].copy()
    if subset.empty: return

    # Identify available valuation metrics
    # We look for pairs of (Benign_X, Adversary_X)
    potential_roots = set()
    for col in subset.columns:
        if col.startswith('Benign_'):
            root = col.replace('Benign_', '')
            if root != 'selected':
                potential_roots.add(root)

    # Sort for consistency: e.g., ['influence_score', 'shapley_value', 'sim_cosine']
    roots = sorted(list(potential_roots))

    if not roots:
        return

    # Dynamic Figure Size based on number of metrics
    fig, axes = plt.subplots(1, len(roots), figsize=(6 * len(roots), 5), constrained_layout=True)
    if len(roots) == 1: axes = [axes]

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    defense_order = [d for d in defense_order if d in subset['defense'].unique()]

    for i, root in enumerate(roots):
        ax = axes[i]
        ben_col = f"Benign_{root}"
        adv_col = f"Adversary_{root}"

        # Melt for side-by-side bars
        melted = subset.melt(
            id_vars=['defense'],
            value_vars=[ben_col, adv_col],
            var_name='Type',
            value_name='Score'
        )
        melted['Type'] = melted['Type'].map({ben_col: 'Benign', adv_col: 'Adversary'})

        sns.barplot(
            ax=ax,
            data=melted,
            x='defense',
            y='Score',
            hue='Type',
            order=defense_order,
            palette=TYPE_PALETTE  # Green vs Red
        )

        # Formatting
        title = root.replace('_', ' ').replace('sim', 'Similarity').title()
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Average Score")
        ax.axhline(0, color='black', linewidth=1)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Clean X Labels
        labels = [l.get_text().capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask",
                                                                                                              "SkyMask").replace(
            "Martfl", "MARTFL") for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

        # Legend only on first plot
        if i == 0:
            ax.legend(title=None)
        else:
            ax.get_legend().remove()

    fname = output_dir / f"Step12_Valuation_Row_{dataset}.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"  Saved: {fname.name}")
    plt.close()


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

    # 3. Generate Plots per Dataset
    for dataset in df['dataset'].unique():
        if dataset == 'unknown': continue
        plot_performance_row(df, dataset, output_dir)
        plot_valuation_row(df, dataset, output_dir)

    print("\n✅ Analysis complete.")


if __name__ == "__main__":
    main()