import json
import os
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
        # Expected: step12_main_summary_{defense}_{modality}_{dataset}_{model}
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


def load_valuations_from_jsonl(run_dir: Path) -> Dict[str, float]:
    """
    Reads valuations.jsonl to calculate average metrics and selection rates.
    Returns a flat dictionary like {'Benign_loo': 0.5, 'Adversary_selected': 0.1}
    """
    jsonl_path = run_dir / "valuations.jsonl"
    if not jsonl_path.exists():
        return {}

    # Accumulators
    metrics_acc = {
        'Benign': {'selected': [], 'total_rounds': 0},
        'Adversary': {'selected': [], 'total_rounds': 0}
    }

    # We need to dynamically find keys like 'loo', 'shap', 'influence'
    metric_keys = set()

    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Record structure:
                    # { "round": 10, "selected_ids": [...], "seller_valuations": {"s1": {"loo":...}} }

                    selected_set = set(record.get('selected_ids', []))
                    valuations = record.get('seller_valuations', {})

                    if not valuations: continue

                    # Iterate through all sellers in this round
                    for seller_id, scores in valuations.items():
                        # Determine Type
                        s_type = 'Adversary' if str(seller_id).startswith('adv') else 'Benign'

                        # 1. Track Selection
                        is_selected = 1 if seller_id in selected_set else 0
                        metrics_acc[s_type]['selected'].append(is_selected)

                        # 2. Track Valuation Scores
                        for key, value in scores.items():
                            # Ignore generic keys if any
                            if key in ['round', 'seller_id']: continue

                            metric_keys.add(key)
                            if key not in metrics_acc[s_type]:
                                metrics_acc[s_type][key] = []

                            if value is not None:
                                metrics_acc[s_type][key].append(float(value))

                except json.JSONDecodeError:
                    continue

        # --- Compute Averages ---
        results = {}
        for s_type in ['Benign', 'Adversary']:
            # Selection Rate
            selections = metrics_acc[s_type]['selected']
            if selections:
                results[f"{s_type}_selected"] = sum(selections) / len(selections)
            else:
                results[f"{s_type}_selected"] = 0.0

            # Metric Averages (LOO, Shap, etc.)
            for key in metric_keys:
                vals = metrics_acc[s_type].get(key, [])
                if vals:
                    results[f"{s_type}_{key}"] = sum(vals) / len(vals)
                else:
                    # If missing (e.g., LOO didn't run for this type?), set NaN
                    results[f"{s_type}_{key}"] = np.nan

        return results

    except Exception as e:
        print(f"Error parsing JSONL in {run_dir.name}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks directory, combines JSON global metrics with JSONL detailed metrics."""
    all_runs = []
    base_path = Path(base_dir)
    scenario_folders = [f for f in base_path.glob("step12_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenarios to process.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)

        # 1. Load Global Metrics (Accuracy, Rounds)
        acc = 0
        rounds = 0
        final_metrics_path = scenario_path / "final_metrics.json"
        if final_metrics_path.exists():
            try:
                with open(final_metrics_path, 'r') as f:
                    metrics = json.load(f)
                acc = metrics.get('acc', 0)
                rounds = metrics.get('completed_rounds', 0)
            except:
                pass

        # 2. Load Valuation & Selection (FROM JSONL)
        val_metrics = load_valuations_from_jsonl(scenario_path)

        # Combine
        flat_record = {**run_scenario, "acc": acc, "rounds": rounds, **val_metrics}
        all_runs.append(flat_record)

    return pd.DataFrame(all_runs)


# --- COMPOSITE PLOTTING FUNCTIONS (Unchanged but verified) ---

def plot_performance_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Generates a 1x3 Row: [Accuracy, Rounds, Selection Gap]
    """
    print(f"\n--- Plotting Performance Row: {dataset} ---")
    set_plot_style()

    subset = df[df['dataset'] == dataset].copy()
    if subset.empty: return

    # Calculate Selection Gap (Benign - Adversary)
    if 'Benign_selected' in subset.columns and 'Adversary_selected' in subset.columns:
        subset['Selection Gap'] = (subset['Benign_selected'] - subset['Adversary_selected']) * 100
        subset['Benign Selection'] = subset['Benign_selected'] * 100
    else:
        subset['Selection Gap'] = np.nan

    # Ensure Accuracy is %
    if subset['acc'].max() <= 1.0:
        subset['acc'] *= 100

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
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

    # 3. Selection Gap
    sns.barplot(ax=axes[2], data=subset, x='defense', y='Selection Gap', order=defense_order, palette=CUSTOM_PALETTE)
    axes[2].set_title("Selection Advantage\n(Benign - Adversary)", fontweight='bold')
    axes[2].set_ylabel("Percentage Points")
    axes[2].set_xlabel("")
    axes[2].axhline(0, color='black', linewidth=1.5)

    # Formatting
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

    # Dynamic detection of metrics present in the DF
    potential_roots = set()
    for col in subset.columns:
        if col.startswith('Benign_'):
            root = col.replace('Benign_', '')
            if root != 'selected':
                potential_roots.add(root)

    roots = sorted(list(potential_roots))
    if not roots: return

    fig, axes = plt.subplots(1, len(roots), figsize=(6 * len(roots), 5), constrained_layout=True)
    if len(roots) == 1: axes = [axes]

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    defense_order = [d for d in defense_order if d in subset['defense'].unique()]

    for i, root in enumerate(roots):
        ax = axes[i]
        ben_col = f"Benign_{root}"
        adv_col = f"Adversary_{root}"

        # Melt for grouped bar plot
        melted = subset.melt(
            id_vars=['defense'],
            value_vars=[ben_col, adv_col],
            var_name='Type',
            value_name='Score'
        )
        melted['Type'] = melted['Type'].map({ben_col: 'Benign', adv_col: 'Adversary'})

        sns.barplot(
            ax=ax, data=melted, x='defense', y='Score', hue='Type',
            order=defense_order, palette=TYPE_PALETTE
        )

        # Formatting
        title = root.replace('_', ' ').replace('sim', 'Similarity').title()
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Average Score")
        ax.axhline(0, color='black', linewidth=1)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        labels = [l.get_text().capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask",
                                                                                                              "SkyMask").replace(
            "Martfl", "MARTFL") for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

        if i == 0:
            ax.legend(title=None)
        else:
            ax.get_legend().remove()

    fname = output_dir / f"Step12_Valuation_Row_{dataset}.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"  Saved: {fname.name}")
    plt.close()


# --- MAIN EXECUTION ---
def plot_valuation_distribution(df_raw_points: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Generates a Distribution (Strip) Plot to show Fairness/Differentiation.
    Plots every individual seller's score as a dot.
    """
    print(f"\n--- Plotting Valuation Distribution: {dataset} ---")

    # Filter for the specific dataset
    subset = df_raw_points[df_raw_points['dataset'] == dataset].copy()
    if subset.empty: return

    # We likely have multiple valuation metrics (loo, shap, etc.), pick the most prominent one
    # Heuristic: Find columns ending in '_score' or check known keys
    val_keys = subset['metric_type'].unique()

    defense_order = ['FedAvg', 'FLTrust', 'MARTFL', 'SkyMask']

    for metric in val_keys:
        metric_subset = subset[subset['metric_type'] == metric]

        plt.figure(figsize=(12, 6))

        # Strip Plot: Shows individual points
        ax = sns.stripplot(
            data=metric_subset, x='defense', y='score', hue='type',
            order=[d for d in defense_order if d in metric_subset['defense'].unique()],
            palette={"Benign": "#2ca02c", "Adversary": "#d62728"},
            dodge=True, alpha=0.6, jitter=0.25, size=6
        )

        # Add a Box Plot behind it to show quartiles transparently
        sns.boxplot(
            data=metric_subset, x='defense', y='score', hue='type',
            order=[d for d in defense_order if d in metric_subset['defense'].unique()],
            palette={"Benign": "#2ca02c", "Adversary": "#d62728"},
            dodge=True, ax=ax, boxprops={'facecolor': 'none'},
            fliersize=0, zorder=0, linewidth=1.5
        )

        # Fix Legends (remove duplicates from combining strip/box)
        handles, labels = ax.get_legend_handles_labels()
        # We only want the first 2 handles (Benign, Adversary)
        ax.legend(handles[:2], labels[:2], title="Seller Type", loc='upper right')

        ax.set_title(f"Valuation Distribution ({metric}) - {dataset}", fontweight='bold')
        ax.set_ylabel("Valuation Score")
        ax.set_xlabel("Defense Method")
        ax.axhline(0, color='black', linewidth=1, linestyle='--')

        fname = output_dir / f"Step12_Distribution_{dataset}_{metric}.pdf"
        plt.savefig(fname, bbox_inches='tight', format='pdf')
        print(f"  Saved: {fname.name}")
        plt.close()


# --- HELPER: Load Raw Points (Not Averages) ---
def load_raw_valuations(run_dir: Path, scenario_info: Dict) -> List[Dict]:
    """Extracts INDIVIDUAL scores for distribution plotting."""
    jsonl_path = run_dir / "valuations.jsonl"
    if not jsonl_path.exists(): return []

    raw_points = []
    # Only read the last 10 rounds to get "converged" valuations
    # (Reading all rounds makes the plot too messy)

    try:
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        # Take last 20% of rounds
        start_idx = int(len(lines) * 0.8)

        for line in lines[start_idx:]:
            record = json.loads(line)
            valuations = record.get('seller_valuations', {})

            for seller_id, scores in valuations.items():
                s_type = 'Adversary' if str(seller_id).startswith('adv') else 'Benign'

                for key, val in scores.items():
                    if key in ['round', 'seller_id'] or val is None: continue

                    raw_points.append({
                        "defense": scenario_info.get("defense", "Unknown").title(),
                        "dataset": scenario_info.get("dataset", "Unknown"),
                        "type": s_type,
                        "metric_type": key,
                        "score": float(val)
                    })
    except:
        pass

    return raw_points


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # 1. Collect Data (Uses JSONL now)
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data found in Step 12 folders.")
        return

    # 2. Save Summary CSV
    csv_path = output_dir / "step12_full_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    # 3. Generate Plots per Dataset
    for dataset in df['dataset'].unique():
        if dataset == 'unknown': continue
        try:
            plot_performance_row(df, dataset, output_dir)
            plot_valuation_row(df, dataset, output_dir)
        except Exception as e:
            print(f"Error plotting {dataset}: {e}")
    all_raw_points = []

    # 1. Loop folders
    base_path = Path(BASE_RESULTS_DIR)
    scenario_folders = [f for f in base_path.glob("step12_*") if f.is_dir()]

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)
        # Collect raw points for the distribution plot
        raw_data = load_raw_valuations(folder, info)
        all_raw_points.extend(raw_data)

    # 2. Generate the New Plot
    if all_raw_points:
        df_raw = pd.DataFrame(all_raw_points)
        for dataset in df_raw['dataset'].unique():
            plot_valuation_distribution(df_raw, dataset, output_dir)
    print("\n✅ Analysis complete.")


if __name__ == "__main__":
    main()
