import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step5_figures"

# --- Styling Helper ---
def set_plot_style():
    """Sets a consistent professional style for all plots."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 9

# --- Parsing Functions ---

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses the HP suffix folder name (e.g., 'adv_0.1_poison_0.5')
    """
    hps = {}
    pattern = r'adv_([0-9\.]+)_poison_([0-9\.]+)'
    match = re.search(pattern, hp_folder_name)

    if match:
        hps['adv_rate'] = float(match.group(1))
        hps['poison_rate'] = float(match.group(2))
    else:
        # Fallback for potential other naming conventions if needed
        pass
    return hps


def parse_scenario_name(scenario_name: str) -> Optional[Dict[str, str]]:
    """
    Parses the base scenario name for step5.
    """
    try:
        # Regex to match step5 folders
        pattern = r'step5_atk_sens_(adv|poison)_(fedavg|fltrust|martfl|skymask)_(backdoor|labelflip)_(image|text|tabular)$'
        match = re.search(pattern, scenario_name)

        if match:
            modality = match.group(4)

            # Map modality to dataset name based on your setup
            if modality == 'image':
                dataset_name = 'CIFAR100' # Assumed based on previous context
            elif modality == 'text':
                dataset_name = 'TREC'
            elif modality == 'tabular':
                dataset_name = 'Texas100'
            else:
                dataset_name = 'unknown'

            return {
                "scenario": scenario_name,
                "sweep_type": match.group(1), # 'adv' or 'poison'
                "defense": match.group(2),
                "attack": match.group(3),
                "modality": modality,
                "dataset": dataset_name,
            }
        else:
            return None

    except Exception as e:
        print(f"Warning: Error parsing scenario name '{scenario_name}': {e}")
        return None


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """
    Loads key data from final_metrics.json and marketplace_report.json
    """
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

            run_data['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv_sellers]) if adv_sellers else 0.0
            run_data['benign_selection_rate'] = np.mean([s['selection_rate'] for s in ben_sellers]) if ben_sellers else 0.0
        else:
            # Default if no report exists
            run_data['adv_selection_rate'] = 0.0
            run_data['benign_selection_rate'] = 0.0

        return run_data

    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step5_atk_sens_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step5_atk_sens_*' directories found.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)

        if run_scenario is None:
            continue

        files_in_scenario = list(scenario_path.rglob("final_metrics.json"))

        for metrics_file in files_in_scenario:
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                hp_folder_name = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder_name) # Extract adv_rate and poison_rate

                if not run_hps: continue

                run_metrics = load_run_data(metrics_file)

                if run_metrics:
                    all_runs.append({
                        **run_scenario,
                        **run_hps,
                        **run_metrics,
                        "hp_suffix": hp_folder_name
                    })
            except Exception as e:
                print(f"Error processing file {metrics_file}: {e}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    return df


# --- Plotting Function (The Update) ---

def plot_sensitivity_composite_row(df: pd.DataFrame, dataset: str, attack: str, output_dir: Path):
    """
    Generates a SINGLE wide figure (1x4) for the Sensitivity Analysis of a specific dataset/attack pair.
    Plots:
      1. ASR vs. Adversary Rate
      2. Benign Selection vs. Adversary Rate
      3. ASR vs. Poison Rate
      4. Accuracy vs. Poison Rate
    """
    print(f"\n--- Plotting Composite Sensitivity Row: {dataset} ({attack}) ---")

    # Filter data for this specific scenario
    subset = df[(df['dataset'] == dataset) & (df['attack'] == attack)].copy()

    if subset.empty:
        print("  -> No data found for this combination.")
        return

    # Convert rates to percentages for better readability
    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        if col in subset.columns:
            subset[col] = subset[col] * 100

    # Split into the two sweep types based on folder name
    # 'sweep_type' comes from the folder name (step5_atk_sens_adv_... vs step5_atk_sens_poison_...)
    df_adv_sweep = subset[subset['sweep_type'] == 'adv']
    df_poison_sweep = subset[subset['sweep_type'] == 'poison']

    if df_adv_sweep.empty and df_poison_sweep.empty:
        print("  -> No valid sweep data found.")
        return

    set_plot_style()

    # Initialize Figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.5), constrained_layout=True)

    # Determine Defense Order for consistent colors
    defense_order = sorted(subset['defense'].unique())

    # --- PLOT 1 & 2: Adversary Rate Sweep ---
    if not df_adv_sweep.empty:
        # (a) ASR vs Adv Rate
        sns.lineplot(ax=axes[0], data=df_adv_sweep, x='adv_rate', y='asr', hue='defense',
                     style='defense', markers=True, dashes=False, hue_order=defense_order, style_order=defense_order)
        axes[0].set_title("(a) ASR vs. Adversary Rate", fontweight='bold')
        axes[0].set_xlabel("Adversary Rate")
        axes[0].set_ylabel("ASR (%)")
        axes[0].set_ylim(-5, 105)
        axes[0].get_legend().remove()

        # (b) Benign Selection vs Adv Rate
        sns.lineplot(ax=axes[1], data=df_adv_sweep, x='adv_rate', y='benign_selection_rate', hue='defense',
                     style='defense', markers=True, dashes=False, hue_order=defense_order, style_order=defense_order)
        axes[1].set_title("(b) Benign Select vs. Adv Rate", fontweight='bold')
        axes[1].set_xlabel("Adversary Rate")
        axes[1].set_ylabel("Selection Rate (%)")
        axes[1].set_ylim(-5, 105)
        axes[1].get_legend().remove()
    else:
        axes[0].text(0.5, 0.5, "No Adv Rate Data", ha='center', va='center')
        axes[1].text(0.5, 0.5, "No Adv Rate Data", ha='center', va='center')

    # --- PLOT 3 & 4: Poison Rate Sweep ---
    if not df_poison_sweep.empty:
        # (c) ASR vs Poison Rate
        sns.lineplot(ax=axes[2], data=df_poison_sweep, x='poison_rate', y='asr', hue='defense',
                     style='defense', markers=True, dashes=False, hue_order=defense_order, style_order=defense_order)
        axes[2].set_title("(c) ASR vs. Poison Rate", fontweight='bold')
        axes[2].set_xlabel("Poison Rate")
        axes[2].set_ylabel("ASR (%)")
        axes[2].set_ylim(-5, 105)
        axes[2].get_legend().remove()

        # (d) Accuracy vs Poison Rate
        sns.lineplot(ax=axes[3], data=df_poison_sweep, x='poison_rate', y='acc', hue='defense',
                     style='defense', markers=True, dashes=False, hue_order=defense_order, style_order=defense_order)
        axes[3].set_title("(d) Accuracy vs. Poison Rate", fontweight='bold')
        axes[3].set_xlabel("Poison Rate")
        axes[3].set_ylabel("Accuracy (%)")
        axes[3].set_ylim(-5, 105)

        # Handle Legend extraction from this plot
        handles, labels = axes[3].get_legend_handles_labels()
        axes[3].get_legend().remove()
    else:
        axes[2].text(0.5, 0.5, "No Poison Rate Data", ha='center', va='center')
        axes[3].text(0.5, 0.5, "No Poison Rate Data", ha='center', va='center')
        # Attempt to get handles from first plot if 3rd is empty
        if not df_adv_sweep.empty:
            handles, labels = axes[0].get_legend_handles_labels()
        else:
            handles, labels = [], []

    # --- Global Legend ---
    if handles:
        # Capitalize labels
        labels = [l.capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask", "SkyMask").replace("Martfl", "MARTFL") for l in labels]
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                   ncol=len(defense_order), frameon=True, title="Defense Methods")

    # Save
    safe_dataset = re.sub(r'[^\w]', '', dataset)
    filename = output_dir / f"plot_sensitivity_composite_{safe_dataset}_{attack}.pdf"
    plt.savefig(filename, bbox_inches='tight', format='pdf', dpi=300)
    print(f"  -> Saved plot to: {filename}")
    plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # 1. Collect Data
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    # 2. Debug Print
    print("\n--- Data Loaded ---")
    print(f"Total runs: {len(df)}")
    print(f"Datasets found: {df['dataset'].unique()}")
    print(f"Attacks found: {df['attack'].unique()}")

    # 3. Generate Composite Plots for each Dataset/Attack pair
    # Get unique combinations
    combinations = df[['dataset', 'attack']].drop_duplicates().values

    for dataset, attack in combinations:
        # Skip if parsing failed
        if dataset == 'unknown': continue
        if pd.isna(attack): continue

        plot_sensitivity_composite_row(df, dataset, attack, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()