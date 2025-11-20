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

# Define the relative 'usability' threshold
RELATIVE_ACC_THRESHOLD = 0.90

# ---------------------

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses the HP suffix folder name (e.g., 'opt_Adam_lr_0.001_epochs_2')
    """
    hps = {}
    pattern = r'opt_(\w+)_lr_([0-9\.]+)_epochs_([0-9]+)'
    match = re.search(pattern, hp_folder_name)

    if match:
        hps['optimizer'] = match.group(1)
        hps['learning_rate'] = float(match.group(2))
        hps['local_epochs'] = int(match.group(3))
    else:
        pass # Silence warnings for cleaner output
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    Parses the base scenario name (e.g., 'step2.5_find_hps_martfl_image_CIFAR10')
    """
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
            raise ValueError(f"Pattern not matched for: {scenario_name}")

    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """
    Loads metrics from final_metrics.json (acc, rounds)
    AND calculates both benign and adversary selection rates from marketplace_report.json.
    """
    run_data = {}
    try:
        # 1. Load basic metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        # 2. Load Selection Rates
        report_file = metrics_file.parent / "marketplace_report.json"

        if not report_file.exists():
            run_data['benign_selection_rate'] = np.nan
            run_data['adv_selection_rate'] = np.nan
            return run_data

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = report.get('seller_summaries', {}).values()
        adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
        ben_sellers = [s for s in sellers if s.get('type') == 'benign']

        run_data['adv_selection_rate'] = np.mean(
            [s['selection_rate'] for s in adv_sellers]
        ) if adv_sellers else 0.0

        run_data['benign_selection_rate'] = np.mean(
            [s['selection_rate'] for s in ben_sellers]
        ) if ben_sellers else np.nan

        return run_data
    except Exception as e:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """
    Walks the results directory, aggregates all run data,
    AND filters out any folders ending in '_nolocalclip'.
    """
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [
        f for f in base_path.glob("step2.5_find_hps_*")
        if f.is_dir() and not f.name.endswith("_nolocalclip")
    ]

    if not scenario_folders:
        print(f"Error: No 'step2.5_find_hps_*' directories (excluding '_nolocalclip') found.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts:
                    continue

                hp_folder_name = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder_name)
                run_metrics = load_run_data(metrics_file)

                if run_metrics:
                    all_runs.append({
                        **run_scenario,
                        **run_hps,
                        **run_metrics,
                    })
            except Exception as e:
                continue

    if not all_runs:
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)

    print("Calculating thresholds...")
    dataset_max_acc = df.groupby('dataset')['acc'].max().to_dict()
    df['dataset_max_acc'] = df['dataset'].map(dataset_max_acc)
    df['platform_usable_threshold'] = df['dataset_max_acc'] * RELATIVE_ACC_THRESHOLD
    df['platform_usable'] = (df['acc'] >= df['platform_usable_threshold'])

    return df


def style_axis(ax, title, xlabel, ylabel):
    """
    Helper to apply the consistent "Publication Quality" typography.
    """
    # Title
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)

    # Labels
    ax.set_xlabel(xlabel, fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel(ylabel, fontsize=22, fontweight='bold', labelpad=15)

    # Ticks
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18)


def plot_platform_usability_with_selection(df: pd.DataFrame, output_dir: Path):
    """
    Plots the 5 key platform metrics using the requested "Sybil Comparison" visual style:
    - Whitegrid, Talk context
    - Bold labels, Large fonts
    - Black edges on bars
    - Bottom Legends
    """
    print("\n--- Plotting Platform Metrics (High-Res Style) ---")

    if df.empty:
        print("No data to plot.")
        return

    # --- 1. CONFIGURATION (Apply Global Theme) ---
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.1)

    # Data Prep
    required_cols = ['acc', 'rounds', 'platform_usable', 'benign_selection_rate', 'adv_selection_rate']
    if not all(col in df.columns for col in required_cols):
        print("Error: Missing required data columns.")
        return

    # --- Metric Calculations ---

    # 1-4: Usability Metrics
    df_usability = df.groupby(['defense', 'dataset'])['platform_usable'].mean().reset_index()
    df_usability['Value'] = df_usability['platform_usable'] * 100
    df_usability['Metric'] = 'Usability Rate (%)'

    df_perf = df[df['platform_usable'] == True].groupby(['defense', 'dataset'])['acc'].mean().reset_index()
    df_perf['Value'] = df_perf['acc'] * 100
    df_perf['Metric'] = 'Avg. Usable Accuracy (%)'

    df_speed = df[df['platform_usable'] == True].groupby(['defense', 'dataset'])['rounds'].mean().reset_index()
    df_speed['Value'] = df_speed['rounds']
    df_speed['Metric'] = 'Avg. Usable Rounds'

    df_speed_stability = df.groupby(['defense', 'dataset'])['rounds'].std().reset_index()
    df_speed_stability['Value'] = df_speed_stability['rounds']
    df_speed_stability['Metric'] = 'Rounds Instability (Std)'

    df_metrics_1_4 = pd.concat([df_usability, df_perf, df_speed, df_speed_stability], ignore_index=True)

    # 5: Selection Metrics
    df_selection_raw = df.groupby(['defense', 'dataset'])[
        ['benign_selection_rate', 'adv_selection_rate']].mean().reset_index()

    df_selection_raw = df_selection_raw.rename(columns={
        'benign_selection_rate': 'Benign',
        'adv_selection_rate': 'Adversary'
    })

    df_selection_melted = df_selection_raw.melt(
        id_vars=['defense', 'dataset'],
        value_vars=['Benign', 'Adversary'],
        var_name='Seller Type',
        value_name='Selection Rate'
    )
    df_selection_melted['Selection Rate'] *= 100

    # --- SAVE SUMMARY CSV ---
    csv_path = output_dir / "step2.5_metrics_summary.csv"
    # (Simplified CSV save for brevity, focusing on plotting)
    df_metrics_1_4.to_csv(csv_path, index=False)

    # --- PLOTTING LOOP ---
    all_datasets = df_metrics_1_4['dataset'].unique()

    # Pretty formatter for defense names
    def format_defense(d):
        d_map = {'fedavg': 'FedAvg', 'fltrust': 'FLTrust', 'martfl': 'MARTFL', 'skymask': 'SkyMask'}
        return d_map.get(d.lower(), d.title())

    for dataset in all_datasets:
        # Determine defense order
        if dataset in ['CIFAR10', 'CIFAR100']:
            defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
        else:
            defense_order = ['fedavg', 'fltrust', 'martfl']

        # Format x-tick labels
        formatted_labels = [format_defense(d) for d in defense_order]

        # === A. Plot Metrics 1-4 (Single Bar Charts) ===
        for metric in df_metrics_1_4['Metric'].unique():
            plot_df = df_metrics_1_4[
                (df_metrics_1_4['dataset'] == dataset) &
                (df_metrics_1_4['Metric'] == metric)
            ]
            # Filter valid defenses
            plot_df = plot_df[plot_df['defense'].isin(defense_order)]

            if plot_df.empty: continue

            # Cleanup Filename
            safe_metric = re.sub(r'[^\w]', '', metric.split(' ')[0]) + "_" + re.sub(r'[^\w]', '', metric.split(' ')[-1])

            plt.figure(figsize=(10, 8)) # Consistent sizing

            ax = sns.barplot(
                data=plot_df,
                x='defense',
                y='Value',
                order=defense_order,
                palette='viridis',
                edgecolor="black", # KEY: Black border
                linewidth=1.2      # KEY: Thicker line
            )

            # Apply Labels & Style
            y_label = "Value"
            if "%" in metric: y_label = "Percentage (%)"
            elif "Rounds" in metric: y_label = "Rounds"

            style_axis(ax, f"{metric} ({dataset})", "Defense Strategy", y_label)

            # Set correct X-labels
            ax.set_xticklabels(formatted_labels)

            # Annotations
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f'{p.get_height():.1f}',
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom',
                                xytext=(0, 5), textcoords='offset points',
                                fontsize=16, fontweight='bold') # Bold annotation

            # Save
            plt.tight_layout()
            plt.savefig(output_dir / f"plot_{dataset}_{safe_metric}.pdf", bbox_inches='tight', format='pdf', dpi=300)
            plt.close('all')

        # === B. Plot Metric 5 (Grouped Selection Rates) ===
        sel_df = df_selection_melted[
            (df_selection_melted['dataset'] == dataset) &
            (df_selection_melted['defense'].isin(defense_order))
        ]

        if not sel_df.empty:
            plt.figure(figsize=(12, 8)) # Wider for grouped bars

            ax = sns.barplot(
                data=sel_df,
                x='defense',
                y='Selection Rate',
                hue='Seller Type',
                order=defense_order,
                palette={'Benign': '#2ecc71', 'Adversary': '#e74c3c'}, # Nice Green/Red
                edgecolor="black",
                linewidth=1.2
            )

            style_axis(ax, f"Avg. Selection Rates ({dataset})", "Defense Strategy", "Selection Rate (%)")
            ax.set_xticklabels(formatted_labels)
            ax.set_ylim(0, 105)

            # Annotations
            for p in ax.patches:
                h = p.get_height()
                if h > 0: # Only annotate non-zero
                    ax.annotate(f'{h:.1f}',
                                (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom',
                                xytext=(0, 5), textcoords='offset points',
                                fontsize=14, fontweight='bold')

            # Legend Styling (Bottom, Horizontal)
            plt.legend(
                title=None,
                fontsize=18,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.22), # Move below X-axis
                ncol=2,
                frameon=False
            )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.25) # Make room for legend

            plt.savefig(output_dir / f"plot_{dataset}_Selection_Rates.pdf", format='pdf', dpi=300)
            plt.close('all')
            print(f"  Saved metrics for {dataset}")

def plot_composite_row(df: pd.DataFrame, output_dir: Path):
    """
    Generates a SINGLE wide figure with 4 subplots in a row:
    1. Usability Rate
    2. Avg. Usable Accuracy
    3. Avg. Usable Rounds
    4. Selection Rates
    """
    print("\n--- Plotting Composite Row (4-in-1) ---")
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.4) # 'paper' context fits better for composite

    # Filter for CIFAR100 (or loop datasets)
    dataset = "CIFAR100"
    subset = df[df['dataset'] == dataset]

    if subset.empty:
        return

    # Defense Order
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    formatted_labels = ['FedAvg', 'FLTrust', 'MARTFL', 'SkyMask']

    # Create Figure: 1 Row, 4 Columns. Wide aspect ratio.
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)

    # --- DATA PREP ---
    # 1. Usability
    d1 = subset.groupby('defense')['platform_usable'].mean().reindex(defense_order).reset_index()
    d1['Value'] = d1['platform_usable'] * 100

    # 2. Accuracy
    d2 = subset[subset['platform_usable']==True].groupby('defense')['acc'].mean().reindex(defense_order).reset_index()
    d2['Value'] = d2['acc'] * 100

    # 3. Rounds
    d3 = subset[subset['platform_usable']==True].groupby('defense')['rounds'].mean().reindex(defense_order).reset_index()
    d3['Value'] = d3['rounds']

    # 4. Selection
    d4 = subset.groupby('defense')[['benign_selection_rate', 'adv_selection_rate']].mean().reindex(defense_order).reset_index()
    d4 = d4.melt(id_vars='defense', var_name='Type', value_name='Rate')
    d4['Rate'] = d4['Rate'] * 100
    d4['Type'] = d4['Type'].replace({'benign_selection_rate': 'Benign', 'adv_selection_rate': 'Adversary'})

    # --- PLOTTING ---

    # Plot 1: Usability
    sns.barplot(ax=axes[0], data=d1, x='defense', y='Value', order=defense_order, palette='viridis', edgecolor='black')
    axes[0].set_title("Usability Rate (%)", fontweight='bold')
    axes[0].set_ylabel("Percentage", fontweight='bold')
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 105)

    # Plot 2: Accuracy
    sns.barplot(ax=axes[1], data=d2, x='defense', y='Value', order=defense_order, palette='viridis', edgecolor='black')
    axes[1].set_title("Avg. Usable Acc (%)", fontweight='bold')
    axes[1].set_ylabel("") # Save space
    axes[1].set_xlabel("")
    # Auto scale Y to zoom in on differences (e.g. 40-60) if needed, or keep 0-100
    axes[1].set_ylim(0, 105)

    # Plot 3: Rounds
    sns.barplot(ax=axes[2], data=d3, x='defense', y='Value', order=defense_order, palette='viridis', edgecolor='black')
    axes[2].set_title("Avg. Cost (Rounds)", fontweight='bold')
    axes[2].set_ylabel("Rounds", fontweight='bold')
    axes[2].set_xlabel("")

    # Plot 4: Selection
    sns.barplot(ax=axes[3], data=d4, x='defense', y='Rate', hue='Type', order=defense_order,
                palette={'Benign': '#2ecc71', 'Adversary': '#e74c3c'}, edgecolor='black')
    axes[3].set_title("Avg. Selection Rates", fontweight='bold')
    axes[3].set_ylabel("")
    axes[3].set_xlabel("")
    axes[3].set_ylim(0, 105)
    axes[3].legend(title=None, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False) # Legend ABOVE plot to save vertical space

    # --- COMMON STYLING ---
    letters = ['(a)', '(b)', '(c)', '(d)']
    for i, ax in enumerate(axes):
        ax.set_xticklabels(formatted_labels, fontsize=14, fontweight='bold', rotation=15)
        ax.grid(axis='y', alpha=0.5)
        # Add letter label inside plot (top left) to save space
        # ax.text(-0.1, 1.05, letters[i], transform=ax.transAxes, size=16, weight='bold')

        # Annotations
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f'{h:.0f}', (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 2), textcoords='offset points')

    # Save
    outfile = output_dir / f"plot_row_combined_{dataset}.pdf"
    plt.savefig(outfile, bbox_inches='tight', format='pdf', dpi=300)
    print(f"Saved composite row to: {outfile}")

# --- Add this to your main() function ---
# plot_composite_row(df, output_dir)

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    plot_platform_usability_with_selection(df, output_dir)
    plot_composite_row(df, output_dir)
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()