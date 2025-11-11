import argparse
import sys
from pathlib import Path
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the metric to plot (your 'score')
METRIC_TO_PLOT = 'score'  # score = mean_test_acc - mean_backdoor_asr
INPUT_FILE = "step3_defense_tuning_all_aggregated.csv"


# --- 1. Heatmap Plotting Function ---
def create_heatmap(df_slice: pd.DataFrame, dataset: str, attack: str, defense: str):
    """
    Generates a heatmap for 2-parameter tuning (e.g., MartFL).
    """
    if df_slice.empty:
        return

    # We need exactly two HP columns to make a heatmap
    hp_cols = [col for col in df_slice.columns if col not in [
        'scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix',
        'mean_test_acc', 'std_test_acc', 'mean_backdoor_asr', 'std_backdoor_asr',
        'num_success_runs', 'score', 'hp_folder'
    ] and not col.startswith('raw_')]

    if len(hp_cols) != 2:
        logger.info(f"Skipping heatmap for {defense} (requires exactly 2 HPs, found {len(hp_cols)}).")
        return

    # Get the two HP columns
    hp_x, hp_y = hp_cols[0], hp_cols[1]

    logger.info(f"Generating heatmap for: {dataset} / {defense} / {attack}...")

    try:
        # Create the pivot table
        pivot_df = df_slice.pivot(index=hp_y, columns=hp_x, values=METRIC_TO_PLOT)
    except Exception as e:
        logger.error(f"Failed to create pivot table for heatmap: {e}")
        return

    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    # Draw the heatmap
    ax = sns.heatmap(
        pivot_df,
        annot=True,  # Show the score values
        fmt=".2f",  # Format to 2 decimal places
        cmap="Greys",  # Black-and-white colormap
        linewidths=.5,
        cbar_kws={'label': f"Score ({METRIC_TO_PLOT})"}
    )

    ax.set_title(f"Defense Tuning Heatmap: {defense} (on {dataset} / {attack})")
    ax.set_xlabel(hp_x.replace("_", " ").title())
    ax.set_ylabel(hp_y.replace("_", " ").title())

    # Save the figure
    output_dir = Path("figures") / "step3_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"heatmap_{dataset}_{defense}_{attack}.png"
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()


# --- 2. Final Comparison Bar Plot Function ---
def create_comparison_plot(best_df: pd.DataFrame, dataset: str, attack: str):
    """
    Generates a grouped bar plot comparing the *best* performance of each
    defense against the FedAvg baseline for a specific scenario.
    """
    if best_df.empty:
        return

    logger.info(f"Generating final comparison plot for: {dataset} / {attack}...")

    # "Melt" the dataframe to plot acc and asr side-by-side
    plot_df = best_df.melt(
        id_vars=['defense'],
        value_vars=['mean_test_acc', 'mean_backdoor_asr'],
        var_name='Metric',
        value_name='Performance'
    )

    # Make metric names prettier
    plot_df['Metric'] = plot_df['Metric'].map({
        'mean_test_acc': 'Test Accuracy (Utility)',
        'mean_backdoor_asr': 'ASR (Robustness)'
    })

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=plot_df,
        x='defense',
        y='Performance',
        hue='Metric',
        palette='Greys_r',  # B/W palette (reversed)
        edgecolor='black'
    )

    # Add hatches for B/W
    hatches = ['/', '\\\\']
    for i, bar in enumerate(ax.patches):
        bar.set_hatch(hatches[i // len(best_df['defense'].unique())])

    ax.set_title(f"Best Defense Performance vs. FedAvg\n({dataset} / {attack})")
    ax.set_xlabel("Defense")
    ax.set_ylabel("Performance (%)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title="Metric")

    # Save the figure
    output_dir = Path("figures") / "step3_final_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"barchart_comparison_{dataset}_{attack}.png"
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()


# --- Main function to drive plotting ---
def main():
    parser = argparse.ArgumentParser(
        description="Visualize FL Defense Tuning results (Step 3)."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="The ROOT results directory containing the Step 3 run folders (e.g., './results/')"
    )
    args = parser.parse_args()
    results_path = Path(args.results_dir).resolve()

    # Load the aggregated CSV file
    agg_csv_path = results_path / INPUT_FILE
    if not agg_csv_path.exists():
        logger.error(f"❌ ERROR: Could not find aggregated file: {agg_csv_path}")
        logger.error("Please run the `analyze_step3_defense_tuning.py` script first.")
        return

    try:
        agg_df = pd.read_csv(agg_csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    # --- Find the "best" row for each (defense, dataset, attack_type, model) ---
    group_cols = ['defense', 'attack_type', 'modality', 'dataset', 'model_suffix']
    try:
        best_indices = agg_df.loc[agg_df.groupby(group_cols)['score'].idxmax()]
    except Exception as e:
        logger.error(f"Failed to find best scores (is the 'score' column present?): {e}")
        return

    # --- Loop and create plots ---
    # We create one set of plots for each (dataset, attack_type) pair
    scenarios = best_df[['dataset', 'attack_type', 'modality', 'model_suffix']].drop_duplicates()

    for _, row in scenarios.iterrows():
        dataset = row['dataset']
        attack = row['attack_type']
        modality = row['modality']
        model = row['model_suffix']

        # 1. Get the data for the final bar chart (the "best" HPs)
        best_df_slice = best_df[
            (best_df['dataset'] == dataset) &
            (best_df['attack_type'] == attack) &
            (best_df['model_suffix'] == model)
            ]

        # 2. Get the *full* data for the heatmaps (all HPs)
        full_df_slice = agg_df[
            (agg_df['dataset'] == dataset) &
            (agg_df['attack_type'] == attack) &
            (agg_df['model_suffix'] == model)
            ]

        # Create the final comparison bar plot
        create_comparison_plot(best_df_slice, dataset, attack)

        # Create heatmaps for defenses that have 2 HPs (like MartFL)
        for defense in full_df_slice['defense'].unique():
            if defense == 'fedavg': continue
            defense_df = full_df_slice[full_df_slice['defense'] == defense]
            create_heatmap(defense_df, dataset, attack, defense)

    logger.info("\n✅ All visualizations generated.")


if __name__ == "__main__":
    main()