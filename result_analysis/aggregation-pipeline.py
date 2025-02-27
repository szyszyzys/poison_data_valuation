import argparse
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from result_analysis.selection_analysis import analyze_selection_patterns
from result_analysis.visualization import run_visualization

# Set plotting style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_aggregation_comparison(summary_df, output_dir):
    """
    Create visualizations comparing attack performance across different aggregation methods.

    Args:
        summary_df: DataFrame containing summary metrics for all experiments
        output_dir: Directory to save output visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Skip if AGGREGATION_METHOD column is missing or only has one value
    if 'AGGREGATION_METHOD' not in summary_df.columns or len(summary_df['AGGREGATION_METHOD'].unique()) <= 1:
        print("Skipping aggregation comparison: not enough aggregation methods to compare")
        return

    # 1. Overall comparison of ASR and Main Accuracy by Aggregation Method
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    sns.boxplot(x='AGGREGATION_METHOD', y='FINAL_ASR', data=summary_df)
    plt.title('Attack Success Rate by Aggregation Method', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(2, 1, 2)
    sns.boxplot(x='AGGREGATION_METHOD', y='FINAL_MAIN_ACC', data=summary_df)
    plt.title('Main Task Accuracy by Aggregation Method', fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregation_method_comparison.png", dpi=300)
    plt.close()

    # 2. Comparison by Gradient Mode
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    sns.boxplot(x='AGGREGATION_METHOD', y='FINAL_ASR', hue='GRAD_MODE', data=summary_df)
    plt.title('Attack Success Rate by Aggregation Method and Gradient Mode', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(2, 1, 2)
    sns.boxplot(x='AGGREGATION_METHOD', y='FINAL_MAIN_ACC', hue='GRAD_MODE', data=summary_df)
    plt.title('Main Task Accuracy by Aggregation Method and Gradient Mode', fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregation_method_by_grad_mode.png", dpi=300)
    plt.close()

    # 3. Comparison by Sybil Mode
    if 'IS_SYBIL' in summary_df.columns:
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        sns.boxplot(x='AGGREGATION_METHOD', y='FINAL_ASR', hue='IS_SYBIL', data=summary_df)
        plt.title('Attack Success Rate by Aggregation Method and Sybil Mode', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.subplot(2, 1, 2)
        sns.boxplot(x='AGGREGATION_METHOD', y='FINAL_MAIN_ACC', hue='IS_SYBIL', data=summary_df)
        plt.title('Main Task Accuracy by Aggregation Method and Sybil Mode', fontsize=14)
        plt.ylim(0.5, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/aggregation_method_by_sybil.png", dpi=300)
        plt.close()

    # 4. Heatmap Comparison across Aggregation Methods
    if len(summary_df['AGGREGATION_METHOD'].unique()) == 2:
        agg_methods = sorted(summary_df['AGGREGATION_METHOD'].unique())

        # Compare ASR for both gradient modes
        plt.figure(figsize=(20, 8))

        for i, grad_mode in enumerate(['cmd', 'single']):
            if grad_mode not in summary_df['GRAD_MODE'].unique():
                continue

            plt.subplot(1, 2, i + 1)

            grad_data = summary_df[summary_df['GRAD_MODE'] == grad_mode]

            # Create a pivot table
            if 'IS_SYBIL' in grad_data.columns and 'N_ADV' in grad_data.columns:
                pivot = grad_data.pivot_table(
                    values='FINAL_ASR',
                    index=['IS_SYBIL', 'N_ADV'],
                    columns='AGGREGATION_METHOD',
                    aggfunc='mean'
                )

                # Calculate the difference
                pivot['diff'] = pivot[agg_methods[0]] - pivot[agg_methods[1]]

                # Create a heatmap of the difference
                sns.heatmap(pivot['diff'].unstack(level=0),
                            annot=True,
                            cmap='RdBu_r',
                            center=0,
                            fmt='.2f')

                plt.title(f'ASR Difference ({agg_methods[0]} - {agg_methods[1]}) for {grad_mode}', fontsize=14)
                plt.tight_layout()

        plt.savefig(f"{output_dir}/aggregation_method_heatmap_diff.png", dpi=300)
        plt.close()

    # 5. Improved Paired Comparison if we have exactly 2 aggregation methods
    if len(summary_df['AGGREGATION_METHOD'].unique()) == 2:
        agg_methods = sorted(summary_df['AGGREGATION_METHOD'].unique())

        # Group by all parameters except aggregation method
        group_cols = ['GRAD_MODE', 'TRIGGER_RATE', 'IS_SYBIL', 'N_ADV']
        if 'POISON_STRENGTH' in summary_df.columns:
            group_cols.append('POISON_STRENGTH')

        # Only use columns that exist
        group_cols = [col for col in group_cols if col in summary_df.columns]

        # Find configurations that exist in both aggregation methods
        config_counts = summary_df.groupby(group_cols + ['AGGREGATION_METHOD']).size().unstack()
        common_configs = config_counts.dropna().index

        if len(common_configs) > 0:
            # Filter to only configs that exist in both methods
            filtered_df = summary_df.set_index(group_cols)
            filtered_df = filtered_df.loc[filtered_df.index.isin(common_configs)].reset_index()

            # Create paired plot
            plt.figure(figsize=(15, 10))

            for i, metric in enumerate(['FINAL_ASR', 'FINAL_MAIN_ACC']):
                if metric not in filtered_df.columns:
                    continue

                plt.subplot(2, 1, i + 1)

                # Reshape data for paired comparison
                paired_data = filtered_df.pivot_table(
                    index=group_cols,
                    columns='AGGREGATION_METHOD',
                    values=metric
                ).reset_index()

                # Plot paired data points
                plt.scatter(
                    paired_data[agg_methods[0]],
                    paired_data[agg_methods[1]],
                    alpha=0.7
                )

                # Add diagonal line (y=x)
                min_val = min(paired_data[agg_methods[0]].min(), paired_data[agg_methods[1]].min())
                max_val = max(paired_data[agg_methods[0]].max(), paired_data[agg_methods[1]].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

                # Set axis labels
                plt.xlabel(agg_methods[0], fontsize=12)
                plt.ylabel(agg_methods[1], fontsize=12)

                # Set title
                metric_name = "Attack Success Rate" if metric == "FINAL_ASR" else "Main Task Accuracy"
                plt.title(f'Paired Comparison of {metric_name} across Aggregation Methods', fontsize=14)

                # Set equal aspect ratio
                plt.axis('equal')
                plt.grid(linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/aggregation_method_paired_comparison.png", dpi=300)
            plt.close()

    # 6. Progression Comparison (if round data is available)
    # This would require all_rounds_df, but we can add placeholders

    # Print summary of comparisons
    print("\nAggregation Method Comparison Summary:")
    agg_summary = summary_df.groupby('AGGREGATION_METHOD')[['FINAL_ASR', 'FINAL_MAIN_ACC']].mean().round(3)
    print(agg_summary)

    # Print comparison by gradient mode
    print("\nAggregation Method Comparison by Gradient Mode:")
    grad_summary = summary_df.groupby(['AGGREGATION_METHOD', 'GRAD_MODE'])[
        ['FINAL_ASR', 'FINAL_MAIN_ACC']].mean().round(3)
    print(grad_summary)


def integrate_aggregation_comparison(run_visualization_function):
    """
    Decorator to integrate aggregation comparison into the main visualization_226 function.
    """

    def wrapper(summary_csv, all_rounds_csv, output_dir):
        # Run the original visualization_226 function
        run_visualization_function(summary_csv, all_rounds_csv, output_dir)

        # Add aggregation comparison
        summary_df = pd.read_csv(summary_csv)

        if 'AGGREGATION_METHOD' in summary_df.columns and len(summary_df['AGGREGATION_METHOD'].unique()) > 1:
            print("\nGenerating aggregation method comparison visualizations...")
            agg_output_dir = os.path.join(output_dir, "aggregation_comparison")
            plot_aggregation_comparison(summary_df, agg_output_dir)

    return wrapper


def main():
    """Main function to run the entire analysis pipeline"""

    parser = argparse.ArgumentParser(description="Analyze federated learning backdoor attack results")
    parser.add_argument("--local_epoch", type=int, default=5, help="Local epoch setting used in experiments")
    parser.add_argument("--output_dir", default="./processed_results",
                        help="Output directory for processed data and visualizations")
    parser.add_argument("--skip_processing", action="store_true", help="Skip data processing step (use existing CSVs)")
    parser.add_argument("--skip_visualization", action="store_true", help="Skip visualization_226 step")
    parser.add_argument("--aggregation_methods", nargs='+', default=['martfl'],
                        help="List of aggregation methods to process (e.g., martfl fedavg)")
    parser.add_argument("--selection_analysis", action="store_true", help="Run additional selection pattern analysis")

    args = parser.parse_args()

    # Create output directories
    processed_data_dir = os.path.join(args.output_dir, "data")
    visualization_dir = os.path.join(args.output_dir, "visualizations")
    selection_analysis_dir = os.path.join(args.output_dir, "selection_analysis")
    aggregation_comparison_dir = os.path.join(visualization_dir, "aggregation_comparison")

    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # Define paths for processed CSV files
    summary_csv = os.path.join(processed_data_dir, "summary.csv")
    all_rounds_csv = os.path.join(processed_data_dir, "all_rounds.csv")

    # Step 1: Process the experimental logs
    if not args.skip_processing:
        print("Step 1: Processing experimental logs...")
        print(f"Aggregation methods: {args.aggregation_methods}")

        # Import the processing module
        from process_logs import process_all_experiments

        all_rounds_df, summary_df = process_all_experiments(
            output_dir=processed_data_dir,
            local_epoch=args.local_epoch,
            aggregation_methods=args.aggregation_methods
        )

        print(f"Processed data saved to {processed_data_dir}")
    else:
        print("Skipping data processing step.")
        if not (os.path.exists(summary_csv) and os.path.exists(all_rounds_csv)):
            print(f"Error: Required CSV files not found in {processed_data_dir}")
            print("Please run without --skip_processing flag first.")
            sys.exit(1)

    # Step 2: Generate visualizations
    if not args.skip_visualization:
        print("\nStep 2: Generating visualizations...")

        # Run the core visualizations
        run_visualization(summary_csv, all_rounds_csv, visualization_dir)
        print(f"Core visualizations saved to {visualization_dir}")

        # Add aggregation method comparison if we have multiple methods
        summary_df = pd.read_csv(summary_csv)
        if 'AGGREGATION_METHOD' in summary_df.columns and len(summary_df['AGGREGATION_METHOD'].unique()) > 1:
            print("\nGenerating aggregation method comparison visualizations...")

            plot_aggregation_comparison(summary_df, aggregation_comparison_dir)
            print(f"Aggregation comparison visualizations saved to {aggregation_comparison_dir}")

    else:
        print("Skipping visualization_226 step.")

    # Step 3: Run selection pattern analysis if requested
    if args.selection_analysis:
        print("\nStep 3: Running selection pattern analysis...")
        os.makedirs(selection_analysis_dir, exist_ok=True)

        # Load the processed data
        all_rounds_df = pd.read_csv(all_rounds_csv)

        # Convert boolean columns
        if 'IS_SYBIL' in all_rounds_df.columns:
            all_rounds_df['IS_SYBIL'] = all_rounds_df['IS_SYBIL'].astype(bool)

        # Process used_sellers and outlier_ids if they're stored as strings
        for col in ['used_sellers', 'outlier_ids']:
            if col in all_rounds_df.columns and all_rounds_df[col].dtype == 'object':
                all_rounds_df[col] = all_rounds_df[col].apply(
                    lambda x: eval(x) if isinstance(x, str) and x != '' else []
                )

        # Run the analysis
        analyze_selection_patterns(all_rounds_df, selection_analysis_dir)
        print(f"Selection pattern analysis saved to {selection_analysis_dir}")

    # Done
    print("\nAnalysis pipeline completed successfully!")

    # Print summary of results
    if os.path.exists(summary_csv):
        summary_df = pd.read_csv(summary_csv)

        print("\nOverall Results Summary:")
        print(f"Total experiments analyzed: {len(summary_df)}")
        print(f"Average Final ASR: {summary_df['FINAL_ASR'].mean():.4f}")
        print(f"Average Main Accuracy: {summary_df['FINAL_MAIN_ACC'].mean():.4f}")

        # If we have multiple aggregation methods, show comparison
        if 'AGGREGATION_METHOD' in summary_df.columns and len(summary_df['AGGREGATION_METHOD'].unique()) > 1:
            print("\nResults by Aggregation Method:")
            agg_summary = summary_df.groupby('AGGREGATION_METHOD')[['FINAL_ASR', 'FINAL_MAIN_ACC']].mean().round(4)
            print(agg_summary)

            # Results by gradient mode within each aggregation method
            print("\nResults by Aggregation Method and Gradient Mode:")
            grad_summary = summary_df.groupby(['AGGREGATION_METHOD', 'GRAD_MODE'])[
                ['FINAL_ASR', 'FINAL_MAIN_ACC']].mean().round(4)
            print(grad_summary)


if __name__ == "__main__":
    main()
