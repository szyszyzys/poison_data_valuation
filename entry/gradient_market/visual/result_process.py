import ast  # For safely evaluating string representations of lists/dicts
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # Optional, for nicer plots


# --- Helper for Safe String Evaluation ---

def safe_literal_eval(val):
    """Safely evaluates a string literal, handling common issues."""
    if pd.isna(val):
        return None
    try:
        # Handle potential numpy string representations slightly differently
        # This is heuristic; might need adjustment based on exact string format
        if isinstance(val, str) and val.startswith('array('):
            # Simplified eval for basic numpy arrays if needed, otherwise rely on ast
            # This part might be tricky and depends on how numpy arrays are saved.
            # Often, they are saved as lists anyway.
            pass  # Fall through to ast.literal_eval for list-like representation

        # Attempt standard literal evaluation
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        # If eval fails, return the original value or None/NaN
        # print(f"Warning: Could not evaluate literal: {val}")
        return None  # Or return np.nan, or val itself


# --- 1. Loading Function ---

def load_all_results(base_dir: str, csv_filename: str = "results.csv") -> Dict[str, List[pd.DataFrame]]:
    """
    Loads all experiment results from CSV files in a structured directory.

    Args:
        base_dir (str): The base directory containing experiment setup folders.
        csv_filename (str): The name of the CSV file within each run folder.

    Returns:
        Dict[str, List[pd.DataFrame]]: A dictionary where keys are experiment setup names
                                        and values are lists of DataFrames (one per run).
    """
    base_path = Path(base_dir)
    all_results = {}
    if not base_path.is_dir():
        print(f"Error: Base directory '{base_dir}' not found.")
        return all_results

    print(f"Loading results from: {base_path}")
    # Iterate through potential experiment setup directories
    for experiment_path in base_path.iterdir():
        if experiment_path.is_dir():
            experiment_name = experiment_path.name
            run_dfs = []
            print(f"  Loading Experiment: {experiment_name}")
            # Iterate through potential run directories
            for run_path in experiment_path.glob("run_*"):  # Use glob to find run folders
                if run_path.is_dir():
                    csv_file = run_path / csv_filename
                    if csv_file.is_file():
                        try:
                            df = pd.read_csv(csv_file)
                            if not df.empty:
                                run_id = run_path.name  # e.g., "run_0"
                                df['run_id'] = run_id  # Add run identifier
                                df['experiment_setup'] = experiment_name  # Add experiment identifier
                                run_dfs.append(df)
                                print(f"    Loaded {run_path.name}/{csv_filename} ({len(df)} rounds)")
                            else:
                                print(f"    Warning: Empty CSV file found: {csv_file}")
                        except Exception as e:
                            print(f"    Error loading {csv_file}: {e}")
                    else:
                        print(f"    Warning: CSV file '{csv_filename}' not found in {run_path}")

            if run_dfs:
                all_results[experiment_name] = run_dfs
            else:
                print(f"  Warning: No valid run data found for experiment '{experiment_name}'")

    if not all_results:
        print(f"Error: No experiment data loaded from {base_dir}. Check directory structure and file names.")

    return all_results


# --- 2. Preprocessing Function ---

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a single DataFrame of results.
    - Converts string representations of lists/dicts.
    - Extracts key performance metrics into separate columns.
    """
    # Columns that likely contain stringified lists or dicts
    eval_columns = [
        "selected_sellers", "outlier_sellers", "perf_global", "perf_local",
        "selection_rate_info", "defense_metrics"
    ]

    for col in eval_columns:
        if col in df.columns:
            # Apply safe literal evaluation
            df[col] = df[col].apply(safe_literal_eval)
            # Handle cases where evaluation resulted in None but should be empty list/dict
            if col in ["selected_sellers", "outlier_sellers"]:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
            elif col in ["perf_global", "perf_local", "selection_rate_info", "defense_metrics"]:
                df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else {})
        else:
            print(f"Warning: Expected column '{col}' not found in DataFrame.")

    # --- Extract nested performance metrics ---
    # Global Performance
    if 'perf_global' in df.columns:
        df['global_acc'] = df['perf_global'].apply(lambda x: x.get('accuracy') if isinstance(x, dict) else np.nan)
        df['global_loss'] = df['perf_global'].apply(lambda x: x.get('loss') if isinstance(x, dict) else np.nan)
        # Add others as needed (e.g., attack success rate)
        df['global_asr'] = df['perf_global'].apply(
            lambda x: x.get('attack_success_rate') if isinstance(x, dict) else np.nan)

    # Local Performance (Example: average accuracy across clients if stored)
    if 'perf_local' in df.columns:
        # This depends heavily on how perf_local is structured.
        # If it's a dict of {client_id: {acc: val, ...}}, you might average:
        def get_avg_local_acc(local_perf_dict):
            if not isinstance(local_perf_dict, dict) or not local_perf_dict:
                return np.nan
            accs = [stats.get('accuracy') for stats in local_perf_dict.values() if
                    isinstance(stats, dict) and 'accuracy' in stats]
            return np.mean(accs) if accs else np.nan

        # df['local_avg_acc'] = df['perf_local'].apply(get_avg_local_acc)
        pass  # Add extraction logic based on your perf_local structure

    # Selection Rate Info
    if 'selection_rate_info' in df.columns:
        df['selection_tpr'] = df['selection_rate_info'].apply(
            lambda x: x.get('true_positive_rate') if isinstance(x, dict) else np.nan)
        df['selection_fpr'] = df['selection_rate_info'].apply(
            lambda x: x.get('false_positive_rate') if isinstance(x, dict) else np.nan)
        df['selection_fnr'] = df['selection_rate_info'].apply(
            lambda x: x.get('false_negative_rate') if isinstance(x, dict) else np.nan)
        df['selection_accuracy'] = df['selection_rate_info'].apply(
            lambda x: x.get('accuracy') if isinstance(x, dict) else np.nan)

    # Defense Metrics (Example: extracting a specific metric)
    if 'defense_metrics' in df.columns:
        df['some_defense_metric'] = df['defense_metrics'].apply(
            lambda x: x.get('your_metric_name') if isinstance(x, dict) else np.nan)

    # Convert timestamp if needed (assuming it's parseable)
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            print(f"Warning: Could not parse 'timestamp' column: {e}")

    # Convert round number to numeric if it isn't already
    if 'round_number' in df.columns:
        df['round_number'] = pd.to_numeric(df['round_number'], errors='coerce')

    df = df.dropna(subset=['round_number'])  # Drop rows where round couldn't be parsed
    df['round_number'] = df['round_number'].astype(int)

    return df


# --- 3. Visualization Functions ---

def plot_metric_comparison(
        results_dict: Dict[str, List[pd.DataFrame]],
        metric_column: str,
        title: Optional[str] = None,
        xlabel: str = "Communication Round",
        ylabel: Optional[str] = None,
        use_seaborn: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        confidence_interval: Optional[Union[str, float]] = 'sd'
        # 'sd' for std dev, ('ci', 95) for 95% CI, None for no error band
):
    """
    Plots a metric over rounds, comparing different experiment setups with error bands.

    Args:
        results_dict: Dictionary of preprocessed results (output of load_all_results + preprocess_data).
        metric_column: The name of the column in the DataFrames to plot (e.g., 'global_acc').
        title: The title for the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis. If None, uses metric_column.
        use_seaborn: Whether to use seaborn for plotting (recommended for auto error bands).
        save_path: Path to save the figure. If None, shows the plot.
        figsize: Figure size.
        confidence_interval: Type of error band for seaborn ('sd', 'ci', None).
    """
    if not results_dict:
        print("Error: No results data provided for plotting.")
        return

    plt.figure(figsize=figsize)

    # Combine all runs for all experiments into a single DataFrame for seaborn
    all_runs_list = []
    for experiment_name, run_dfs in results_dict.items():
        if not run_dfs:
            print(f"Warning: No data for experiment '{experiment_name}', skipping.")
            continue
        # Check if metric column exists in the first df (assume others are same)
        if metric_column not in run_dfs[0].columns:
            print(
                f"Error: Metric '{metric_column}' not found in data for experiment '{experiment_name}'. Available: {run_dfs[0].columns.tolist()}")
            plt.close()  # Close the empty figure
            return
        # Concatenate DataFrames for the current experiment
        combined_exp_df = pd.concat(run_dfs, ignore_index=True)
        all_runs_list.append(combined_exp_df)

    if not all_runs_list:
        print("Error: No valid data found across all experiments for plotting.")
        plt.close()
        return

    all_runs_df = pd.concat(all_runs_list, ignore_index=True)

    # Drop rows where the metric is NaN, as they interfere with plotting/aggregation
    all_runs_df = all_runs_df.dropna(subset=[metric_column])

    if all_runs_df.empty:
        print(f"Error: No non-NaN data found for metric '{metric_column}' after preprocessing.")
        plt.close()
        return

    plot_ylabel = ylabel if ylabel is not None else metric_column.replace('_', ' ').title()
    plot_title = title if title is not None else f"{plot_ylabel} vs. {xlabel}"

    if use_seaborn:
        try:
            # Seaborn automatically calculates mean and error bands across runs
            sns.lineplot(
                data=all_runs_df,
                x="round_number",
                y=metric_column,
                hue="experiment_setup",  # Color lines by experiment
                errorbar=confidence_interval,  # Show std dev or CI bands
                linewidth=2
                # style="experiment_setup" # Optional: use different line styles too
            )
            plt.grid(True, linestyle='--', alpha=0.6)
        except Exception as e:
            print(f"Error during seaborn plotting: {e}. Check data for metric '{metric_column}'.")
            plt.close()
            return
    else:
        # Manual plotting with Matplotlib (more work for error bands)
        for experiment_name, run_dfs in results_dict.items():
            if not run_dfs: continue
            if metric_column not in run_dfs[0].columns: continue  # Skip if metric missing

            exp_df = pd.concat(run_dfs, ignore_index=True)
            exp_df = exp_df.dropna(subset=[metric_column])
            if exp_df.empty: continue

            # Calculate mean and std error across runs for each round
            grouped = exp_df.groupby("round_number")[metric_column]
            mean_metric = grouped.mean()
            std_err_metric = grouped.sem()  # Standard Error of the Mean

            plt.plot(mean_metric.index, mean_metric.values, label=experiment_name, linewidth=2)
            if confidence_interval is not None:  # Check if error band is desired
                plt.fill_between(
                    mean_metric.index,
                    mean_metric.values - std_err_metric.values,
                    mean_metric.values + std_err_metric.values,
                    alpha=0.2
                )
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.title(plot_title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(plot_ylabel, fontsize=12)
    plt.tight_layout()

    if save_path:
        try:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if needed
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
    else:
        plt.show()

    plt.close()  # Close the figure after saving/showing


def plot_final_round_comparison(
        results_dict: Dict[str, List[pd.DataFrame]],
        metric_column: str,
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 5)
):
    """
    Creates a bar chart comparing the mean metric value of the final round across experiments.

    Args:
        results_dict: Dictionary of preprocessed results.
        metric_column: The metric to compare (e.g., 'global_acc').
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Path to save the figure. If None, shows the plot.
        figsize: Figure size.
    """
    final_metrics = {}
    final_errors = {}

    for exp_name, run_dfs in results_dict.items():
        if not run_dfs: continue
        if metric_column not in run_dfs[0].columns: continue

        final_round_values = []
        for df in run_dfs:
            df = df.dropna(subset=[metric_column, 'round_number'])
            if not df.empty:
                # Find the value at the maximum round number for this run
                final_round_df = df[df['round_number'] == df['round_number'].max()]
                if not final_round_df.empty:
                    final_round_values.append(final_round_df[metric_column].iloc[0])  # Get the metric value

        if final_round_values:
            final_metrics[exp_name] = np.mean(final_round_values)
            final_errors[exp_name] = np.std(final_round_values) / np.sqrt(len(final_round_values))  # Standard error

    if not final_metrics:
        print(f"Error: No final round data found for metric '{metric_column}'.")
        return

    # Create Bar Chart
    exp_names = list(final_metrics.keys())
    mean_values = list(final_metrics.values())
    std_errors = list(final_errors.values())

    plt.figure(figsize=figsize)
    bars = plt.bar(exp_names, mean_values, yerr=std_errors, capsize=5,
                   color=sns.color_palette("viridis", len(exp_names)))

    plot_ylabel = ylabel if ylabel is not None else metric_column.replace('_', ' ').title()
    plot_title = title if title is not None else f"Final Round {plot_ylabel} Comparison"

    plt.title(plot_title, fontsize=15)
    plt.ylabel(plot_ylabel, fontsize=12)
    plt.xticks(rotation=15, ha='right')  # Rotate labels if long
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}', va='bottom' if yval >= 0 else 'top',
                 ha='center')  # va='bottom' places text above the bar

    if save_path:
        try:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
    else:
        plt.show()

    plt.close()


# --- Example Usage ---

if __name__ == "__main__":
    # 1. Define the base directory where experiment results are stored
    RESULTS_BASE_DIR = "./path_to_your_results"  # IMPORTANT: Change this path

    # 2. Load all results
    all_results_raw = load_all_results(RESULTS_BASE_DIR)

    # 3. Preprocess the loaded data
    all_results_processed = {}
    if all_results_raw:
        for exp_name, run_dfs_raw in all_results_raw.items():
            processed_runs = []
            print(f"Preprocessing experiment: {exp_name}")
            for i, df_raw in enumerate(run_dfs_raw):
                print(f"  Preprocessing run {i}...")
                try:
                    df_processed = preprocess_data(df_raw.copy())  # Process a copy
                    processed_runs.append(df_processed)
                except Exception as e:
                    print(f"  Error preprocessing run {i} for {exp_name}: {e}")
            if processed_runs:
                all_results_processed[exp_name] = processed_runs

    # 4. Generate Plots (only if data was loaded and processed)
    if all_results_processed:
        print("\n--- Generating Plots ---")
        output_plot_dir = Path("./analysis_plots")
        output_plot_dir.mkdir(exist_ok=True)

        # Plot Global Accuracy Comparison
        plot_metric_comparison(
            results_dict=all_results_processed,
            metric_column="global_acc",
            title="Global Model Accuracy vs. Communication Round",
            ylabel="Global Accuracy",
            use_seaborn=True,
            save_path=output_plot_dir / "global_accuracy_comparison.png"
        )

        # Plot Global Loss Comparison
        plot_metric_comparison(
            results_dict=all_results_processed,
            metric_column="global_loss",
            title="Global Model Loss vs. Communication Round",
            ylabel="Global Loss",
            use_seaborn=True,
            save_path=output_plot_dir / "global_loss_comparison.png"
        )

        # Plot Attack Success Rate (if available and extracted)
        plot_metric_comparison(
            results_dict=all_results_processed,
            metric_column="global_asr",  # Assumes you extracted this
            title="Attack Success Rate vs. Communication Round",
            ylabel="Attack Success Rate (ASR)",
            use_seaborn=True,
            save_path=output_plot_dir / "global_asr_comparison.png"
        )

        # Plot Selection False Positive Rate (if available and extracted)
        plot_metric_comparison(
            results_dict=all_results_processed,
            metric_column="selection_fpr",  # Assumes you extracted this
            title="Outlier Selection FPR vs. Communication Round",
            ylabel="False Positive Rate (FPR)",
            use_seaborn=True,
            save_path=output_plot_dir / "selection_fpr_comparison.png"
        )

        # Plot Number of Sellers Selected
        plot_metric_comparison(
            results_dict=all_results_processed,
            metric_column="num_sellers_selected",
            title="Number of Selected Sellers vs. Communication Round",
            ylabel="Number Selected",
            use_seaborn=True,
            save_path=output_plot_dir / "num_selected_comparison.png"
        )

        # Plot Final Round Global Accuracy Bar Chart
        plot_final_round_comparison(
            results_dict=all_results_processed,
            metric_column="global_acc",
            title="Final Round Global Accuracy",
            ylabel="Global Accuracy",
            save_path=output_plot_dir / "final_global_accuracy_bar.png"
        )

    else:
        print("\nNo processed data available to generate plots.")
