import pandas as pd
import json
from pathlib import Path
import warnings


def flatten_dict(d, parent_key='', sep='.'):
    """Flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def aggregate_experiment_results(base_results_dir: str) -> pd.DataFrame:
    """
    Crawls the results directory and aggregates all experiment
    configs and final metrics into a single DataFrame.
    """
    results_path = Path(base_results_dir)
    all_experiment_runs = []

    # Use rglob to find all 'final_metrics.json' files
    # This automatically finds all 'run_X_seed_Y' folders
    metric_files = list(results_path.rglob("final_metrics.json"))

    if not metric_files:
        print(f"Error: No 'final_metrics.json' files found in {base_results_dir}")
        return pd.DataFrame()

    print(f"Found {len(metric_files)} experiment runs. Aggregating...")

    for metric_file in metric_files:
        run_dir = metric_file.parent
        config_file = run_dir / "config_snapshot.json"

        # Ensure config file exists
        if not config_file.exists():
            warnings.warn(f"Skipping {run_dir}: 'config_snapshot.json' not found.")
            continue

        try:
            # Load config and flatten it
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Extract scenario name from the path
            # Assumes path is like: .../scenario_name_.../run_0_seed_42
            scenario_name = run_dir.parent.name
            flat_config = flatten_dict(config_data)
            flat_config['scenario_name'] = scenario_name
            flat_config['run_dir'] = str(run_dir)

            # Load final metrics
            with open(metric_file, 'r') as f:
                metrics_data = json.load(f)

            # Combine config and metrics
            flat_config.update(metrics_data)
            all_experiment_runs.append(flat_config)

        except Exception as e:
            warnings.warn(f"Skipping {run_dir}: Error processing files. {e}")

    df = pd.DataFrame(all_experiment_runs)

    # --- Post-processing (optional but recommended) ---

    # Average the results for the same parameters (across different seeds)
    # Get all columns that are NOT metrics
    param_columns = [col for col in df.columns if not col.startswith(('test_', 'val_', 'asr', 'B-Acc'))]
    # Keep 'run_dir' out of grouping
    param_columns = [c for c in param_columns if c not in ['run_dir', 'timestamp']]

    # Calculate mean and std dev across seeds
    df_mean = df.groupby(param_columns).mean(numeric_only=True).reset_index()
    df_std = df.groupby(param_columns).std(numeric_only=True).reset_index()

    # Rename std dev columns
    std_cols = {col: f"{col}_std" for col in df_std.columns if col not in param_columns}
    df_std = df_std.rename(columns=std_cols)

    # Merge mean and std dev dataframes
    df_agg = pd.merge(df_mean, df_std, on=param_columns)

    print(f"Aggregation complete. Master DataFrame has {len(df_agg)} unique parameter combinations.")
    return df_agg


# --- HOW TO RUN ---
# 1. Update this path to your RESULTS directory
#    (the one that contains 'main_summary_texas100_...', etc.)
RESULTS_DIRECTORY = "./configs_generated/tabular_fixed2"

master_df = aggregate_experiment_results(RESULTS_DIRECTORY)

if not master_df.empty:
    master_df.to_csv("master_results_summary.csv", index=False)
    print("\n✅ Master results summary saved to 'master_results_summary.csv'")
    print("\n--- First 5 Rows of Your Data ---")
    print(master_df.head())