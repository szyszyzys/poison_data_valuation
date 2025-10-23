# FILE: analyze_tuning_results.py

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# --- Configuration ---
# Adjust this path if your aggregated results file is elsewhere
AGGREGATED_RESULTS_CSV = Path("./master_results_summary.csv")
# The primary metric to maximize (usually final test accuracy)
PRIMARY_METRIC = 'test_acc'
# Columns identifying each unique model/dataset pair
GROUPING_COLUMNS = [
    'experiment.dataset_name',
    'experiment.model_structure',
    # Add other relevant keys if needed, e.g., 'experiment.dataset_type'
    # 'experiment.tabular_model_config_name', # Usually redundant if grouped by dataset/model
    # 'experiment.image_model_config_name',
    # 'experiment.text_model_config_name',
]
# The hyperparameters we tuned
TUNED_PARAMS = ['training.learning_rate', 'training.local_epochs']

# --- Main Analysis Function ---
def find_best_tuning_params(df: pd.DataFrame):
    """
    Analyzes the aggregated tuning results to find the best hyperparameters
    for each model/dataset combination based on the primary metric.
    """

    # Check if necessary columns exist
    required_cols = GROUPING_COLUMNS + TUNED_PARAMS + [PRIMARY_METRIC]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: The results CSV is missing required columns: {missing_cols}")
        print("   Please ensure your aggregation script includes these columns.")
        return None

    print(f"--- Analyzing Tuning Results ---")
    print(f"Optimizing for: {PRIMARY_METRIC} (higher is better)")

    best_params_dict = {}

    # Group by each unique model/dataset configuration
    # Dropna() handles cases where grouping columns might have NaNs if aggregation had issues
    grouped = df.groupby(GROUPING_COLUMNS, dropna=False)

    for name, group in grouped:
        group_id = name # name is a tuple if multiple grouping columns
        if isinstance(group_id, tuple):
             group_id_str = "_".join(map(str, group_id)) # Create a readable ID string
        else:
             group_id_str = str(group_id)

        print(f"\nAnalyzing Group: {group_id_str}")
        print(f"  Found {len(group)} hyperparameter combinations.")

        # Find the row with the maximum primary metric value
        best_row = group.loc[group[PRIMARY_METRIC].idxmax()]

        best_lr = best_row['training.learning_rate']
        best_epochs = best_row['training.local_epochs']
        best_metric_value = best_row[PRIMARY_METRIC]

        # Get standard deviation if available
        std_col = f"{PRIMARY_METRIC}_std"
        best_metric_std = best_row.get(std_col, np.nan) # Use .get for safety

        print(f"  🏆 Best Result:")
        print(f"     - Learning Rate (lr): {best_lr}")
        print(f"     - Local Epochs (E):   {best_epochs}")
        if not np.isnan(best_metric_std):
            print(f"     - {PRIMARY_METRIC}:     {best_metric_value:.4f} ± {best_metric_std:.4f}")
        else:
            print(f"     - {PRIMARY_METRIC}:     {best_metric_value:.4f} (std dev not found)")

        # --- Stability Check Warning ---
        # Look for potential instability indicators (optional, basic check)
        high_std_threshold = 0.05 # Example: Flag if std dev > 5% accuracy
        if not np.isnan(best_metric_std) and best_metric_std > high_std_threshold:
             warnings.warn(
                f"    ⚠️ High standard deviation ({best_metric_std:.4f}) across seeds "
                f"for the best config ({group_id_str}, lr={best_lr}, E={best_epochs}). "
                f"Consider checking training_log.csv for stability issues."
             )

        # Check if the best result is on the boundary of the search space
        if best_lr == group['training.learning_rate'].min() or \
           best_lr == group['training.learning_rate'].max() or \
           best_epochs == group['training.local_epochs'].min() or \
           best_epochs == group['training.local_epochs'].max():
            warnings.warn(
                f"    ⚠️ Best parameters (lr={best_lr}, E={best_epochs}) are on the "
                f"boundary of the search space for group '{group_id_str}'. "
                f"Consider expanding the search range if results aren't satisfactory."
            )

        # Store the best parameters
        best_params_dict[group_id_str] = {
            'learning_rate': best_lr,
            'local_epochs': int(best_epochs), # Ensure epochs is an int
            'best_metric_value': best_metric_value,
            'best_metric_std': best_metric_std
        }

    return best_params_dict

# --- Main Execution ---
if __name__ == "__main__":
    if not AGGREGATED_RESULTS_CSV.exists():
        print(f"❌ Error: Aggregated results file not found at '{AGGREGATED_RESULTS_CSV}'")
        print("   Please run your aggregation script first.")
    else:
        try:
            # Load the aggregated data
            results_df = pd.read_csv(AGGREGATED_RESULTS_CSV)

            # Find the best parameters
            golden_parameters = find_best_tuning_params(results_df)

            if golden_parameters:
                print("\n\n--- 🌟 Recommended Golden Training Parameters ---")
                for group_id, params in golden_parameters.items():
                    print(f"\n➡️ For Model/Dataset: {group_id}")
                    print(f"   - training.learning_rate: {params['learning_rate']}")
                    print(f"   - training.local_epochs:   {params['local_epochs']}")
                    print(f"   (Achieved {PRIMARY_METRIC}: {params['best_metric_value']:.4f})")

                print("\n--- IMPORTANT NEXT STEPS ---")
                print("1. 🧐 Manually review the `training_log.csv` files for the selected")
                print("   configurations to confirm training stability (smooth convergence).")
                print("   Prioritize stability over minor gains in the final metric.")
                print("2. ✏️ Hard-code these 'Golden Parameters' into your base config ")
                print("   factory functions (e.g., get_base_tabular_config) before")
                print("   running attack/defense experiments.")

        except Exception as e:
            print(f"\n❌ An error occurred during analysis: {e}")
            import traceback
            traceback.print_exc()