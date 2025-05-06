import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List  # For type hints

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.io import read_image  # If you ever save as PNG/JPG directly
from torchvision.utils import make_grid

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Functions (similar to previous, slightly adapted) ---

def prep_for_grid(tensor_batch: torch.Tensor) -> torch.Tensor:
    """Prepares a batch of image tensors for make_grid by normalizing to [0,1]."""
    # Ensure it's a float tensor for min/max operations
    tensor_batch = tensor_batch.float()
    _min, _max = tensor_batch.min(), tensor_batch.max()
    # Normalize to [0, 1] if not already close
    if _min < -0.01 or _max > 1.01 or (_max - _min) < 1e-5:  # Check if scaling is needed
        tensor_batch = tensor_batch - _min
        current_max = tensor_batch.max()
        if current_max > 1e-5:  # Avoid division by zero
            tensor_batch = tensor_batch / current_max
    return torch.clamp(tensor_batch, 0, 1)


def load_image_tensor_from_path(full_path: Path) -> Optional[torch.Tensor]:
    """Loads an image tensor from a .pt file or common image file."""
    if not full_path.exists():
        logging.warning(f"Image file not found: {full_path}")
        return None
    try:
        if full_path.suffix == '.pt':
            tensor = torch.load(full_path, map_location=torch.device('cpu'))
            if not isinstance(tensor, torch.Tensor):
                logging.error(f"File {full_path} did not contain a PyTorch tensor.")
                return None
            # Ensure it's NCHW (add N if CHW)
            if tensor.ndim == 3: return tensor.unsqueeze(0)
            if tensor.ndim == 4: return tensor
            logging.error(f"Tensor from {full_path} has unsupported ndim: {tensor.ndim}")
            return None
        elif full_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            img_tensor = read_image(str(full_path))  # Reads as CHW, uint8
            if img_tensor.shape[0] == 4:  # RGBA
                img_tensor = img_tensor[:3, :, :]  # Keep RGB
            return (img_tensor / 255.0).unsqueeze(0)  # Normalize to [0,1] and add N dim
        else:
            logging.error(f"Unsupported image file type: {full_path.suffix} for path {full_path}")
            return None
    except Exception as e:
        logging.error(f"Failed to load image from {full_path}: {e}")
        return None


def display_gia_instance(
        log_entry: pd.Series,
        base_data_path: Path,  # Base path where 'reconstructed_image_file' etc. are relative to
        output_dir: Path,
        max_images_in_plot: int = 1  # GIA usually reconstructs few images
):
    """
    Generates and saves a visualization for a single GIA log entry.
    """
    victim_id = log_entry.get('victim_id', 'UnknownVictim')
    round_num = log_entry.get('round', log_entry.get('round_num', 'N/A'))

    rec_file = log_entry.get('reconstructed_image_file')
    gt_file = log_entry.get('ground_truth_image_file')  # Might be NaN/None

    if pd.isna(rec_file):
        logging.warning(
            f"No reconstructed image file path for victim {victim_id}, round {round_num}. Skipping visualization.")
        return

    rec_tensor = load_image_tensor_from_path(base_data_path / str(rec_file))
    if rec_tensor is None:
        logging.warning(f"Could not load reconstructed image for victim {victim_id}, round {round_num}. Skipping.")
        return

    gt_tensor = None
    if pd.notna(gt_file):
        gt_tensor = load_image_tensor_from_path(base_data_path / str(gt_file))

    num_display = min(rec_tensor.shape[0], max_images_in_plot)
    rec_display = rec_tensor[:num_display]

    rows = 1
    title = f"Victim: {victim_id}, Round: {round_num}\nReconstructed"
    if gt_tensor is not None and gt_tensor.shape[0] >= num_display:
        rows = 2
        gt_display = gt_tensor[:num_display]
        title = f"Victim: {victim_id}, Round: {round_num}"  # Main title
    elif gt_tensor is not None:  # Fewer GT images than rec
        rows = 2
        gt_display = gt_tensor  # Show all available GTs
        title = f"Victim: {victim_id}, Round: {round_num}"

    fig, axes = plt.subplots(rows, 1, figsize=(num_display * 2.5, rows * 3.0 + 1.0))  # +1 for metrics space
    if rows == 1:
        axes = [axes]  # Make it a list for consistent indexing

    ax_idx = 0
    if rows == 2 and gt_tensor is not None:
        gt_grid = make_grid(prep_for_grid(gt_display), nrow=gt_display.shape[0])
        axes[ax_idx].imshow(gt_grid.permute(1, 2, 0).numpy())
        axes[ax_idx].set_title("Ground Truth")
        axes[ax_idx].axis('off')
        gt_labels_str = log_entry.get('ground_truth_labels_str', 'N/A')
        axes[ax_idx].text(0.5, -0.05, f"GT Labels: {gt_labels_str}", size=8, ha="center",
                          transform=axes[ax_idx].transAxes)
        ax_idx += 1

    rec_grid = make_grid(prep_for_grid(rec_display), nrow=num_display)
    axes[ax_idx].imshow(rec_grid.permute(1, 2, 0).numpy())
    axes[ax_idx].set_title("Reconstructed")
    axes[ax_idx].axis('off')
    rec_labels_str = log_entry.get('reconstructed_labels_str', 'N/A')
    axes[ax_idx].text(0.5, -0.05, f"Rec Labels: {rec_labels_str}", size=8, ha="center",
                      transform=axes[ax_idx].transAxes)

    # Add metrics from log entry
    metrics_text_parts = []
    if pd.notna(log_entry.get('error')):  # Top-level GIA error
        metrics_text_parts.append(f"GIA Error: {log_entry.get('error')}")
    else:
        for col in log_entry.index:
            if col.startswith('metric_') and pd.notna(log_entry[col]):
                metric_name = col.replace('metric_', '').upper()
                try:
                    metrics_text_parts.append(f"{metric_name}: {float(log_entry[col]):.3f}")
                except ValueError:  # If metric is an error string itself
                    metrics_text_parts.append(f"{metric_name}: {log_entry[col]}")
        if pd.notna(log_entry.get('duration_sec')):
            metrics_text_parts.append(f"Duration: {log_entry.get('duration_sec'):.2f}s")

    fig.suptitle(title, fontsize=12, y=0.98 if rows == 1 else 0.99)
    if metrics_text_parts:
        fig.text(0.5, 0.01, "\n".join(metrics_text_parts), ha='center', va='bottom', fontsize=7,
                 bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05 if metrics_text_parts else 0, 1, 0.95])

    output_filename = f"GIA_viz_victim_{victim_id}_round_{round_num}.png"
    fig_save_path = output_dir / output_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_save_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved visualization: {fig_save_path}")


def generate_summary_tables(df: pd.DataFrame, group_by_cols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Generates summary tables of GIA metrics.
    Can group by specified columns if provided.
    """
    results = {}
    metric_cols = [col for col in df.columns if col.startswith('metric_')]
    if not metric_cols:
        logging.warning("No 'metric_*' columns found in the DataFrame for summary.")
        return results

    # Filter out rows where the main GIA attempt had an error
    successful_df = df[df['error'].isna()].copy()  # Use .copy() to avoid SettingWithCopyWarning
    if successful_df.empty:
        logging.warning("No successful GIA attempts (where 'error' is NaN) found for summary.")
        # Still try to summarize all data if no successful ones
        if df.empty: return results
        successful_df = df  # Fallback to all data

    # Convert metric columns to numeric, coercing errors to NaN
    for col in metric_cols:
        successful_df[col] = pd.to_numeric(successful_df[col], errors='coerce')

    if group_by_cols:
        valid_group_by_cols = [col for col in group_by_cols if col in successful_df.columns]
        if not valid_group_by_cols:
            logging.warning(f"None of the group_by columns {group_by_cols} found. Performing overall summary.")
            summary_overall = successful_df[metric_cols].agg(['mean', 'median', 'std', 'count'])
            results['overall_summary'] = summary_overall.T  # Transpose for better readability
        else:
            logging.info(f"Generating summary grouped by: {valid_group_by_cols}")
            grouped_summary = successful_df.groupby(valid_group_by_cols)[metric_cols].agg(
                ['mean', 'median', 'std', 'count'])
            results['grouped_summary'] = grouped_summary
    else:
        summary_overall = successful_df[metric_cols].agg(['mean', 'median', 'std', 'count'])
        results['overall_summary'] = summary_overall.T  # Transpose for better readability

    # Add summary of attack duration
    if 'duration_sec' in successful_df.columns:
        duration_summary = successful_df['duration_sec'].agg(['mean', 'median', 'std', 'min', 'max'])
        results['duration_summary'] = pd.DataFrame(duration_summary)

    return results


def main_analysis_pipeline(
        log_csv_path: Path,
        base_data_path: Path,  # Base path where image tensor files are stored
        output_dir: Path,
        num_best_viz: int = 5,
        num_worst_viz: int = 5,
        num_random_viz: int = 5,
        group_by: Optional[List[str]] = None
        # Columns to group summary tables by (e.g., ['attack_setting_A', 'victim_type'])
):
    """
    Main pipeline to analyze GIA logs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_output_dir = output_dir / "visualizations"
    viz_output_dir.mkdir(parents=True, exist_ok=True)
    tables_output_dir = output_dir / "summary_tables"
    tables_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(log_csv_path)
    except FileNotFoundError:
        logging.error(f"Log file not found: {log_csv_path}")
        return
    except Exception as e:
        logging.error(f"Error reading log CSV {log_csv_path}: {e}")
        return

    logging.info(f"Loaded {len(df)} entries from {log_csv_path}.")

    # --- Generate Summary Tables ---
    summary_tables = generate_summary_tables(df.copy(), group_by_cols=group_by)  # Pass a copy
    for name, table_df in summary_tables.items():
        if not table_df.empty:
            table_path = tables_output_dir / f"{name}.csv"
            table_df.to_csv(table_path)
            logging.info(f"Saved summary table: {table_path}")
            print(f"\n--- {name.replace('_', ' ').title()} ---")
            print(table_df.to_string())

    # --- Select and Generate Visualizations ---
    # Consider only successful attacks for best/worst visualization
    successful_df = df[df['error'].isna()].copy()
    if successful_df.empty:
        logging.warning("No successful attacks found for detailed visualization.")
        return

    # Convert metric columns to numeric for sorting, coercing errors
    metric_cols_for_sort = [col for col in successful_df.columns if col.startswith('metric_')]
    for col in metric_cols_for_sort:
        successful_df[col] = pd.to_numeric(successful_df[col], errors='coerce')
    successful_df = successful_df.dropna(subset=metric_cols_for_sort, how='all')  # Drop rows where all metrics are NaN

    # Visualize Best PSNR
    if 'metric_psnr' in successful_df.columns and num_best_viz > 0:
        logging.info(f"\n--- Visualizing Best {num_best_viz} PSNR cases ---")
        best_psnr_df = successful_df.sort_values(by='metric_psnr', ascending=False).head(num_best_viz)
        for _, row in best_psnr_df.iterrows():
            display_gia_instance(row, base_data_path, viz_output_dir)

    # Visualize Worst PSNR (Lowest non-NaN)
    if 'metric_psnr' in successful_df.columns and num_worst_viz > 0:
        logging.info(f"\n--- Visualizing Worst {num_worst_viz} PSNR cases ---")
        # Ensure we only take rows where PSNR is a valid number for sorting
        worst_psnr_df = successful_df[successful_df['metric_psnr'].notna()].sort_values(by='metric_psnr',
                                                                                        ascending=True).head(
            num_worst_viz)
        for _, row in worst_psnr_df.iterrows():
            display_gia_instance(row, base_data_path, viz_output_dir)

    # Visualize Best LPIPS (Lowest LPIPS is better)
    if 'metric_lpips' in successful_df.columns and num_best_viz > 0:
        logging.info(f"\n--- Visualizing Best {num_best_viz} LPIPS cases (Low LPIPS is good) ---")
        best_lpips_df = successful_df[successful_df['metric_lpips'].notna()].sort_values(by='metric_lpips',
                                                                                         ascending=True).head(
            num_best_viz)
        for _, row in best_lpips_df.iterrows():
            display_gia_instance(row, base_data_path, viz_output_dir)

    # Visualize Random Samples
    if num_random_viz > 0 and not successful_df.empty:
        logging.info(f"\n--- Visualizing {num_random_viz} Random successful cases ---")
        random_samples_df = successful_df.sample(n=min(num_random_viz, len(successful_df)))
        for _, row in random_samples_df.iterrows():
            display_gia_instance(row, base_data_path, viz_output_dir)

    logging.info("Post-hoc analysis and visualization generation finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post-hoc analysis of Gradient Inversion Attack results.")
    parser.add_argument("log_csv_path", type=Path, help="Path to the GIA results CSV file.")
    parser.add_argument("base_data_path", type=Path,
                        help="Base directory where reconstructed/GT image tensor files (.pt) are stored, "
                             "relative to paths in the CSV.")
    parser.add_argument("--output_dir", type=Path, default=Path("./gia_analysis_results"),
                        help="Directory to save generated tables and visualizations.")
    parser.add_argument("--best_n", type=int, default=3, help="Number of best cases to visualize per metric.")
    parser.add_argument("--worst_n", type=int, default=3, help="Number of worst cases to visualize per metric.")
    parser.add_argument("--random_n", type=int, default=3, help="Number of random cases to visualize.")
    parser.add_argument("--group_by", nargs='*',
                        help="List of column names to group summary tables by (e.g., model_type dataset_name).")

    args = parser.parse_args()

    main_analysis_pipeline(
        log_csv_path=args.log_csv_path,
        base_data_path=args.base_data_path,
        output_dir=args.output_dir,
        num_best_viz=args.best_n,
        num_worst_viz=args.worst_n,
        num_random_viz=args.random_n,
        group_by=args.group_by
    )
