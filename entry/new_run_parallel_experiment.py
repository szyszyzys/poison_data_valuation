import argparse
import copy
import logging
import multiprocessing
import os
import shutil
from multiprocessing.pool import Pool
from pathlib import Path

import torch
from filelock import FileLock  # pip install filelock

from entry.gradient_market.automate_exp.config_parser import load_config
from entry.gradient_market.run_all_exp import run_attack


# ==============================================================================
# == FIX for Nested Parallelism ==
# ==============================================================================
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


class NestablePool(Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CORE_EXPERIMENTS = [
    # --- 1. Main Summary Results ---
    "main_summary_cifar100_cnn",
    "main_summary_cifar100_resnet18",
    "main_summary_cifar10_cnn",
    "main_summary_cifar10_resnet18",
    "main_summary_trec",

    # --- 2. Sybil Attack Analysis ---
    "sybil_baseline_cifar10_cnn",
    # "sybil_baseline_cifar10_resnet18",
    "sybil_knock_out_cifar10_cnn",
    # "sybil_knock_out_cifar10_resnet18",
    "sybil_mimic_cifar10_cnn",
    # "sybil_mimic_cifar10_resnet18",
    "sybil_pivot_cifar10_cnn",
    # "sybil_pivot_cifar10_resnet18",

    # --- 3. Ablation Studies & Specific Analyses ---
    "oracle_vs_buyer_bias_cifar10_cnn",
    "buyer_data_impact_cifar10_cnn",
    "heterogeneity_impact_cifar10_cnn",
    # "heterogeneity_impact_cifar10_resnet18",
    "selection_rate_baseline_cifar10_cnn",
    "selection_rate_cluster_cifar10_cnn",
    "trend_adv_rate_martfl_cifar10_cnn",

    # --- 4. Alternative Attack Scenarios ---
    "label_flip_cifar10_cnn",
    # "label_flip_cifar10_resnet18",
    "label_flip_trec",
    "adaptive_evasion_data_poisoning_cifar10_cnn",
    "adaptive_evasion_gradient_manipulation_cifar10_cnn",
    "drowning_attack_cifar10_cnn",

    # --- 5. Scalability Experiments ---
    "scalability_backdoor_sybil_cifar100_cnn",
    "scalability_backdoor_sybil_cifar10_cnn",
    # "scalability_backdoor_sybil_cifar10_resnet18",
    "scalability_backdoor_trec",
    "scalability_baseline_no_attack_cifar10_cnn",
    "scalability_buyer_class_exclusion_cifar10_cnn",
    "scalability_buyer_oscillating_cifar10_cnn",
    "scalability_combined_backdoor_buyer_cifar10_cnn",
    "extreme_scale_backdoor_martfl",
    # "extreme_scale_buyer_class_exclusion_fltrust",
]


def is_run_completed(run_save_path: Path) -> bool:
    """
    Check if a run is already completed by verifying multiple success indicators.
    More robust than checking a single file.
    """
    success_marker = run_save_path / ".success"
    final_metrics = run_save_path / "final_metrics.json"

    # Both files should exist for a truly successful run
    return success_marker.exists() and final_metrics.exists()


def mark_run_in_progress(run_save_path: Path):
    """Create a marker file to indicate run is in progress."""
    in_progress_marker = run_save_path / ".in_progress"
    in_progress_marker.touch()


def mark_run_completed(run_save_path: Path):
    """Mark a run as successfully completed."""
    in_progress_marker = run_save_path / ".in_progress"
    success_marker = run_save_path / ".success"

    # Remove in-progress marker and create success marker
    if in_progress_marker.exists():
        in_progress_marker.unlink()
    success_marker.touch()


def clear_existing_results(run_save_path: Path):
    """Deletes common result files and directories within a run path."""
    logger.info(f"Clearing existing results in: {run_save_path}")
    files_to_delete = [
        ".success",
        ".failed",
        ".in_progress",  # Should ideally be handled by lock, but good to clean
        "training_log.csv",
        "round_aggregates.csv",
        "seller_round_metrics_flat.csv",
        "selection_history.csv",
        "final_metrics.json",
        "final_model.pth",
        "marketplace_report.json",
        "config_snapshot.json",  # Snapshot is usually regenerated anyway
    ]
    dirs_to_delete = [
        "evaluations",
        "individual_gradients",  # If you save these
        # Add any seller-specific subdirs if they contain persistent state
        # e.g., "adv_seller_0/history", "bn_seller_1/history"
    ]

    # Delete files
    for filename in files_to_delete:
        filepath = run_save_path / filename
        if filepath.exists():
            try:
                filepath.unlink()
                logger.debug(f"  Deleted file: {filename}")
            except OSError as e:
                logger.warning(f"  Could not delete file {filepath}: {e}")

    # Delete directories
    for dirname in dirs_to_delete:
        dirpath = run_save_path / dirname
        if dirpath.exists() and dirpath.is_dir():
            try:
                shutil.rmtree(dirpath)
                logger.debug(f"  Deleted directory: {dirname}")
            except OSError as e:
                logger.warning(f"  Could not delete directory {dirpath}: {e}")


def run_single_experiment(config_path: str, run_id: int, sample_idx: int, seed: int,
                          gpu_id: int = None, force_rerun: bool = False):
    """
    Function to run a SINGLE experiment instance.
    Reverted to setting CUDA_VISIBLE_DEVICES internally and using 'cuda:0'.
    Assumes the parent process correctly maps physical IDs to the gpu_id argument (0, 1, 2...).
    """
    run_save_path = None  # Define early for error handling
    try:
        # ===== GPU/Seed Setup (Reverted Logic) =====
        target_device = "cpu"  # Default
        log_prefix_gpu_info = "CPU"

        if gpu_id is not None:
            # Set CUDA_VISIBLE_DEVICES to make ONLY this GPU visible, mapped to ID 0
            # This gpu_id is the logical ID (0, 1, 2...) relative to the externally set CUDA_VISIBLE_DEVICES
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            log_prefix_gpu_info = f"Logical GPU {gpu_id}"
            # PyTorch will now see this GPU as 'cuda:0'
            target_device = "cuda:0"
            # Quick check if CUDA is available *after* setting the environment variable
            if not torch.cuda.is_available():
                logger.error(
                    f"[Run {run_id}...] CUDA not available after setting CUDA_VISIBLE_DEVICES='{gpu_id}'. Check parent visibility settings.")
                # Fallback to CPU or raise error? For now, fallback might hide issues.
                target_device = "cpu"
                log_prefix_gpu_info = f"Logical GPU {gpu_id} (CUDA FAILED, using CPU)"
                # raise RuntimeError(f"CUDA failed for gpu_id {gpu_id}") # Alternative: Fail fast
        else:
            log_prefix_gpu_info = "CPU"

        log_prefix = f"[Run {run_id} | Sample {sample_idx} | Seed {seed} | {log_prefix_gpu_info}]"

        # --- Seed everything ---
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if target_device.startswith("cuda"):  # Check if we intend to use CUDA
            torch.cuda.manual_seed_all(seed)  # Seed the visible device(s)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logger.info(f"{log_prefix} Starting. Config: {config_path}")
        logger.info(f"{log_prefix} Target device internally seen as: {target_device}")

        # --- Config loading and path setup ---
        app_config = load_config(config_path)
        original_base_save_path = Path(app_config.experiment.save_path)
        run_save_path = original_base_save_path / f"run_{sample_idx - 1}_seed_{seed}"
        run_save_path.mkdir(parents=True, exist_ok=True)

        # --- Locking and Cleanup Logic ---
        lock_file = run_save_path / ".lock"
        lock = FileLock(str(lock_file), timeout=10)
        with lock.acquire(timeout=5):
            run_completed = is_run_completed(run_save_path)
            if run_completed and not force_rerun:
                logger.info(f"{log_prefix} ✅ Already completed. Skipping.")
                return
            if force_rerun:
                logger.info(f"{log_prefix} 🔄 Force rerun enabled. Clearing previous results...")
                clear_existing_results(run_save_path)

            mark_run_in_progress(run_save_path)

            # --- Prepare config (Use the remapped target_device "cuda:0" or "cpu") ---
            run_cfg = copy.deepcopy(app_config)
            run_cfg.seed = seed
            run_cfg.experiment.save_path = str(run_save_path)
            run_cfg.experiment.device = target_device  # <<< Pass "cuda:0" or "cpu"

            # --- Run the experiment ---
            run_attack(run_cfg)
            mark_run_completed(run_save_path)
            logger.info(f"{log_prefix} ✅ Completed successfully.")

            # Clean up GPU memory
            if target_device.startswith("cuda"):
                try:
                    del run_cfg
                    torch.cuda.empty_cache()
                    logger.info(f"{log_prefix} Cleared CUDA cache.")
                except Exception as e_clear:
                    logger.warning(f"{log_prefix} Error clearing CUDA cache: {e_clear}")

    except TimeoutError:
        logger.warning(f"{log_prefix} ⏳ Lock acquisition timed out. Skipping.")
    except Exception as e:
        logger.error(f"{log_prefix} ❌ Error running experiment: {e}", exc_info=True)
        if run_save_path:
            (run_save_path / ".failed").touch()


def discover_configs(configs_base_dir: str) -> list:
    """Discover all config.yaml files in the directory tree."""
    all_config_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(configs_base_dir)
        for file in files if file == "config.yaml"
    ]
    return sorted(all_config_files)  # Sort for reproducibility


def setup_gpu_allocation(num_processes: int, gpu_ids_str: str = None):
    """
    Determine GPU allocation strategy.

    Returns:
        tuple: (actual_num_processes, assigned_gpu_ids)
    """
    assigned_gpu_ids = None

    if gpu_ids_str:
        assigned_gpu_ids = [int(g.strip()) for g in gpu_ids_str.split(',')]
        actual_num_processes = len(assigned_gpu_ids)
        logger.info(f"🎯 Using specified GPUs: {assigned_gpu_ids}")
        logger.info(f"🔧 Setting num_processes to {actual_num_processes} to match GPU count")

    elif torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        actual_num_processes = min(num_processes, num_cuda_devices)
        assigned_gpu_ids = list(range(actual_num_processes))

        if num_processes > num_cuda_devices:
            logger.warning(f"⚠️  Requested {num_processes} processes but only {num_cuda_devices} GPUs available. "
                           f"Limiting to {actual_num_processes}.")
        logger.info(f"🎮 Auto-detected GPUs: {assigned_gpu_ids}")

    else:
        actual_num_processes = num_processes
        logger.info(f"💻 No GPUs available. Running {actual_num_processes} processes on CPU.")

    return actual_num_processes, assigned_gpu_ids


def main_parallel(configs_base_dir: str, num_processes: int, gpu_ids_str: str = None,
                  force_rerun: bool = False, config_filter: str = None, core_only: bool = False):
    """
    Main function to orchestrate parallel execution of experiments.
    """
    all_config_files = discover_configs(configs_base_dir)
    if not all_config_files:
        logger.warning(f"❌ No config.yaml files found in {configs_base_dir}. Exiting.")
        return

    logger.info(f"📋 Found {len(all_config_files)} total configuration files")

    # --- START: NEW FILTERING LOGIC ---
    if core_only:
        if config_filter:
            logger.warning("⚠️ Both --core_only and --filter provided. --core_only takes precedence.")

        logger.info(f"Filtering based on {len(CORE_EXPERIMENTS)} core experiments list.")
        filtered_config_files = []
        core_exp_set = set(CORE_EXPERIMENTS)  # Use a set for fast O(1) lookups

        for path in all_config_files:
            # Assumes config.yaml is in a folder named after the experiment
            # e.g., .../configs_generated/sybil_mimic_cifar10_cnn/config.yaml
            exp_name = Path(path).parent.name
            if exp_name in core_exp_set:
                filtered_config_files.append(path)

        if not filtered_config_files:
            logger.warning(f"❌ No config files matched the CORE_EXPERIMENTS list.")
            logger.warning("Please check your --configs_dir path.")
            return

        logger.info(
            f"📋 Found {len(filtered_config_files)} matching configurations (out of {len(all_config_files)} total)")
        all_config_files = filtered_config_files

    elif config_filter:
        logger.info(f"🔍 Applying path filter: '{config_filter}'")
        filtered_config_files = [
            path for path in all_config_files
            if config_filter in path
        ]

        if not filtered_config_files:
            logger.warning(f"❌ No config files matched the filter '{config_filter}'. Exiting.")
            return

        logger.info(
            f"📋 Found {len(filtered_config_files)} matching configurations (out of {len(all_config_files)} total)")
        all_config_files = filtered_config_files  # Overwrite the list
    else:
        logger.info(f"📋 Running all {len(all_config_files)} configurations (no filter applied)")

    actual_num_processes, assigned_gpu_ids = setup_gpu_allocation(num_processes, gpu_ids_str)

    # === Build the full list of individual run tasks ===
    tasks = []
    run_counter = 0
    for config_path in all_config_files:
        temp_cfg = load_config(config_path)
        initial_seed = temp_cfg.seed
        n_samples = temp_cfg.n_samples if hasattr(temp_cfg, 'n_samples') else 10

        for i in range(n_samples):
            run_counter += 1
            current_seed = initial_seed + i
            # FIX: Use None for CPU, not run_id
            if assigned_gpu_ids:
                current_gpu_id = assigned_gpu_ids[len(tasks) % len(assigned_gpu_ids)]
            else:
                current_gpu_id = None  # Changed from run_id
            tasks.append((config_path, run_counter, i + 1, current_seed, current_gpu_id, force_rerun))

    logger.info(f"{'=' * 60}")
    logger.info(f"🚀 Starting parallel execution:")
    logger.info(f"   - Total Individual Runs: {len(tasks)}")
    logger.info(f"   - Parallel Processes: {actual_num_processes}")
    logger.info(f"   - GPUs: {assigned_gpu_ids if assigned_gpu_ids else 'CPU only'}")
    logger.info(f"   - Force Rerun: {force_rerun}")
    logger.info(f"{'=' * 60}")

    with NestablePool(processes=actual_num_processes) as pool:
        pool.starmap(run_single_experiment, tasks)

    logger.info("🎉 All parallel experiments completed!")


if __name__ == "__main__":
    # Set start method for CUDA compatibility
    if multiprocessing.get_start_method(allow_none=True) is None:
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    parser = argparse.ArgumentParser(
        description="Run multiple FL experiments in parallel with caching support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--configs_dir",
        type=str,
        default="configs_generated",
        help="Base directory containing generated config files"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel processes (will be adjusted to match GPU count if using GPUs)"
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        help="Comma-separated CUDA device IDs, e.g., '0,1,2,3'"
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Ignore cached results and rerun all experiments"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only run configs whose path contains this string (e.g., 'sybil_mimic' or 'main_summary')"
    )
    parser.add_argument(
        "--core_only",
        action="store_true",
        help="Only run experiments from the built-in CORE_EXPERIMENTS list"
    )
    parser.add_argument(
        "--clear_results_on_rerun",  # New Flag Name
        action="store_true",
        help="[DEPRECATED by --force_rerun logic] Delete existing result files within a run directory before rerunning."
    )
    # --- END OF ADDITION ---
    args = parser.parse_args()

    main_parallel(
        configs_base_dir=args.configs_dir,
        num_processes=args.num_processes,
        gpu_ids_str=args.gpu_ids,
        force_rerun=args.force_rerun,
        config_filter=args.filter,
        core_only=args.core_only
    )
