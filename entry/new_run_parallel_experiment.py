import argparse
import copy
import logging
import multiprocessing
import os
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


def run_single_experiment(config_path: str, run_id: int, sample_idx: int, seed: int,
                          gpu_id: int = None, force_rerun: bool = False):
    """
    Function to run a SINGLE experiment instance (one config, one seed)
    with proper locking and caching.
    """
    try:
        # ===== CRITICAL: Set GPU and seed IMMEDIATELY at the very beginning =====
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            log_prefix = f"[Run {run_id} | Sample {sample_idx} | Seed {seed} | GPU {gpu_id}]"
        else:
            log_prefix = f"[Run {run_id} | Sample {sample_idx} | Seed {seed} | CPU]"

        # Seed everything BEFORE any PyTorch operations
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            # CRITICAL: Seed CUDA before any CUDA operations
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

            # Make operations deterministic (prevents NaN from non-deterministic ops)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # NOW clear cache after seeding
            torch.cuda.empty_cache()

        logger.info(f"{log_prefix} Starting. Config: {config_path}")

        app_config = load_config(config_path)
        original_base_save_path = Path(app_config.experiment.save_path)
        run_save_path = original_base_save_path / f"run_{sample_idx - 1}_seed_{seed}"

        # Create directory early
        run_save_path.mkdir(parents=True, exist_ok=True)

        # Use a lock file to prevent race conditions
        lock_file = run_save_path / ".lock"
        lock = FileLock(str(lock_file), timeout=10)

        with lock.acquire(timeout=2):
            if is_run_completed(run_save_path) and not force_rerun:
                logger.info(f"{log_prefix} ✅ Already completed. Skipping.")
                return

            if is_run_completed(run_save_path) and force_rerun:
                logger.info(f"{log_prefix} 🔄 Force rerun enabled.")

            mark_run_in_progress(run_save_path)

            # Prepare config for this specific run
            run_cfg = copy.deepcopy(app_config)
            # Don't call set_seed again here - we already did it above
            run_cfg.seed = seed
            run_cfg.experiment.save_path = str(run_save_path)
            run_cfg.experiment.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Run the actual experiment
            run_attack(run_cfg)
            mark_run_completed(run_save_path)

            logger.info(f"{log_prefix} ✅ Completed successfully.")

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except TimeoutError:
        logger.warning(f"{log_prefix} ⏳ Another process is running this. Skipping.")
    except Exception as e:
        logger.error(f"{log_prefix} ❌ Error running experiment: {e}", exc_info=True)
        # Optionally create a .failed marker
        if 'run_save_path' in locals():
            (run_save_path / ".failed").touch()
        raise


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
                  force_rerun: bool = False):
    """
    Main function to orchestrate parallel execution of experiments.
    """
    all_config_files = discover_configs(configs_base_dir)
    if not all_config_files:
        logger.warning(f"❌ No config.yaml files found in {configs_base_dir}. Exiting.")
        return

    logger.info(f"📋 Found {len(all_config_files)} configuration files")
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

    args = parser.parse_args()

    main_parallel(
        configs_base_dir=args.configs_dir,
        num_processes=args.num_processes,
        gpu_ids_str=args.gpu_ids,
        force_rerun=args.force_rerun
    )
