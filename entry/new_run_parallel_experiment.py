import argparse
import copy
import logging
import multiprocessing
import os
from multiprocessing.pool import Pool
from pathlib import Path

import torch

from common.utils import set_seed
from entry.gradient_market.automate_exp.config_parser import load_config
# Make sure this import points to your main experiment function
from entry.gradient_market.run_all_exp import run_attack


# ==============================================================================
# == FIX for Nested Parallelism ==
# These classes create a custom Pool that uses non-daemonic processes,
# allowing them to have their own child processes (like DataLoader workers).
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

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_single_experiment(config_path: str, run_id: int, gpu_id: int = None):
    """
    Function to run a single experiment. It now skips runs that are already completed.
    """
    try:
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"[Run {run_id}] Assigned to GPU {gpu_id}. Starting experiment with config: {config_path}")
        else:
            logger.info(f"[Run {run_id}] Starting experiment with config: {config_path}")

        app_config = load_config(config_path)
        initial_seed = app_config.seed

        for i in range(app_config.n_samples):
            current_seed = initial_seed + i
            original_base_save_path = Path(app_config.experiment.save_path)
            run_save_path = original_base_save_path / f"run_{i}_seed_{current_seed}"

            # --- CHECK FOR COMPLETION ---
            # Define a "success marker" file that indicates a run is finished.
            # Change "final_metrics.json" if your run_attack function creates a different file at the end.
            success_marker_path = run_save_path / "final_metrics.json"

            if success_marker_path.exists():
                logger.info(f"[Run {run_id} - Sub-run {i + 1}] Already completed. Skipping: {run_save_path}")
                continue  # Skip to the next sub-run

            # --- If not completed, proceed with the run ---
            logger.info(
                f"[Run {run_id} - Sub-run {i + 1}] Starting... Config: {config_path}, Seed: {current_seed}, Save Path: {run_save_path}")

            run_cfg = copy.deepcopy(app_config)
            set_seed(current_seed)

            run_save_path.mkdir(parents=True, exist_ok=True)
            run_cfg.experiment.save_path = str(run_save_path)

            if gpu_id is not None:
                run_cfg.experiment.device = "cuda:0"
            else:
                run_cfg.experiment.device = "cpu"

            run_attack(run_cfg)

        logger.info(f"[Run {run_id}] Finished processing all sub-runs for config: {config_path}")
    except Exception as e:
        logger.error(f"[Run {run_id}] Error running experiment {config_path} (GPU {gpu_id}): {e}", exc_info=True)


def main_parallel(configs_base_dir: str, num_processes: int, gpu_ids_str: str = None):
    """
    Main function to orchestrate parallel execution of experiments.
    """
    all_config_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(configs_base_dir)
        for file in files if file == "config.yaml"
    ]

    if not all_config_files:
        logger.warning(f"No config.yaml files found in {configs_base_dir}. Exiting.")
        return

    logger.info(f"Found {len(all_config_files)} configuration files.")

    assigned_gpu_ids = None
    if gpu_ids_str:
        assigned_gpu_ids = [int(g.strip()) for g in gpu_ids_str.split(',')]
        logger.info(f"Using specified GPUs: {assigned_gpu_ids} for parallel execution.")
    elif torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        if num_processes > num_cuda_devices:
            logger.warning(f"Requested {num_processes} processes but only {num_cuda_devices} CUDA devices available. "
                           f"Limiting processes to {num_cuda_devices}.")
            num_processes = num_cuda_devices
        assigned_gpu_ids = list(range(num_cuda_devices))
        logger.info(f"Automatically detected and using {num_cuda_devices} GPUs: {assigned_gpu_ids}.")
    else:
        logger.info("No GPUs specified or detected. Running on CPU.")

    logger.info(f"Starting parallel execution with {num_processes} processes.")

    # Use the custom NestablePool instead of multiprocessing.Pool
    with NestablePool(processes=num_processes) as pool:
        tasks = []
        for i, config_path in enumerate(all_config_files):
            current_gpu_id = assigned_gpu_ids[i % len(assigned_gpu_ids)] if assigned_gpu_ids else None
            tasks.append((config_path, i + 1, current_gpu_id))
        pool.starmap(run_single_experiment, tasks)

    logger.info("All parallel experiments completed.")


if __name__ == "__main__":
    # It's good practice to set the start method to 'spawn' for CUDA.
    if multiprocessing.get_start_method(allow_none=True) is None:
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass  # It might already be set

    parser = argparse.ArgumentParser(description="Run multiple FL experiments in parallel.")
    parser.add_argument("--configs_dir", type=str, default="configs_generated",
                        help="Base directory containing generated config files.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(),
                        help="Number of parallel processes to run.")
    parser.add_argument("--gpu_ids", type=str,
                        help="Comma-separated list of CUDA device IDs to use, e.g., '2,3,4,5'.")
    args = parser.parse_args()

    # Adjust num_processes to match the number of specified GPUs
    if args.gpu_ids:
        num_specified_gpus = len(args.gpu_ids.split(','))
        if args.num_processes != num_specified_gpus:
            logger.info(f"Adjusting num_processes from {args.num_processes} to {num_specified_gpus} "
                        f"to match the number of specified GPU IDs.")
            args.num_processes = num_specified_gpus

    main_parallel(args.configs_dir, args.num_processes, args.gpu_ids)
