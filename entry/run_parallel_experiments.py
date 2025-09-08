import argparse
import copy
import logging
import multiprocessing
import os
from pathlib import Path

import torch  # Import torch to check for CUDA availability

from common.utils import set_seed
from entry.gradient_market.run_all_exp import run_attack

# --- CRITICAL CHANGE: Set the multiprocessing start method ---
# This must be done AT THE VERY BEGINNING of the script,
# before any other multiprocessing code or CUDA operations.
if __name__ == "__main__":  # Ensure this runs only in the main process
    # Check if a start method has already been set (e.g., by a library)
    # If not, set it to 'spawn' for CUDA compatibility
    if multiprocessing.get_start_method(allow_none=True) is None:
        try:
            multiprocessing.set_start_method("spawn")
            logging.info("Set multiprocessing start method to 'spawn' for CUDA compatibility.")
        except RuntimeError:
            # Handle cases where it might already be set or cannot be set
            logging.warning("Could not set multiprocessing start method to 'spawn'. It might already be set.")
            pass  # It might already be set by another library

# Adjust this import based on your actual test.py structure
# For example, if run_attack is in entry/gradient_market/run_all_exp.py
# from entry.gradient_market.run_all_exp import run_attack
from entry.gradient_market.automate_exp.config_parser import load_config

# Configure basic logging for the parallel runner
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Use 'logger' consistently


def run_single_experiment(config_path: str, run_id: int, gpu_id: int = None):
    """
    Function to run a single experiment based on a config file.
    This function will be executed in a separate process.
    """
    try:
        # It's crucial that CUDA_VISIBLE_DEVICES is set *before* any
        # CUDA operations (like model.to('cuda')) happen in this child process.
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # Clear any CUDA caches specific to this process for good measure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"[Run {run_id}] Assigned to GPU {gpu_id}. Starting experiment with config: {config_path}")
        else:
            if torch.cuda.is_available():
                logger.warning(
                    f"[Run {run_id}] CUDA is available but no specific GPU_ID assigned. PyTorch will use default CUDA device.")
            logger.info(f"[Run {run_id}] Starting experiment with config: {config_path}")

        # Load the configuration
        app_config = load_config(config_path)

        initial_seed = app_config.seed
        for i in range(app_config.n_samples):
            current_seed = initial_seed + i
            original_base_save_path = Path(app_config.experiment.save_path)
            run_save_path = original_base_save_path / f"run_{i}_seed_{current_seed}"

            # 1. DEFINE a "success marker" file that indicates a run is finished.
            #    IMPORTANT: Change "final_metrics.json" to whatever file your `run_attack`
            #    function creates at the very end of a successful run.
            success_marker_path = run_save_path / "final_metrics.json"

            # 2. CHECK if this file already exists.
            if success_marker_path.exists():
                logger.info(f"[Run {run_id} - Sub-run {i + 1}] Already completed. Skipping: {run_save_path}")
                # 3. SKIP to the next iteration if the run is done.
                continue

            run_cfg = copy.deepcopy(app_config)
            current_seed = initial_seed + i
            # If you have a set_seed function, uncomment and use it
            set_seed(current_seed)

            original_base_save_path = Path(run_cfg.experiment.save_path)
            run_save_path.mkdir(parents=True, exist_ok=True)
            run_cfg.experiment.save_path = str(run_save_path)

            logger.info(
                f"[Run {run_id} - Sub-run {i + 1}] Config: {config_path}, Seed: {current_seed}, Save Path: {run_save_path}")
            # Ensure the config's device setting matches the assigned GPU, if applicable
            if gpu_id is not None:
                # Inside each process, the assigned GPU is exposed as 'cuda:0' due to CUDA_VISIBLE_DEVICES
                run_cfg.experiment.device = f"cuda:{0}"
            else:
                run_cfg.experiment.device = "cpu"  # Default to CPU if no GPU assigned (or if no CUDA available)

            run_attack(run_cfg)

        logger.info(f"[Run {run_id}] Finished experiment with config: {config_path}")
    except Exception as e:
        logger.error(f"[Run {run_id}] Error running experiment {config_path} (GPU {gpu_id}): {e}", exc_info=True)


def main_parallel(configs_base_dir: str, num_processes: int, gpu_ids_str: str = None):
    """
    Main function to orchestrate parallel execution of experiments.
    """
    all_config_files = []
    for root, _, files in os.walk(configs_base_dir):
        for file in files:
            if file == "config.yaml":
                all_config_files.append(os.path.join(root, file))

    if not all_config_files:
        logger.warning(f"No config.yaml files found in {configs_base_dir}. Exiting.")
        return

    logger.info(f"Found {len(all_config_files)} configuration files.")

    # Determine GPU IDs to use
    assigned_gpu_ids = None
    if gpu_ids_str:
        assigned_gpu_ids = [int(g.strip()) for g in gpu_ids_str.split(',')]
        if len(assigned_gpu_ids) != num_processes:
            logger.warning(
                f"Number of specified GPU IDs ({len(assigned_gpu_ids)}) does not match num_processes ({num_processes}). "
                f"Will use GPU IDs in a round-robin fashion for {num_processes} processes.")
        logger.info(f"Using GPUs: {assigned_gpu_ids} for parallel execution.")
    elif torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        if num_processes > num_cuda_devices:
            logger.warning(f"Requested {num_processes} processes but only {num_cuda_devices} CUDA devices available. "
                           f"Limiting processes to {num_cuda_devices} and assigning one GPU per process.")
            num_processes = num_cuda_devices
        assigned_gpu_ids = list(range(num_cuda_devices))
        logger.info(f"Automatically detected and using {num_cuda_devices} GPUs: {assigned_gpu_ids}.")
    else:
        logger.info("No GPUs specified or detected. Running on CPU.")

    logger.info(f"Starting parallel execution with {num_processes} processes.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        tasks = []
        for i, config_path in enumerate(all_config_files):
            current_gpu_id = None
            if assigned_gpu_ids:
                # Round-robin assignment of GPUs to tasks
                current_gpu_id = assigned_gpu_ids[i % len(assigned_gpu_ids)]
            tasks.append((config_path, i + 1, current_gpu_id))

        pool.starmap(run_single_experiment, tasks)

    logger.info("All parallel experiments completed.")


if __name__ == "__main__":
    # The multiprocessing.set_start_method("spawn") must be called here,
    # before any other multiprocessing or CUDA-related code in the main block.
    # The `if __name__ == "__main__":` block at the very top already handles this.

    parser = argparse.ArgumentParser(description="Run multiple FL experiments in parallel.")
    parser.add_argument("--configs_dir", type=str, default="configs_generated",
                        help="Base directory containing generated config files.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(),
                        help="Number of parallel processes to run. Defaults to CPU count.")
    parser.add_argument("--gpu_ids", type=str,
                        help="Comma-separated list of CUDA device IDs to use, e.g., '0,1,2,3'. "
                             "If not specified, will attempt to use all available GPUs up to num_processes.")
    args = parser.parse_args()

    # If num_processes is explicitly set to 4 and gpu_ids is also 4, great.
    # If gpu_ids is provided, ensure num_processes matches, or use len(gpu_ids) as num_processes.
    if args.gpu_ids:
        num_specified_gpus = len(args.gpu_ids.split(','))
        if args.num_processes != num_specified_gpus:
            logger.info(f"Adjusting num_processes from {args.num_processes} to {num_specified_gpus} "
                        f"to match the number of specified GPU IDs.")
        args.num_processes = num_specified_gpus
    elif torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        if args.num_processes > num_cuda_devices:
            logger.warning(
                f"Requested {args.num_processes} processes but only {num_cuda_devices} CUDA devices available. "
                f"Limiting processes to {num_cuda_devices}.")
            args.num_processes = num_cuda_devices
    else:
        logger.warning("No CUDA devices detected. Running on CPU only.")

    main_parallel(args.configs_dir, args.num_processes, args.gpu_ids)
