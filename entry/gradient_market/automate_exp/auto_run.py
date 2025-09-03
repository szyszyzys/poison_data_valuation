# automate_runs.py
import argparse
import itertools
import logging
import os
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

# --- Configuration ---
DEFAULT_CONFIG_DIR = "./configs_generated"
DEFAULT_RUNNER_SCRIPT = "./entry/gradient_market/attack_new.py"

# Configure logging
log_file = "automation_runs.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s/%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger("AutomationRunner")


def get_gpu_count() -> int:
    """Checks for available GPUs using PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
    except ImportError:
        logger.warning("PyTorch not found. Assuming 0 GPUs.")
        return 0


def run_single_experiment(
        config_path: str,
        runner_script: str,
        gpu_id: Optional[int] = None
) -> bool:
    """Runs a single experiment, optionally assigning a specific GPU."""
    if not os.path.exists(config_path):
        logger.error(f"[GPU {gpu_id or 'CPU'}] Config file not found: {config_path}. Skipping.")
        return False

    experiment_id = Path(config_path).stem
    gpu_assignment_info = f"GPU {gpu_id}" if gpu_id is not None else "CPU/Default"
    logger.info(f"--- [{gpu_assignment_info}] Launching Experiment [{experiment_id}] ---")

    command = [sys.executable, runner_script, config_path]
    start_time = time.time()

    # Prepare environment for the subprocess
    sub_env = os.environ.copy()
    if gpu_id is not None:
        sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, env=sub_env
        )
        duration = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"--- [{gpu_assignment_info}] SUCCESS [{experiment_id}] (Duration: {duration:.2f}s) ---")
            return True
        else:
            logger.error(f"--- [{gpu_assignment_info}] FAILED [{experiment_id}] (Duration: {duration:.2f}s) ---")
            logger.error(f"  Return Code: {result.returncode}\n  STDERR Tail:\n{result.stderr[-2000:]}")
            return False
    except Exception as e:
        logger.error(f"An unexpected error occurred for {config_path} on {gpu_assignment_info}: {e}", exc_info=True)
        return False


def pool_worker(task_info: tuple) -> bool:
    """Unpacks the task tuple and calls the main experiment runner function."""
    config_path, runner_script_path, assigned_gpu_id = task_info
    return run_single_experiment(config_path, runner_script_path, assigned_gpu_id)


def main():
    parser = argparse.ArgumentParser(description="Run all experiments defined by YAML config files.")
    parser.add_argument('--config-dir', type=str, default=DEFAULT_CONFIG_DIR)
    parser.add_argument('--runner-script', type=str, default=DEFAULT_RUNNER_SCRIPT)
    parser.add_argument('--pattern', type=str, default='**/*.yaml')
    parser.add_argument('--parallel', type=int, default=1,
                        help="Number of experiments to run in parallel. Use 0 for all available cores/GPUs.")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    runner_script = Path(args.runner_script)

    if not config_dir.is_dir():
        logger.critical(f"Configuration directory not found: {config_dir}")
        return
    if not runner_script.is_file():
        logger.critical(f"Runner script not found: {runner_script}")
        return

    config_files = sorted(list(config_dir.glob(args.pattern)))
    if not config_files:
        logger.warning(f"No config files found matching '{args.pattern}' in {config_dir}. Exiting.")
        return

    logger.info("=" * 60)
    logger.info(f"Found {len(config_files)} experiments to run.")

    num_gpus = get_gpu_count()
    max_workers = num_gpus if num_gpus > 0 else cpu_count()

    num_workers = args.parallel
    if num_workers <= 0:
        num_workers = max_workers
    else:
        num_workers = min(num_workers, max_workers)

    if num_gpus > 0:
        logger.info(f"Detected {num_gpus} GPUs. Running up to {num_workers} experiments in parallel.")
        gpu_id_cycle = itertools.cycle(range(num_gpus))
    else:
        logger.warning("No GPUs detected. Running on CPU.")
        gpu_id_cycle = itertools.cycle([None])

    # Prepare tasks for the multiprocessing pool
    tasks = [(str(p.resolve()), str(runner_script.resolve()), next(gpu_id_cycle)) for p in config_files]

    start_total_time = time.time()

    if num_workers > 1:
        logger.info(f"Starting parallel execution with {num_workers} workers.")
        with Pool(processes=num_workers) as pool:
            results = pool.map(pool_worker, tasks)
    else:
        logger.info("Starting sequential execution.")
        results = [pool_worker(task) for task in tasks]

    total_duration = time.time() - start_total_time

    # --- Summarize Results ---
    successful_runs = sum(1 for r in results if r)
    failed_runs = len(results) - successful_runs

    logger.info("=" * 60)
    logger.info("Automation Summary")
    logger.info(f"Total experiments processed: {len(results)}")
    logger.info(f"  Successful: {successful_runs}")
    logger.info(f"  Failed:     {failed_runs}")
    logger.info(f"Total duration: {total_duration:.2f} seconds")
    logger.info(f"Detailed logs are in: {log_file}")
    logger.info("=" * 60)

    if failed_runs > 0:
        logger.warning("Some experiments failed. Please check the logs.")


if __name__ == "__main__":
    main()