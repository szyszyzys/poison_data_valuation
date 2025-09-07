import argparse
import copy
import logging
import multiprocessing
import os
from pathlib import Path

import torch  # Import torch to check for CUDA availability

from entry.gradient_market.automate_exp.config_parser import load_config
from entry.gradient_market.run_all_exp import run_attack

# Configure basic logging for the parallel runner
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_single_experiment(config_path: str, run_id: int, gpu_id: int = None):
    """
    Function to run a single experiment based on a config file.
    This function will be executed in a separate process.
    """
    try:
        if gpu_id is not None:
            # Crucially, set CUDA_VISIBLE_DEVICES for this process
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logging.info(f"[Run {run_id}] Assigned to GPU {gpu_id}. Starting experiment with config: {config_path}")
        else:
            # If no GPU ID is provided (e.g., running on CPU only or single GPU without explicit assignment)
            if torch.cuda.is_available():
                logging.warning(
                    f"[Run {run_id}] CUDA is available but no specific GPU_ID assigned. PyTorch will use default CUDA device.")
            logging.info(f"[Run {run_id}] Starting experiment with config: {config_path}")

        # Load the configuration
        app_config = load_config(config_path)

        # Your original `main` function in test.py already handles n_samples,
        # creating `run_0_seed_X`, `run_1_seed_Y` etc. This loop replicates that
        # behavior for each base config in a parallel manner.
        initial_seed = app_config.seed
        for i in range(app_config.n_samples):
            run_cfg = copy.deepcopy(app_config)
            current_seed = initial_seed + i
            # If you have a set_seed function, uncomment and use it
            # set_seed(current_seed)

            # Ensure unique save path for each sub-run within a config.
            # The base `save_path` in `app_config` should initially point to
            # `configs_generated/poison_vary_adv_rate_celeba/pr-0p3_adv-0p1_aggregation_method-fedavg/`
            # The following line then appends `run_i_seed_current_seed` to this.
            original_base_save_path = Path(run_cfg.experiment.save_path)
            run_save_path = original_base_save_path / f"run_{i}_seed_{current_seed}"
            run_save_path.mkdir(parents=True, exist_ok=True)
            run_cfg.experiment.save_path = str(run_save_path)

            logging.info(
                f"[Run {run_id} - Sub-run {i + 1}] Config: {config_path}, Seed: {current_seed}, Save Path: {run_save_path}")
            # Ensure the config's device setting matches the assigned GPU, if applicable
            if gpu_id is not None:
                run_cfg.experiment.device = f"cuda:{0}"  # Each process sees its assigned GPU as 'cuda:0'
            else:
                run_cfg.experiment.device = "cpu"  # Default to CPU if no GPU assigned

            run_attack(run_cfg)

        logging.info(f"[Run {run_id}] Finished experiment with config: {config_path}")
    except Exception as e:
        logging.error(f"[Run {run_id}] Error running experiment {config_path} (GPU {gpu_id}): {e}", exc_info=True)


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
        logging.warning(f"No config.yaml files found in {configs_base_dir}. Exiting.")
        return

    logging.info(f"Found {len(all_config_files)} configuration files.")

    # Determine GPU IDs to use
    assigned_gpu_ids = None
    if gpu_ids_str:
        assigned_gpu_ids = [int(g.strip()) for g in gpu_ids_str.split(',')]
        if len(assigned_gpu_ids) != num_processes:
            logging.warning(
                f"Number of specified GPU IDs ({len(assigned_gpu_ids)}) does not match num_processes ({num_processes}). "
                f"Will use GPU IDs in a round-robin fashion for {num_processes} processes.")
        logging.info(f"Using GPUs: {assigned_gpu_ids} for parallel execution.")
    elif torch.cuda.is_available():
        # If GPUs are available but not explicitly specified, try to use all available
        num_cuda_devices = torch.cuda.device_count()
        if num_processes > num_cuda_devices:
            logging.warning(f"Requested {num_processes} processes but only {num_cuda_devices} CUDA devices available. "
                            f"Limiting processes to {num_cuda_devices} and assigning one GPU per process.")
            num_processes = num_cuda_devices
        assigned_gpu_ids = list(range(num_cuda_devices))
        logging.info(f"Automatically detected and using {num_cuda_devices} GPUs: {assigned_gpu_ids}.")
    else:
        logging.info("No GPUs specified or detected. Running on CPU.")

    logging.info(f"Starting parallel execution with {num_processes} processes.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        tasks = []
        for i, config_path in enumerate(all_config_files):
            current_gpu_id = None
            if assigned_gpu_ids:
                # Round-robin assignment of GPUs to tasks
                current_gpu_id = assigned_gpu_ids[i % len(assigned_gpu_ids)]
            tasks.append((config_path, i + 1, current_gpu_id))

        pool.starmap(run_single_experiment, tasks)

    logging.info("All parallel experiments completed.")


if __name__ == "__main__":
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
        # Override num_processes based on the number of GPUs provided
        num_specified_gpus = len(args.gpu_ids.split(','))
        if args.num_processes != num_specified_gpus:
            logging.info(f"Adjusting num_processes from {args.num_processes} to {num_specified_gpus} "
                         f"to match the number of specified GPU IDs.")
        args.num_processes = num_specified_gpus
    elif torch.cuda.is_available():
        # If no gpu_ids specified, but CUDA is available, adjust num_processes to available GPUs
        num_cuda_devices = torch.cuda.device_count()
        if args.num_processes > num_cuda_devices:
            logging.warning(
                f"Requested {args.num_processes} processes but only {num_cuda_devices} CUDA devices available. "
                f"Limiting processes to {num_cuda_devices}.")
            args.num_processes = num_cuda_devices
    else:
        logging.warning("No CUDA devices detected. Running on CPU only.")

    main_parallel(args.configs_dir, args.num_processes, args.gpu_ids)
