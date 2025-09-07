import argparse
import copy
import logging
import multiprocessing
import os
from pathlib import Path

from entry.gradient_market.automate_exp.config_parser import load_config
from entry.gradient_market.run_all_exp import run_attack

# Configure basic logging for the parallel runner
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_single_experiment(config_path: str, run_id: int):
    """
    Function to run a single experiment based on a config file.
    This function will be executed in a separate process.
    """
    try:
        logging.info(f"[Run {run_id}] Starting experiment with config: {config_path}")
        # Load the configuration
        app_config = load_config(config_path)

        # Apply any modifications for this specific run
        # Your original `main` function in test.py already handles n_samples,
        # but if we are running each config file as a single distinct experiment,
        # we might want to ensure save paths are unique for parallel runs.
        # However, your generated config files are already unique per varying parameter.

        # For the original `test.py`'s `main` function:
        # If your `test.py`'s `main()` function already handles `n_samples` internally
        # and creates `run_0_seed_X`, `run_1_seed_Y` etc., then calling `main()`
        # with just `app_config` (which already has the correct `save_path` for the
        # base config) might be what you want.

        # However, if `test.py` expects to be run with a *single* config file
        # and `n_samples` is handled by *this* parallel runner, then you'd
        # want to integrate the `n_samples` loop here.

        # Let's assume `run_attack` is the function that takes an `AppConfig` object
        # and executes a single experiment for that config (potentially including its own internal n_samples loop if `run_attack` calls `main()` like behavior).

        # If `run_attack` directly takes the AppConfig and performs a single run:
        # Make sure the save_path in the config is unique for this specific run.
        # The `config.yaml` paths are already unique, so `app_config.experiment.save_path`
        # will naturally point to something like `configs_generated/poison_vary_adv_rate_celeba/pr-0p3_adv-0p1_aggregation_method-fedavg/`.
        # Your `run_attack` then saves results into that unique base path.
        # If `test.py`'s `main` is called for each run, it will handle `n_samples`.
        # If we want to run `n_samples` *within* each parallel process, you can do:

        initial_seed = app_config.seed
        for i in range(app_config.n_samples):
            run_cfg = copy.deepcopy(app_config)
            current_seed = initial_seed + i
            # set_seed(current_seed) # If you have this function defined

            # Ensure unique save path for each sub-run within a config
            original_base_save_path = Path(
                run_cfg.experiment.save_path)  # Assumes save_path is initially the config folder
            run_save_path = original_base_save_path / f"run_{i}_seed_{current_seed}"
            run_save_path.mkdir(parents=True, exist_ok=True)
            run_cfg.experiment.save_path = str(run_save_path)

            logging.info(
                f"[Run {run_id} - Sub-run {i + 1}] Starting with seed {current_seed}, saving to {run_save_path}")
            run_attack(run_cfg)

        logging.info(f"[Run {run_id}] Finished experiment with config: {config_path}")
    except Exception as e:
        logging.error(f"[Run {run_id}] Error running experiment {config_path}: {e}", exc_info=True)


def main_parallel(configs_base_dir: str, num_processes: int, device_mapping: dict = None):
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
    logging.info(f"Starting parallel execution with {num_processes} processes.")

    # Use multiprocessing.Pool for managing a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Create arguments for each task: (config_path, run_id)
        tasks = []
        for i, config_path in enumerate(all_config_files):
            tasks.append((config_path, i + 1))  # run_id for logging

        pool.starmap(run_single_experiment, tasks)

    logging.info("All parallel experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple FL experiments in parallel.")
    parser.add_argument("--configs_dir", type=str, default="configs_generated",
                        help="Base directory containing generated config files.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(),
                        help="Number of parallel processes to run. Defaults to CPU count.")
    parser.add_argument("--cuda_devices", type=str,
                        help="Comma-separated list of CUDA device IDs, e.g., '0,1,2'. "
                             "Processes will be assigned devices in a round-robin fashion. "
                             "Only applicable if using GPUs.")
    args = parser.parse_args()

    # If using GPUs, you might want to adjust `run_single_experiment` to set CUDA_VISIBLE_DEVICES
    # per process. This is more complex and usually handled at the `run_single_experiment` level
    # by setting the environment variable or configuring torch.
    # For simplicity, we'll keep the device assignment simple for now, relying on PyTorch's
    # default or config's device setting.

    # Example of how to modify `run_single_experiment` for GPU assignment:
    # def run_single_experiment(config_path: str, run_id: int, gpu_id: str = None):
    #     if gpu_id:
    #         os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    #         logging.info(f"[Run {run_id}] Assigned to GPU {gpu_id}")
    #     # ... rest of your code ...
    #
    # Then in main_parallel:
    #     if args.cuda_devices:
    #         gpu_ids = args.cuda_devices.split(',')
    #         # Distribute tasks with GPU IDs
    #         tasks = []
    #         for i, config_path in enumerate(all_config_files):
    #             assigned_gpu_id = gpu_ids[i % len(gpu_ids)]
    #             tasks.append((config_path, i + 1, assigned_gpu_id))
    #         pool.starmap(run_single_experiment, tasks)
    #     else:
    #         # ... current logic ...

    # Basic logic for `test.py`'s `main` is now in `run_single_experiment`.
    # Make sure `run_attack` is imported correctly.
    # You might need to change `from test import run_attack` to `from entry.gradient_market.run_all_exp import run_attack`
    # or wherever your `run_attack` function is actually defined.

    main_parallel(args.configs_dir, args.num_processes)
