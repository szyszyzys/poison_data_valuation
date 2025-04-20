# automate_experiments.py
import subprocess
import os
import logging
import argparse
import time

# Configure logging for the automation script itself
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("automation.log"), # Log automation steps to a file
        logging.StreamHandler() # Also print to console
    ]
)
logger = logging.getLogger("Automation")

# --- Configuration ---
# Path to the main script that runs a single experiment
# Assumes it's in the same directory, adjust if needed
RUNNER_SCRIPT_PATH = "run_experiment.py"

# List of configuration files to run
# Create a 'configs' directory and put your YAML files there
CONFIG_FILES_TO_RUN = [
    "configs/exp_cifar_dirichlet.yaml",
    "configs/exp_fmnist_iid.yaml",
    # Add more config file paths here...
    # "configs/exp_agnews_discovery.yaml",
    # "configs/exp_trec_label.yaml",
]
# --- End Configuration ---


def run_single_exp_process(config_path: str) -> bool:
    """
    Runs a single experiment defined by the config file using a subprocess.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        True if the subprocess completed successfully (return code 0), False otherwise.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}. Skipping.")
        return False
    if not os.path.exists(RUNNER_SCRIPT_PATH):
        logger.error(f"Runner script '{RUNNER_SCRIPT_PATH}' not found. Aborting.")
        # Or raise an exception if this is critical
        return False # Indicate failure for this specific one if looping

    logger.info(f"--- Launching Experiment from: {config_path} ---")
    command = ["python", RUNNER_SCRIPT_PATH, config_path]
    start_time = time.time()

    try:
        # Execute the runner script as a separate process
        # Capture output to check for errors
        result = subprocess.run(
            command,
            capture_output=True, # Capture stdout and stderr
            text=True,           # Decode output as text
            check=False,         # Don't raise exception on non-zero exit code
            encoding='utf-8'     # Specify encoding
        )
        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            logger.info(f"--- Experiment SUCCEEDED: {config_path} (Duration: {duration:.2f}s) ---")
            # Optionally log some tail output if needed
            # logger.debug(f"Stdout tail:\n{result.stdout[-1000:]}")
            return True
        else:
            logger.error(f"--- Experiment FAILED: {config_path} (Duration: {duration:.2f}s) ---")
            logger.error(f"Return Code: {result.returncode}")
            # Log stdout and stderr for debugging failed runs
            logger.error("--- Captured STDOUT ---")
            logger.error(result.stdout if result.stdout else " (No stdout captured)")
            logger.error("--- Captured STDERR ---")
            logger.error(result.stderr if result.stderr else " (No stderr captured)")
            return False

    except FileNotFoundError:
         logger.error(f"Error: 'python' command not found or script path incorrect.")
         return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while running subprocess for {config_path}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Automate running multiple FL experiments.")
    parser.add_argument(
        '-c', '--configs',
        nargs='*', # 0 or more arguments
        default=CONFIG_FILES_TO_RUN, # Use the hardcoded list by default
        help='Optional list of specific config file paths to run.'
    )
    # Potential future arguments: --parallel N, --rerun-failed

    args = parser.parse_args()
    configs_to_process = args.configs

    logger.info("="*60)
    logger.info("Starting Experiment Automation")
    logger.info(f"Using runner script: {RUNNER_SCRIPT_PATH}")
    logger.info(f"Processing {len(configs_to_process)} configuration file(s):")
    for cfg in configs_to_process:
        logger.info(f"  - {cfg}")
    logger.info("="*60)

    successful_runs = 0
    failed_runs = 0
    start_total_time = time.time()

    for config_file in configs_to_process:
        success = run_single_exp_process(config_file)
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
        # Optional: Add a small delay between runs if needed
        # time.sleep(2)

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time

    logger.info("="*60)
    logger.info("Automation Summary")
    logger.info(f"Total experiments attempted: {len(configs_to_process)}")
    logger.info(f"  Successful: {successful_runs}")
    logger.info(f"  Failed:     {failed_runs}")
    logger.info(f"Total automation duration: {total_duration:.2f} seconds")
    logger.info("Automation finished. Check 'automation.log' and individual experiment results.")
    logger.info("="*60)


if __name__ == "__main__":
    # --- Ensure the main runner script exists ---
    if not os.path.exists(RUNNER_SCRIPT_PATH):
         logger.critical(f"FATAL ERROR: The main experiment runner script '{RUNNER_SCRIPT_PATH}' was not found. Aborting.")
    else:
         main()