#!/usr/bin/env python3
"""
automate_runs.py

Walk through all generated YAML configs, load each one exactly as your experiment runner does,
and invoke the data‐distribution step (without training) to ensure every experiment’s dataset
splits are saved to disk.
"""
import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict

from entry.gradient_market.tmp.data_distribution_regen import get_image_data_distribution, \
    poisoning_attack_text_reg_distri
from model.utils import get_domain


def main():
    # 1) CLI
    parser = argparse.ArgumentParser(
        description="For each YAML config, load it and save its dataset splits."
    )
    parser.add_argument(
        "--config-dir", "-c",
        type=str,
        default="./configs_generated",
        help="Directory containing YAML config files"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="**/*.yaml",
        help="Glob pattern to match config files"
    )
    args = parser.parse_args()

    # 2) Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        logger.critical(f"Config directory not found: {config_dir}")
        sys.exit(1)

    # 3) Find config files
    configs = sorted(config_dir.glob(args.pattern))
    if not configs:
        logger.warning(f"No configs found in {config_dir} with pattern '{args.pattern}'")
        return

    logger.info(f"Found {len(configs)} config(s) to process")

    # 4) Loop
    for cfg_path in configs:
        logger.info(f"Processing config: {cfg_path.name}")
        try:
            cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())
        except Exception as e:
            logger.error(f"  ✗ Failed to parse YAML: {e}")
            continue

        # 5) Extract common fields
        dataset_name = cfg.get("dataset_name")
        if not dataset_name:
            logger.error("  ✗ Missing 'dataset_name'; skipping")
            continue

        final_save = cfg.get("output", {}).get("final_save_path")
        if not final_save:
            logger.error("  ✗ Missing 'output.final_save_path'; skipping")
            continue
        final_save = Path(final_save)
        final_save.mkdir(parents=True, exist_ok=True)

        # 6) Build arguments for distribution functions
        base_args = {
            "dataset_name": dataset_name,
            "n_sellers": cfg.get("n_sellers", 10),
            "adv_rate": cfg.get("adv_rate", 0.0),
            "buyer_percentage": cfg.get("buyer_percentage", 0.02),
            "data_split_mode": cfg.get("data_split_mode", "NonIID"),
            "discovery_quality": cfg.get("discovery_quality", 0.3),
            "buyer_data_mode": cfg.get("buyer_data_mode", "random"),
            "batch_size": cfg.get("local_training_params", {}).get("batch_size", 64),
            "save_path": str(final_save),
            "seed": cfg.get("seed", 42),
        }

        # 7) Dispatch based on domain
        domain = get_domain(dataset_name)
        try:
            if domain == "image":
                logger.info(f"  → Saving image splits to {final_save}")
                get_image_data_distribution(**base_args)
            else:
                logger.info(f"  → Saving text splits to {final_save}")
                poisoning_attack_text_reg_distri(**base_args)
            logger.info(f"  ✓ Completed dataset saving for {cfg_path.name}")
        except Exception as e:
            logger.error(f"  ✗ Error during split saving: {e}", exc_info=True)

    logger.info("All configs processed.")


if __name__ == "__main__":
    main()
