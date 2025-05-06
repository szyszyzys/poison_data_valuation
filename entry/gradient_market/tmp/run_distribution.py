#!/usr/bin/env python3
"""
automate_distribution.py

Walk all your YAML attack‐config files, and for each one run the exact same “data‐split” code
(get_image_data_distribution or poisoning_attack_text_reg_distri) that your attack uses,
across run_0, run_1, … run_{n_samples-1}.  This ensures every config’s splits are saved,
and uses identical arguments to your poison‐attack functions for perfect correlation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from entry.gradient_market.automate_exp.config_parser import parse_config_for_attack_function
from entry.gradient_market.backdoor_attack import set_seed, load_config
from entry.gradient_market.tmp.data_distribution_regen import get_image_data_distribution, \
    poisoning_attack_text_reg_distri
from model.utils import get_domain, get_model_name


def main():
    # 1) CLI
    parser = argparse.ArgumentParser(
        description="For each YAML config, generate & save its data splits (run_0...run_N)"
    )
    parser.add_argument(
        "--config-dir", "-c",
        type=str,
        default="./configs_generated",
        help="Directory containing your generated YAML configs"
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

    # 3) Find all config files
    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        logger.critical(f"Config directory not found: {config_dir}")
        sys.exit(1)

    configs = sorted(config_dir.glob(args.pattern))
    if not configs:
        logger.warning(f"No configs found in {config_dir} matching '{args.pattern}'")
        return

    logger.info(f"Found {len(configs)} config(s) to process")

    # 4) Process each config
    for cfg_path in configs:
        logger.info(f"\n=== Config: {cfg_path.name} ===")
        cfg = load_config(str(cfg_path))
        if cfg is None:
            logger.error("  ✗ load_config failed, skipping")
            continue

        # 5) Determine experiment folder
        final_save = cfg.get("output", {}).get("final_save_path")
        if not final_save:
            logger.error("  ✗ Missing 'output.final_save_path' in config, skipping")
            continue
        exp_base = Path(final_save)
        exp_base.mkdir(parents=True, exist_ok=True)

        # 6) Build shared args from your parse function
        attack_args: Dict[str, Any] = parse_config_for_attack_function(cfg)
        if attack_args is None:
            logger.error("  ✗ parse_config_for_attack_function failed, skipping")
            continue

        # 7) Inject model & save config snapshot
        dataset_name = cfg["dataset_name"]
        attack_args["model_structure"] = get_model_name(dataset_name)

        # 8) Determine domain & run counts
        domain = get_domain(dataset_name)
        n_samples = cfg.get("n_samples", 1)
        base_seed = cfg.get("seed", 42)

        # 9) Loop runs
        for i in range(n_samples):
            seed = base_seed + i
            set_seed(seed)

            run_dir = exp_base / f"run_{i}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # 10) Prepare per‐run kwargs
            run_kwargs = attack_args.copy()
            run_kwargs.update({
                "save_path": str(run_dir),
                "seed": seed
            })

            # 11) Save data splits
            try:
                logger.info(f"  → Saving data split (run_{i})")
                if domain == "image":
                    get_image_data_distribution(**run_kwargs)
                else:
                    poisoning_attack_text_reg_distri(**run_kwargs)
                logger.info("  ✓ Data split saved")
            except Exception as e:
                logger.error(f"  ✗ Error saving data split: {e}", exc_info=True)
                # optionally continue to next run

    logger.info("\nAll configurations processed.")


if __name__ == "__main__":
    main()
