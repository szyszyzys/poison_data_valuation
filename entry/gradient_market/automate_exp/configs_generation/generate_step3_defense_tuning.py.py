# FILE: generate_step3_defense_tuning.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports --- (Same as before)
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS, NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES
)
from entry.gradient_market.automate_exp.base_configs import (
    get_base_image_config, get_base_text_config
)
from entry.gradient_market.automate_exp.scenarios import (
    Scenario, use_cifar10_config, use_image_backdoor_attack, use_label_flipping_attack, use_cifar100_config,
    use_trec_config, use_text_backdoor_attack
)
from entry.gradient_market.automate_exp.tbl_new import get_base_tabular_config, use_tabular_backdoor_with_trigger, \
    TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL, PURCHASE100_TARGET_LABEL, PURCHASE100_TRIGGER

try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr, iter_grid
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- TUNING_GRIDS and TUNING_TARGETS_STEP3 (Same as before) ---
TUNING_GRIDS = {
    "fltrust": {
        "aggregation.clip_norm": [1.0, 5.0, 10.0, 20.0, None],
    },
    "martfl": {
        "aggregation.martfl.max_k": [3, 5, 7, 10],
        "aggregation.clip_norm": [5.0, 10.0, 20.0, None],
    },
    "skymask": {
        "aggregation.skymask.mask_epochs": [10, 20, 50],
        "aggregation.skymask.mask_lr": [0.01, 0.001],
        "aggregation.skymask.mask_threshold": [0.5, 0.7, 0.9],
        "aggregation.clip_norm": [10.0],
    }
}
ATTACK_TYPES_TO_TUNE = ["backdoor", "labelflip"]
TUNING_TARGETS_STEP3 = [
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Texas100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_texas100_baseline",
     "dataset_modifier": lambda cfg: cfg,
     "backdoor_attack_modifier": use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL),
     "labelflip_attack_modifier": use_label_flipping_attack},
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Purchase100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_purchase100_baseline",
     "dataset_modifier": lambda cfg: cfg,
     "backdoor_attack_modifier": use_tabular_backdoor_with_trigger(PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL),
     "labelflip_attack_modifier": use_label_flipping_attack},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "CIFAR10",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar10_cnn",
     "dataset_modifier": use_cifar10_config, "backdoor_attack_modifier": use_image_backdoor_attack,
     "labelflip_attack_modifier": use_label_flipping_attack},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "CIFAR100",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar100_cnn",
     "dataset_modifier": use_cifar100_config, "backdoor_attack_modifier": use_image_backdoor_attack,
     "labelflip_attack_modifier": use_label_flipping_attack},
    {"modality_name": "text", "base_config_factory": get_base_text_config, "dataset_name": "TREC",
     "model_config_param_key": "experiment.text_model_config_name", "model_config_name": "textcnn_trec_baseline",
     "dataset_modifier": use_trec_config, "backdoor_attack_modifier": use_text_backdoor_attack,
     "labelflip_attack_modifier": use_label_flipping_attack},
]
# --- generate_defense_tuning_scenarios (Same as before) ---
def generate_defense_tuning_scenarios() -> List[Scenario]:
    """Generates configs to tune defense HPs under fixed attacks."""
    print("\n--- Generating Step 3: Defense Tuning Scenarios ---")
    scenarios = []
    for target in TUNING_TARGETS_STEP3:
        modality = target["modality_name"]
        model_cfg_name = target["model_config_name"]
        print(f"-- Processing: {modality} {model_cfg_name}")

        for attack_type in ATTACK_TYPES_TO_TUNE:
            attack_modifier_key = f"{attack_type}_attack_modifier"
            if attack_modifier_key not in target: continue
            attack_modifier = target[attack_modifier_key]
            print(f"  -- Attack Type: {attack_type}")

            # Modifier to apply Golden HPs and Fixed Attack Settings
            def create_setup_modifier(
                    current_modifier=attack_modifier,
                    current_model_cfg_name=model_cfg_name  # <-- BIND THE VARIABLE HERE
            ):
                # Closure to capture the correct attack modifier
                def modifier(config: AppConfig) -> AppConfig:
                    training_params = GOLDEN_TRAINING_PARAMS.get(current_model_cfg_name)
                    if training_params:
                        for key, value in training_params.items():
                            set_nested_attr(config, key, value)
                    else:
                        print(f"  WARNING: No Golden HPs found for model '{current_model_cfg_name}'!")

                    config.experiment.adv_rate = DEFAULT_ADV_RATE
                    config = current_modifier(config)  # Sets attack type (backdoor/labelflip)
                    set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", DEFAULT_POISON_RATE)
                    config.adversary_seller_config.sybil.is_sybil = False
                    set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
                    set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
                    config.valuation.run_influence = False
                    config.valuation.run_loo = False
                    config.valuation.run_kernelshap = False
                    return config
                return modifier

            setup_modifier_func = create_setup_modifier()

            current_defenses = IMAGE_DEFENSES if modality == "image" else TEXT_TABULAR_DEFENSES
            for defense_name in current_defenses:
                if defense_name == "fedavg": continue
                if defense_name not in TUNING_GRIDS: continue

                defense_grid_to_sweep = TUNING_GRIDS[defense_name]
                print(f"    - Defense: {defense_name}")

                base_grid = {
                    target["model_config_param_key"]: [model_cfg_name],
                    "experiment.dataset_name": [target["dataset_name"]],
                    "n_samples": [NUM_SEEDS_PER_CONFIG],
                    "aggregation.method": [defense_name],
                    "experiment.use_early_stopping": [True],
                    "experiment.patience": [10],
                }
                if defense_name == "skymask":
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    base_grid["aggregation.skymask.sm_model_type"] = [model_struct]

                full_parameter_grid = {**base_grid, **defense_grid_to_sweep}

                scenarios.append(Scenario(
                    name=f"step3_tune_{defense_name}_{attack_type}_{modality}_{target['dataset_name']}_{model_cfg_name}",
                    base_config_factory=target["base_config_factory"],
                    modifiers=[setup_modifier_func, target["dataset_modifier"]],
                    parameter_grid=full_parameter_grid
                ))
    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step3_defense_tuning"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_defense_tuning_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 3 ---")

    # --- START OF MODIFICATION: Manual Loop ---
    # We will manually loop over the HPs to create unique names,
    # just like we do in the Step 4 script.

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        # Get the original base config and apply modifiers
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        # `iter_grid` (from config_generator) expands the grid into a list of dicts
        # Each dict is a single HP combination
        for hp_combo_dict in iter_grid(scenario.parameter_grid):

            # --- Build a unique name from the HPs that are being tuned ---
            hp_parts = []

            # Get the defense method for this scenario
            defense_name = hp_combo_dict.get("aggregation.method")

            # This logic MUST match your TUNING_GRIDS
            if defense_name == "fltrust":
                val = hp_combo_dict.get("aggregation.clip_norm", "None")
                hp_parts.append(f"aggregation.clip_norm_{val}")

            elif defense_name == "martfl":
                val = hp_combo_dict.get("aggregation.martfl.max_k", "None")
                hp_parts.append(f"aggregation.martfl.max_k_{val}")
                val = hp_combo_dict.get("aggregation.clip_norm", "None")
                hp_parts.append(f"aggregation.clip_norm_{val}")

            elif defense_name == "skymask":
                val = hp_combo_dict.get("aggregation.skymask.mask_epochs", "None")
                hp_parts.append(f"aggregation.skymask.mask_epochs_{val}")
                val = hp_combo_dict.get("aggregation.skymask.mask_lr", "None")
                hp_parts.append(f"aggregation.skymask.mask_lr_{val}")
                val = hp_combo_dict.get("aggregation.skymask.mask_threshold", "None")
                hp_parts.append(f"aggregation.skymask.mask_threshold_{val}")
                val = hp_combo_dict.get("aggregation.clip_norm", "None")
                hp_parts.append(f"aggregation.clip_norm_{val}")

            # Join parts to make the folder name
            hp_suffix = "_".join(hp_parts)
            if not hp_suffix:
                hp_suffix = "default_hps" # Fallback

            # --- Create a new temporary scenario for this single config ---

            # Create a grid that has only *one* value for each parameter
            current_grid = {key: [value] for key, value in hp_combo_dict.items()}

            # Set the unique save path for the *results*
            # This is the path your run_parallel.py will use
            unique_save_path = f"./results/{scenario.name}/{hp_suffix}"
            current_grid["experiment.save_path"] = [unique_save_path]

            # Set the unique name for the *config file*
            temp_scenario_name = f"{scenario.name}/{hp_suffix}"

            temp_scenario = Scenario(
                name=temp_scenario_name,
                base_config_factory=scenario.base_config_factory,
                modifiers=scenario.modifiers, # Modifiers are already applied, but good to keep
                parameter_grid=current_grid   # This grid has no lists, only single values
            )

            # Generate the single config file
            # We pass modified_base_config so modifiers aren't run again
            num_gen = generator.generate(modified_base_config, temp_scenario)
            task_configs += num_gen

        # --- END OF MODIFICATION ---

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs


    print(f"\n✅ Step 3 (Defense Tuning) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS in config_common_utils.py is correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(
        f"3. Analyze results using step3_analyze.py pointing to './results/'")
    print("4. Find the best defense HPs (good Acc, low ASR) for each defense/attack_type/model/dataset combo.")
    print(
        "5. Record these winning HPs -> TUNED_DEFENSE_PARAMS in config_common_utils.py")