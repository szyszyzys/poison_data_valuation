import sys
import copy
from typing import Callable, Dict, List, Any, Optional
from pathlib import Path # Added Path

# --- Assume these are correctly imported based on your project structure ---
from entry.gradient_market.automate_exp.base_configs import (
    get_base_image_config, get_base_text_config, get_base_tabular_config
)
from entry.gradient_market.automate_exp.scenarios import (
    Scenario,
    use_cifar10_config, use_cifar100_config, use_trec_config,
    disable_all_seller_attacks
)
# Import ALL required attack modifiers
from your_module import (
    use_image_backdoor_attack, use_text_backdoor_attack, use_tabular_backdoor_with_trigger,
    use_image_label_flipping_attack, use_text_label_flipping_attack, use_tabular_label_flipping_attack,
    use_sybil_attack_strategy,
    use_adaptive_attack,
    use_buyer_dos_attack, use_buyer_starvation_attack, use_buyer_erosion_attack,
    use_buyer_class_exclusion_attack, use_buyer_oscillating_attack,
    use_competitor_mimicry_attack,
    TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL,
)
# Import Config classes and generator
try:
    from common.gradient_market_configs import AppConfig, PoisonType, BuyerAttackConfig, ValuationConfig # Added ValuationConfig
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- END IMPORTS ---



# === Main Summary (with expensive valuation) ===
def generate_main_summary_scenarios_with_valuation() -> List[Scenario]:
     print("\n--- Generating Main Summary Scenarios (with Valuation) ---")
     scenarios = []
     fixed_attack_params = {
         "experiment.adv_rate": [DEFAULT_ADV_RATE],
         "adversary_seller_config.poisoning.poison_rate": [DEFAULT_POISON_RATE],
     }
     DATASET_MODELS = [
         ("image", get_base_image_config, use_cifar10_config, use_image_backdoor_attack, "cifar10_cnn", IMAGE_DEFENSES),
         ("image", get_base_image_config, use_cifar10_config, use_image_backdoor_attack, "cifar10_resnet18", IMAGE_DEFENSES),
         ("tabular", get_base_tabular_config, lambda cfg:cfg, use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL), "mlp_texas100_baseline", TEXT_TABULAR_DEFENSES),
         # Add more dataset/model combinations as needed
     ]

     for modality, base_factory, dataset_mod, attack_mod, model_cfg_name, defense_list in DATASET_MODELS:
         model_cfg_param_key = f"experiment.{modality}_model_config_name" # Determine key

         for defense_name in defense_list:
              if defense_name not in TUNED_DEFENSE_PARAMS: continue
              tuned_params = TUNED_DEFENSE_PARAMS[defense_name]

              fixed_params_modifier = create_fixed_params_modifier(modality, tuned_params, model_cfg_name)

              # --- Enable Valuation: Influence + Periodic LOO & KernelSHAP ---
              valuation_modifier = enable_valuation(
                  influence=True,
                  loo=True, loo_freq=10,        # Run LOO every 10 rounds
                  kernelshap=True, kshap_freq=20 # Run KSHAP every 20 rounds
              )

              grid = {
                  model_cfg_param_key: [model_cfg_name],
                  "experiment.dataset_name": [model_cfg_name.split("_")[0]], # Infer dataset
                  "n_samples": [NUM_SEEDS_PER_CONFIG],
                  **fixed_attack_params
              }

              scenarios.append(Scenario(
                  name=f"main_summary_val_{defense_name}_{model_cfg_name}",
                  base_config_factory=base_factory,
                  modifiers=[fixed_params_modifier, dataset_mod, attack_mod, valuation_modifier], # Added valuation
                  parameter_grid=grid
              ))
     return scenarios

# === Advanced Sybil Comparison (with Influence + Periodic LOO) ===
def generate_advanced_sybil_scenarios_with_valuation() -> List[Scenario]:
    print("\n--- Generating Advanced Sybil Scenarios (with Valuation) ---")
    scenarios = []
    SYBIL_TEST_CONFIG = {"baseline_no_sybil": None, "mimic": {}, "oracle_blend": {"blend_alpha": [0.1, 0.5]}, "systematic_probe": {}} # Example alpha sweep
    DATASET_CONFIG = {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "cifar10", "model_config_suffix": "resnet18", "dataset_modifier": use_cifar10_config, "attack_modifier": use_image_backdoor_attack, "model_config_param_key": "experiment.image_model_config_name"}
    modality = DATASET_CONFIG["modality_name"]
    dataset_name = DATASET_CONFIG["dataset_name"]
    model_config_suffix = DATASET_CONFIG["model_config_suffix"]
    model_config_name = f"{dataset_name}_{model_config_suffix}"
    fixed_attack_params = {"experiment.adv_rate": [DEFAULT_ADV_RATE],"adversary_seller_config.poisoning.poison_rate": [DEFAULT_POISON_RATE]}

    for defense_name in IMAGE_DEFENSES:
        if defense_name not in TUNED_DEFENSE_PARAMS: continue
        tuned_defense_params = TUNED_DEFENSE_PARAMS[defense_name]
        fixed_params_modifier = create_fixed_params_modifier(modality, tuned_defense_params, model_config_name)
        # --- Enable Valuation: Influence + Periodic LOO ---
        valuation_modifier = enable_valuation(influence=True, loo=True, loo_freq=10, kernelshap=False) # LOO useful for Sybil value inflation

        base_grid = {DATASET_CONFIG["model_config_param_key"]: [model_config_name], "experiment.dataset_name": [dataset_name], "n_samples": [NUM_SEEDS_PER_CONFIG], **fixed_attack_params}

        for strategy_name, strategy_params_sweep in SYBIL_TEST_CONFIG.items():
            current_modifiers = [fixed_params_modifier, DATASET_CONFIG["dataset_modifier"], DATASET_CONFIG["attack_modifier"], valuation_modifier] # Added valuation
            current_grid = base_grid.copy()
            scenario_name_suffix = f"{defense_name}_{dataset_name}_{model_config_suffix}"

            if strategy_name == "baseline_no_sybil":
                 scenario_name = f"adv_sybil_baseline_{scenario_name_suffix}"
                 current_grid["adversary_seller_config.sybil.is_sybil"] = [False]
            else:
                 sybil_modifier = use_sybil_attack_strategy(strategy=strategy_name) # Basic modifier
                 current_modifiers.append(sybil_modifier)
                 scenario_name = f"adv_sybil_{strategy_name}_{scenario_name_suffix}"
                 # Handle parameter sweeps (e.g., alpha)
                 if strategy_params_sweep:
                     sweep_key, sweep_values = next(iter(strategy_params_sweep.items()))
                     config_key_path = f"adversary_seller_config.sybil.{sweep_key}"
                     current_grid[config_key_path] = sweep_values
                     scenario_name += f"_sweep_{sweep_key}" # Indicate sweep in name

            scenarios.append(Scenario(name=scenario_name, base_config_factory=DATASET_CONFIG["base_config_factory"], modifiers=current_modifiers, parameter_grid=current_grid))
    return scenarios

# === Adaptive Attack (with Influence only) ===
def generate_adaptive_attack_scenarios_with_valuation() -> List[Scenario]:
    print("\n--- Generating Adaptive Attack Scenarios (with Valuation) ---")
    scenarios = []
    ADAPTIVE_MODES_TO_TEST = ["gradient_manipulation", "data_manipulation"]
    ADAPTIVE_THREAT_MODEL = "black_box"
    DATASET_CONFIG = {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "cifar10", "model_config_suffix": "resnet18", "dataset_modifier": use_cifar10_config, "model_config_param_key": "experiment.image_model_config_name"}
    modality = DATASET_CONFIG["modality_name"]
    dataset_name = DATASET_CONFIG["dataset_name"]
    model_config_suffix = DATASET_CONFIG["model_config_suffix"]
    model_config_name = f"{dataset_name}_{model_config_suffix}"

    for defense_name in IMAGE_DEFENSES:
        if defense_name not in TUNED_DEFENSE_PARAMS: continue
        tuned_defense_params = TUNED_DEFENSE_PARAMS[defense_name]
        fixed_params_modifier = create_fixed_params_modifier(modality, tuned_defense_params, model_config_name)
        # --- Enable Valuation: Influence Only ---
        valuation_modifier = enable_valuation(influence=True, loo=False, kernelshap=False) # Keep it fast

        for adaptive_mode in ADAPTIVE_MODES_TO_TEST:
            adaptive_modifier = use_adaptive_attack(mode=adaptive_mode, threat_model=ADAPTIVE_THREAT_MODEL)
            grid = {DATASET_CONFIG["model_config_param_key"]: [model_config_name], "experiment.dataset_name": [dataset_name], "experiment.adv_rate": [DEFAULT_ADV_RATE], "n_samples": [NUM_SEEDS_PER_CONFIG]}
            scenario_name = f"adaptive_{ADAPTIVE_THREAT_MODEL}_{adaptive_mode}_{defense_name}_{dataset_name}_{model_config_suffix}"
            scenarios.append(Scenario(name=scenario_name, base_config_factory=DATASET_CONFIG["base_config_factory"], modifiers=[fixed_params_modifier, DATASET_CONFIG["dataset_modifier"], adaptive_modifier, valuation_modifier], parameter_grid=grid)) # Added valuation
    return scenarios

# === Competitor Mimicry (with Influence + Periodic LOO) ===
def generate_competitor_mimicry_scenarios_with_valuation() -> List[Scenario]:
    print("\n--- Generating Competitor Mimicry Scenarios (with Valuation) ---")
    scenarios = []
    MIMICRY_STRATEGIES = ["noisy_copy"] # Focus on one
    ADV_RATES = [0.3] # Fix rate for simplicity, or sweep [0.2, 0.3, 0.4]
    DATASET_CONFIG = {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "cifar10", "model_config_suffix": "resnet18", "dataset_modifier": use_cifar10_config, "model_config_param_key": "experiment.image_model_config_name"}
    modality = DATASET_CONFIG["modality_name"]
    dataset_name = DATASET_CONFIG["dataset_name"]
    model_config_suffix = DATASET_CONFIG["model_config_suffix"]
    model_config_name = f"{dataset_name}_{model_config_suffix}"

    for defense_name in IMAGE_DEFENSES: # Or just ['fltrust', 'martfl'] if focused
        if defense_name not in TUNED_DEFENSE_PARAMS: continue
        tuned_defense_params = TUNED_DEFENSE_PARAMS[defense_name]
        fixed_params_modifier = create_fixed_params_modifier(modality, tuned_defense_params, model_config_name)
        # --- Enable Valuation: Influence + Periodic LOO ---
        valuation_modifier = enable_valuation(influence=True, loo=True, loo_freq=10, kernelshap=False) # LOO needed to see target's value drop

        for strategy in MIMICRY_STRATEGIES:
            mimicry_modifier = use_competitor_mimicry_attack(target_seller_id="bn_0", strategy=strategy, noise_scale=0.03) # Target first benign
            grid = {DATASET_CONFIG["model_config_param_key"]: [model_config_name], "experiment.dataset_name": [dataset_name], "experiment.adv_rate": ADV_RATES, "n_samples": [NUM_SEEDS_PER_CONFIG]}
            scenario_name = f"comp_mimicry_{strategy}_{defense_name}_{dataset_name}_{model_config_suffix}"
            scenarios.append(Scenario(name=scenario_name, base_config_factory=DATASET_CONFIG["base_config_factory"], modifiers=[fixed_params_modifier, DATASET_CONFIG["dataset_modifier"], mimicry_modifier, valuation_modifier], parameter_grid=grid)) # Added valuation
    return scenarios


# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == "__main__":

    base_output_dir = "./configs_generated_benchmark_with_valuation"
    Path(base_output_dir).mkdir(parents=True, exist_ok=True) # Create base dir

    # --- Select which experiment groups to generate configs for ---
    generation_tasks_valuation = [
        ("step12_main_summary_val", generate_main_summary_scenarios_with_valuation),
        ("step6_advanced_sybil_val", generate_advanced_sybil_scenarios_with_valuation),
        ("step7_adaptive_attack_val", generate_adaptive_attack_scenarios_with_valuation),
        ("step9_competitor_mimicry_val", generate_competitor_mimicry_scenarios_with_valuation),
        # Add other steps here ONLY IF you specifically want valuation data for them
    ]

    all_generated_configs = 0

    # --- Execute Selected Tasks ---
    for subdir, gen_func in generation_tasks_valuation:
        output_dir = Path(base_output_dir) / subdir
        generator = ExperimentGenerator(str(output_dir))
        scenarios_to_generate = gen_func()

        if not scenarios_to_generate: continue

        task_configs = 0
        for scenario in scenarios_to_generate:
             # --- Default Generation Logic (Handles sweeps in grid) ---
             base_config = scenario.base_config_factory()
             modified_base_config = copy.deepcopy(base_config)
             # Apply ALL modifiers first (sets fixed HPs, attacks, valuation flags)
             for modifier in scenario.modifiers:
                  modified_base_config = modifier(modified_base_config)
             # Generator then expands the parameter grid
             num_gen = generator.generate(modified_base_config, scenario)
             task_configs += num_gen

        print(f"-> Generated {task_configs} configs for {subdir}")
        all_generated_configs += task_configs

    print(f"\n✅✅✅ Valuation config generation complete! ✅✅✅")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved under: {base_output_dir}")
    print("\nNext steps:")
    print("1. CRITICAL: Review and fill in GOLDEN_TRAINING_PARAMS and TUNED_DEFENSE_PARAMS.")
    print(f"2. Run experiments using run_parallel.py --configs_dir {base_output_dir}/<subdir_name>")
    print("3. Analyze results focusing on the valuation metrics collected.")