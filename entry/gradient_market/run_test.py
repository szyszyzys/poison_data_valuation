import copy

from common.gradient_market_configs import AppConfig
from entry.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config
from entry.gradient_market.automate_exp.config_generator import set_nested_attr
from entry.gradient_market.automate_exp.scenarios import smoke_test_scenario, smoke_test_text_scenario
from entry.gradient_market.run_all_exp import run_attack


def create_test_config() -> AppConfig:
    """
    Builds the smoke test AppConfig object in memory.
    """
    print("--- Building Smoke Test Configuration ---")

    # 1. Get the base config
    base_config = get_base_image_config()

    # 2. Apply scenario modifiers
    scenario_config = copy.deepcopy(base_config)
    for modifier_func in smoke_test_scenario.modifiers:
        scenario_config = modifier_func(scenario_config)

    # 3. Apply the single parameter grid combination
    # Since the grid values are lists of size 1, we just take the first element.
    final_config = copy.deepcopy(scenario_config)
    for key, value_list in smoke_test_scenario.parameter_grid.items():
        set_nested_attr(final_config, key, value_list[0])

    # Manually set a temporary save path for the test run
    final_config.experiment.save_path = "./tmp_smoke_test_results"

    print("Configuration built successfully.")
    return final_config


def create_text_test_config() -> AppConfig:
    """
    Builds the text smoke test AppConfig object in memory.
    """
    print("--- Building Text Smoke Test Configuration ---")

    base_config = get_base_text_config()

    scenario_config = copy.deepcopy(base_config)
    for modifier_func in smoke_test_text_scenario.modifiers:
        scenario_config = modifier_func(scenario_config)

    final_config = copy.deepcopy(scenario_config)
    for key, value_list in smoke_test_text_scenario.parameter_grid.items():
        set_nested_attr(final_config, key, value_list[0])

    # --- CHANGE 3: Use a different temp directory ---
    final_config.experiment.save_path = "./tmp_smoke_test_text_results"

    print("Configuration built successfully.")
    return final_config


def main():
    """
    Runs the smoke test.
    """
    test_config = create_test_config()

    print("\n--- Starting Smoke Test ---")
    try:
        # Run your main experiment function with the test config
        run_attack(cfg=test_config)

        print("\n✅ ✅ ✅ SMOKE TEST PASSED ✅ ✅ ✅")
        print("The pipeline ran for a few rounds without crashing.")

    except Exception as e:
        print("\n❌ ❌ ❌ SMOKE TEST FAILED ❌ ❌ ❌")
        print("An error occurred during the test run:")
        # Re-raise the exception to see the full traceback
        raise e

    test_text_config = create_text_test_config()

    print("\n--- Starting Smoke Text Test ---")
    try:
        # Run your main experiment function with the test config
        run_attack(cfg=test_text_config)

        print("\n✅ ✅ ✅ SMOKE TEXT TEST PASSED ✅ ✅ ✅")
        print("The pipeline ran for a few rounds without crashing.")

    except Exception as e:
        print("\n❌ ❌ ❌ SMOKE TEXT TEST FAILED ❌ ❌ ❌")
        print("An error occurred during the test run:")
        # Re-raise the exception to see the full traceback
        raise e


if __name__ == "__main__":
    main()
