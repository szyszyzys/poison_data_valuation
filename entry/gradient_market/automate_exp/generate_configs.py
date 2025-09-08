# generate_configs.py

from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator
from entry.gradient_market.automate_exp.scenarios import ALL_SCENARIOS


def main():
    """Generates all configurations defined in scenarios.py."""
    output_dir = "./configs_generated/privacy"
    generator = ExperimentGenerator(output_dir)

    # The loop is now simpler and more powerful
    for scenario in ALL_SCENARIOS:
        # Get the correct base config for THIS specific scenario
        base_config = scenario.base_config_factory()
        # Generate all variations
        generator.generate(base_config, scenario)

    print(f"\n✅ All configurations have been generated in '{output_dir}'")


if __name__ == "__main__":
    main()
