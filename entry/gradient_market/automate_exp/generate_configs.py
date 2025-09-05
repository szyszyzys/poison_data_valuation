# generate_configs.py

import torch

from common.gradient_market_configs import (
    AppConfig, ExperimentConfig, TrainingConfig, AdversarySellerConfig, DataConfig, ImageDataConfig, PropertySkewParams,
    DiscoverySplitParams, VocabConfig,
    TextDataConfig, DebugConfig, ServerAttackConfig
)
from config_generator import ExperimentGenerator
from scenarios import ALL_SCENARIOS


def get_base_image_config() -> AppConfig:
    """Creates the default, base AppConfig for IMAGE-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="CelebA", model_structure="SimpleCNN", aggregation_method="fedavg",
            global_rounds=100, n_sellers=30, adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu"
        ),
        training=TrainingConfig(local_epochs=2, batch_size=64, learning_rate=0.001),
        server_attack_config=ServerAttackConfig(),
        adversary_seller_config=AdversarySellerConfig(),
        data=DataConfig(
            image=ImageDataConfig(
                property_skew=PropertySkewParams()
            )
        ),
        debug=DebugConfig(
            save_individual_gradients=False,
            gradient_save_frequency=10
        ),
        seed=42, n_samples=10,
    )


def get_base_text_config() -> AppConfig:
    """Creates the default, base AppConfig for TEXT-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="AG_NEWS", model_structure="BiLSTM", aggregation_method="fedavg",
            global_rounds=50, n_sellers=20, adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu"
        ),
        training=TrainingConfig(local_epochs=3, batch_size=32, learning_rate=0.001),
        server_attack_config=ServerAttackConfig(),
        adversary_seller_config=AdversarySellerConfig(),
        data=DataConfig(
            text=TextDataConfig(
                vocab=VocabConfig(),
                discovery=DiscoverySplitParams()
            )
        ),
        debug=DebugConfig(
            save_individual_gradients=False,
            gradient_save_frequency=10
        ),
        seed=42, n_samples=10,
    )


def main():
    """Generates all configurations defined in scenarios.py."""
    output_dir = "./configs_generated"
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
