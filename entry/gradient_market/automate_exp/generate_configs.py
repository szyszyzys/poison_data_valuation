# generate_configs.py

import torch

# Make sure all your dataclasses are importable from a single file
from common.gradient_market_configs import (
    AppConfig, ExperimentConfig, TrainingConfig, ServerPrivacyConfig,
    AdversarySellerConfig, DataConfig, ImageDataConfig, PropertySkewParams
)
from config_generator import ExperimentGenerator
from scenarios import ALL_SCENARIOS


def get_base_config() -> AppConfig:
    """Creates the default, base AppConfig object for all experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="CelebA",
            model_structure="SimpleCNN",
            aggregation_method="fedavg",
            global_rounds=100,
            n_sellers=30,
            adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu"
        ),
        training=TrainingConfig(local_epochs=2, batch_size=64, learning_rate=0.001),
        server_privacy=ServerPrivacyConfig(),
        adversary_seller_config=AdversarySellerConfig(),
        data=DataConfig(
            image=ImageDataConfig(
                property_skew=PropertySkewParams()
            )
        ),
        seed=42,
        n_samples=10,
    )


def main():
    """Generates all configurations defined in scenarios.py."""
    output_dir = "./configs_generated"
    base_config = get_base_config()
    generator = ExperimentGenerator(output_dir)

    for scenario in ALL_SCENARIOS:
        generator.generate(base_config, scenario)

    print(f"\n✅ All configurations have been generated in '{output_dir}'")


if __name__ == "__main__":
    main()
