import torch

from common.gradient_market_configs import (
    AppConfig, ExperimentConfig, TrainingConfig, AdversarySellerConfig, DataConfig, ImageDataConfig, PropertySkewParams,
    DiscoverySplitParams, VocabConfig, TextDataConfig, DebugConfig, ServerAttackConfig, AggregationConfig
)


def get_base_image_config() -> AppConfig:
    """Creates the default, base AppConfig for IMAGE-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="CelebA", model_structure="SimpleCNN", aggregation_method="fedavg",
            global_rounds=100, n_sellers=30, adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu", dataset_type="image"
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
        aggregation=AggregationConfig(
            method="fedavg"  # Set the default method here
        ),

    )


def get_base_text_config() -> AppConfig:
    """Creates the default, base AppConfig for TEXT-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="AG_NEWS", model_structure="BiLSTM", aggregation_method="fedavg",
            global_rounds=50, n_sellers=20, adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu", dataset_type="text"

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
        aggregation=AggregationConfig(
            method="fedavg"  # Set the default method here
        ),

    )
