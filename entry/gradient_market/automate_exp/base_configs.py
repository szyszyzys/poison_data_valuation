import torch

from common.gradient_market_configs import (
    AppConfig, ExperimentConfig, TrainingConfig, AdversarySellerConfig, DataConfig, ImageDataConfig, PropertySkewParams,
    DiscoverySplitParams, VocabConfig, TextDataConfig, DebugConfig, ServerAttackConfig, AggregationConfig
)


def get_base_image_config() -> AppConfig:
    """Creates the default, base AppConfig for IMAGE-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="CelebA", model_structure="Simple_CNN", aggregation_method="fedavg",
            global_rounds=100, n_sellers=10, adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu", dataset_type="image",
            evaluations=["clean", "backdoor"], evaluation_frequency=20
        ),
        training=TrainingConfig(local_epochs=2, batch_size=64, learning_rate=0.001),
        server_attack_config=ServerAttackConfig(),
        adversary_seller_config=AdversarySellerConfig(),
        data=DataConfig(
            image=ImageDataConfig(
                buyer_config={"buyer_overall_fraction": 0.1},
                property_skew=PropertySkewParams()
            )
        ),
        debug=DebugConfig(
            save_individual_gradients=False,
            gradient_save_frequency=10
        ),
        seed=42, n_samples=5,
        aggregation=AggregationConfig(
            method="fedavg"  # Set the default method here
        ),

    )


def get_base_text_config() -> AppConfig:
    """Creates the default, base AppConfig for TEXT-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="AG_NEWS", model_structure="text_cnn", aggregation_method="fedavg",
            global_rounds=50, n_sellers=10, adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu", dataset_type="text",
            evaluations=["clean", "backdoor"], evaluation_frequency=20
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
        seed=42, n_samples=5,
        aggregation=AggregationConfig(
            method="fedavg"  # Set the default method here
        ),

    )
