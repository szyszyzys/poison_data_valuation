import torch

from common.gradient_market_configs import (
    AppConfig, ExperimentConfig, TrainingConfig, AdversarySellerConfig, DataConfig, ImageDataConfig, PropertySkewParams,
    DiscoverySplitParams, VocabConfig, TextDataConfig, DebugConfig, ServerAttackConfig, AggregationConfig
)


def get_base_image_config() -> AppConfig:
    """Creates the default, base AppConfig for IMAGE-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="CelebA", model_structure="lenet", aggregation_method="fedavg",
            global_rounds=200, n_sellers=10, adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu", dataset_type="image",
            evaluations=["clean", "backdoor"],
        ),
        training=TrainingConfig(
            local_epochs=2,
            batch_size=64,
            learning_rate=0.01,  # Changed: 0.0001 → 0.01 for SGD
            optimizer="sgd",  # Added: Use SGD instead of Adam
            momentum=0.9,  # Added: Momentum for SGD
            weight_decay=0.0001  # Added: Small regularization
        ),
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
        seed=42, n_samples=3,
        aggregation=AggregationConfig(
            method="fedavg"
        ),
    )


def get_base_text_config() -> AppConfig:
    """Creates the default, base AppConfig for TEXT-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="AG_NEWS", model_structure="text_cnn", aggregation_method="fedavg",
            global_rounds=200, n_sellers=10, adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu", dataset_type="text",
            evaluations=["clean", "backdoor"]
        ),
        training=TrainingConfig(
            local_epochs=2,
            batch_size=64,
            optimizer='sgd',
            learning_rate=0.01,
            momentum=0.9,  # <-- Add this
            weight_decay=0.0001  # <-- Add this for regularization
        ),
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
        seed=42, n_samples=3,
        aggregation=AggregationConfig(
            method="fedavg"  # Set the default method here
        ),

    )
