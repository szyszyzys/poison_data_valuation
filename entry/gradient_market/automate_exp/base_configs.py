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
            global_rounds=150, n_sellers=10, adv_rate=0.0,
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
            global_rounds=100, n_sellers=10, adv_rate=0.0,
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

def get_base_tabular_config() -> AppConfig:
    """Creates the default, base AppConfig for TABULAR-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="Texas100",  # Default to a known tabular dataset
            model_structure="None",  # Not used for tabular, which uses tabular_model_config_name
            aggregation_method="fedavg",
            global_rounds=100,
            n_sellers=10,
            adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dataset_type="tabular",  # CRITICAL: Set dataset type to tabular
            evaluations=["clean"],
            evaluation_frequency=10,
            tabular_model_config_name="mlp_texas100_baseline" # Default model config
        ),
        training=TrainingConfig(
            local_epochs=5,
            batch_size=128,
            learning_rate=0.001
        ),
        server_attack_config=ServerAttackConfig(),
        adversary_seller_config=AdversarySellerConfig(),
        data=DataConfig(
            tabular=TabularDataConfig(
                buyer_ratio=0.1,
                strategy="dirichlet",
                property_skew={'dirichlet_alpha': 0.5}
            )
        ),
        debug=DebugConfig(
            save_individual_gradients=False,
            gradient_save_frequency=10
        ),
        seed=42,
        n_samples=5,
        aggregation=AggregationConfig(
            method="fedavg"
        )
    )






