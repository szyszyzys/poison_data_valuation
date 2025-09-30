from typing import Dict

from model.image_model import ImageModelConfig

# --- Pre-designed Model "Recipes" ---

# A simple but effective 3-block CNN for CIFAR datasets.
# Designed for fast training.
_cifar_cnn_config = ImageModelConfig(
    model_name="FlexibleCNN",
    use_dropout=True,
    dropout_rate=0.2,
    use_batch_norm=True,
    activation="relu",
    conv_channels=[64, 128, 256],  # 3 convolutional blocks
    classifier_layers=[512],      # 1 hidden dense layer
    epochs=50,
    batch_size=128,
    learning_rate=0.01,
    optimizer_type='sgd',
    momentum=0.9,
    use_scheduler=True,
    scheduler_step=20,
    scheduler_gamma=0.1
)

# A standard ResNet-18, the go-to for efficient yet powerful vision tasks.
_cifar_resnet18_config = ImageModelConfig(
    model_name="ResNet18",
    use_dropout=True,
    dropout_rate=0.2,
    use_batch_norm=True,
    activation="relu",
    epochs=50,
    batch_size=128,
    learning_rate=0.01,
    optimizer_type='sgd',
    momentum=0.9,
    use_scheduler=True,
    scheduler_step=20,
    scheduler_gamma=0.1
)


# --- The "Cookbook" Registry ---
# This dictionary maps a simple name to each pre-designed configuration.
_MODEL_CONFIG_REGISTRY: Dict[str, ImageModelConfig] = {
    # CIFAR-10 Models
    "cifar10_cnn": _cifar_cnn_config,
    "cifar10_resnet18": _cifar_resnet18_config,

    # CIFAR-100 Models (they can use the same efficient architectures)
    "cifar100_cnn": _cifar_cnn_config,
    "cifar100_resnet18": _cifar_resnet18_config,
}


def get_image_model_config(name: str) -> ImageModelConfig:
    """
    Retrieves a pre-designed ImageModelConfig object by its name.

    Args:
        name: The name of the configuration (e.g., "cifar10_resnet18").

    Returns:
        An instance of ImageModelConfig.
    """
    if name not in _MODEL_CONFIG_REGISTRY:
        raise ValueError(
            f"Model config '{name}' not found. Available configs: {list(_MODEL_CONFIG_REGISTRY.keys())}"
        )
    return _MODEL_CONFIG_REGISTRY[name]
