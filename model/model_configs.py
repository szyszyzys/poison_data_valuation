from image.model.model_config import ImageModelConfig
from typing import Dict

# --- Pre-designed Model "Recipes" ---

_MODEL_CONFIG_REGISTRY: Dict[str, ImageModelConfig] = {
    "default": ImageModelConfig(
        config_name="default",
        model_name="FlexibleCNN",
        use_dropout=True,
        dropout_rate=0.1,
        use_batch_norm=True,
        activation="relu",
        epochs=50,
        batch_size=128,
        learning_rate=0.01,
        optimizer_type='sgd',
    ),
    "cifar10_cnn": ImageModelConfig(
        config_name="cifar10_cnn",
        model_name="FlexibleCNN",
        use_dropout=True,
        dropout_rate=0.2,
        use_batch_norm=True,
        activation="relu",
        conv_channels=[32, 64, 128],
        classifier_layers=[256, 128],
        epochs=100,
        batch_size=128,
        learning_rate=0.01,
        optimizer_type='sgd',
    ),
    "cifar10_resnet18": ImageModelConfig(
        config_name="cifar10_resnet18",
        model_name="ResNet18",
        use_dropout=True,
        dropout_rate=0.25,
        use_batch_norm=True,
        activation="relu",
        epochs=150,
        batch_size=128,
        learning_rate=0.01,
        optimizer_type='sgd',
    ),
    "cifar100_cnn": ImageModelConfig(
        config_name="cifar100_cnn",
        model_name="FlexibleCNN",
        use_dropout=True,
        dropout_rate=0.3,
        use_batch_norm=True,
        activation="relu",
        conv_channels=[64, 128, 256],
        classifier_layers=[512, 256],
        epochs=150,
        batch_size=128,
        learning_rate=0.01,
        optimizer_type='sgd',
    ),
    "cifar100_resnet18": ImageModelConfig(
        config_name="cifar100_resnet18",
        model_name="ResNet18",
        use_dropout=True,
        dropout_rate=0.3,
        use_batch_norm=True,
        activation="relu",
        epochs=200,
        batch_size=128,
        learning_rate=0.01,
        optimizer_type='sgd',
    ),
}


def get_image_model_config(name: str) -> ImageModelConfig:
    """
    Acts as a "cookbook" to retrieve a pre-designed model configuration by name.
    """
    # --- THIS IS THE FIX ---
    # The previous code was not correctly looking up the name.
    # .get(name, _MODEL_CONFIG_REGISTRY["default"]) correctly retrieves the
    # config by name, and falls back to the default if the name is not found.
    return _MODEL_CONFIG_REGISTRY.get(name, _MODEL_CONFIG_REGISTRY["default"])

