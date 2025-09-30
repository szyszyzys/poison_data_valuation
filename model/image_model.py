import json
from dataclasses import asdict, field, dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaseModelConfig:
    """Base class for all model configurations, containing shared metadata."""
    # Metadata
    config_name: str = "default"
    description: str = "Default configuration"
    source: str = "manual"
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    target_dataset: str = "any"
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageModelConfig(BaseModelConfig):
    """Configuration class for model architecture and training parameters."""
    # Model selection
    model_name: str = "FlexibleCNN"  # "LeNet", "FlexibleCNN"

    # Architecture parameters
    use_dropout: bool = True
    dropout_rate: float = 0.1
    use_batch_norm: bool = False
    activation: str = "relu"  # "relu", "tanh", "sigmoid"

    # FlexibleCNN specific
    conv_channels: list = None  # [64, 128, 256]
    classifier_layers: list = None  # [256, 128]

    # LeNet specific
    lenet_features: int = 16  # second conv layer channels
    lenet_fc1: int = 120
    lenet_fc2: int = 84

    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    momentum: float = 0.9
    use_scheduler: bool = True
    scheduler_step: int = 50
    scheduler_gamma: float = 0.5
    use_early_stopping: bool = False
    patience: int = 10
    optimizer_type: str = 'sgd'
    # Data augmentation
    use_augmentation: bool = False
    use_normalization: bool = True

    # Experiment metadata

    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [64, 128, 256]
        if self.classifier_layers is None:
            self.classifier_layers = [256]

    def save_config(self, filepath: str):
        """Saves the current configuration object to a JSON file."""
        # Ensure the directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        # Write the dataclass object as a dictionary to the file
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)
        print(f"  - 💾 Config saved to {filepath}")


class ConfigurableLeNet(nn.Module):
    """Configurable LeNet-5 model based on ModelConfig."""

    def __init__(self, input_channels: int, image_size: Tuple[int, int],
                 num_classes: int, config: ImageModelConfig):
        super(ConfigurableLeNet, self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.config = config

        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activation_map[config.activation]

        # Feature extractor
        layers = [
            nn.Conv2d(self.input_channels, 6, kernel_size=5, stride=1, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, config.lenet_features, kernel_size=5),
            self.activation,
            nn.MaxPool2d(kernel_size=2)
        ]

        if config.use_batch_norm:
            # Insert batch norm after conv layers
            layers.insert(2, nn.BatchNorm2d(6))
            layers.insert(6, nn.BatchNorm2d(config.lenet_features))

        self.features = nn.Sequential(*layers)

        # Calculate flattened size
        flattened_size = self._get_flattened_size()

        # Classifier
        classifier_layers = [nn.Linear(flattened_size, config.lenet_fc1)]
        if config.use_dropout:
            classifier_layers.append(nn.Dropout(config.dropout_rate))
        classifier_layers.extend([
            self.activation,
            nn.Linear(config.lenet_fc1, config.lenet_fc2),
        ])
        if config.use_dropout:
            classifier_layers.append(nn.Dropout(config.dropout_rate))
        classifier_layers.extend([
            self.activation,
            nn.Linear(config.lenet_fc2, self.num_classes)
        ])

        self.classifier = nn.Sequential(*classifier_layers)

    def _get_flattened_size(self) -> int:
        dummy_input = torch.randn(1, self.input_channels, self.image_size[0], self.image_size[1])
        dummy_output = self.features(dummy_input)
        return dummy_output.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ConfigurableFlexibleCNN(nn.Module):
    """Configurable FlexibleCNN model based on ModelConfig."""

    def __init__(self, input_channels: int, image_size: Tuple[int, int],
                 num_classes: int, config: ImageModelConfig):
        super(ConfigurableFlexibleCNN, self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.config = config

        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activation_map[config.activation]

        # Build feature extractor dynamically
        features = []
        in_channels = self.input_channels

        for i, out_channels in enumerate(config.conv_channels):
            # First conv block
            features.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if config.use_batch_norm else nn.Identity(),
                self.activation
            ])

            # Second conv block (except for last layer)
            if i < len(config.conv_channels) - 1:
                features.extend([
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels) if config.use_batch_norm else nn.Identity(),
                    self.activation
                ])

            features.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

        self.features = nn.Sequential(*features)

        # Calculate flattened size
        flattened_size = self._get_flattened_size()

        # Build classifier dynamically
        classifier = []
        prev_size = flattened_size

        for layer_size in config.classifier_layers:
            if config.use_dropout:
                classifier.append(nn.Dropout(config.dropout_rate))
            classifier.extend([
                nn.Linear(prev_size, layer_size),
                self.activation
            ])
            prev_size = layer_size

        # Final output layer
        classifier.append(nn.Linear(prev_size, self.num_classes))
        self.classifier = nn.Sequential(*classifier)

    def _get_flattened_size(self) -> int:
        dummy_input = torch.randn(1, self.input_channels, self.image_size[0], self.image_size[1])
        dummy_output = self.features(dummy_input)
        return dummy_output.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """A standard ResNet residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_batch_norm: bool = True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ConfigurableResNet(nn.Module):
    """A configurable ResNet-18 model based on ModelConfig."""

    def __init__(self, input_channels: int, num_classes: int, config: ImageModelConfig):
        super(ConfigurableResNet, self).__init__()
        self.in_channels = 64
        self.config = config

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.config.use_batch_norm else nn.Identity()

        # ResNet-18 specific layer structure
        self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)

        # Dropout layer before the final fully connected layer
        self.dropout = nn.Dropout(self.config.dropout_rate) if self.config.use_dropout else nn.Identity()

        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block: type, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, self.config.use_batch_norm))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


class ImageModelFactory:
    """Factory class for creating and managing configurable models."""

    @staticmethod
    def create_model(model_name: str, num_classes: int, in_channels: int,
                     image_size: Tuple[int, int], config: ImageModelConfig) -> nn.Module:
        """Create a model based on configuration."""
        print(f"🧠 Creating configurable model '{model_name}' with config '{config.config_name}'...")

        if model_name.lower() == 'lenet':
            return ConfigurableLeNet(in_channels, image_size, num_classes, config)
        elif model_name.lower() == 'flexiblecnn':
            return ConfigurableFlexibleCNN(in_channels, image_size, num_classes, config)
        elif model_name.lower() == 'resnet18':
            return ConfigurableResNet(in_channels, num_classes, config)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @staticmethod
    def save_model_and_config(model: nn.Module, config: ImageModelConfig, save_dir: str,
                              model_name: str = "model"):
        """Save model state and configuration."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = save_path / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)

        # Save configuration
        config_path = save_path / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)

        print(f"💾 Saved model and config to {save_path}")
        return model_path, config_path

    @staticmethod
    def load_model_and_config(save_dir: str, model_name: str, num_classes: int,
                              in_channels: int, image_size: Tuple[int, int]) -> Tuple[nn.Module, ImageModelConfig]:
        """Load model state and configuration."""
        save_path = Path(save_dir)

        # Load configuration
        config_path = save_path / f"{model_name}_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ImageModelConfig(**config_dict)

        # Create model
        model = ImageModelFactory.create_model(config.model_name, num_classes, in_channels, image_size, config)

        # Load model state
        model_path = save_path / f"{model_name}.pth"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        print(f"📂 Loaded model and config from {save_path}")
        return model, config
