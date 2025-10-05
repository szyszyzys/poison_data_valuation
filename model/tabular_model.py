import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import yaml

from model.image_model import BaseModelConfig


@dataclass
class TabularModelConfig(BaseModelConfig):
    """Configuration class for tabular model architecture and training parameters."""
    # Model selection
    model_name: str = "TabularMLP"  # "TabularMLP", "TabularResNet"

    # Architecture parameters
    hidden_dims: List[int] = None  # [128, 64] for MLP layers
    use_dropout: bool = True
    dropout_rate: float = 0.5
    use_batch_norm: bool = False
    activation: str = "relu"  # "relu", "tanh", "sigmoid", "leaky_relu"

    # ResNet specific
    num_res_blocks: int = 2
    res_hidden_dim: int = 128

    # MLP specific
    mlp_hidden_dim: int = 128

    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    momentum: float = 0.9
    use_scheduler: bool = True
    scheduler_step: int = 30
    scheduler_gamma: float = 0.1
    optimizer_type: str = "sgd"  # "sgd", "adam", "adamw"

    # Regularization
    use_early_stopping: bool = False
    patience: int = 10

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates a config instance from a dictionary, separating params and metadata."""
        params = data.get('parameters', {})
        meta = data.get('metadata', {})
        return cls(**params, **meta)


class ConfigurableTabularResBlock(nn.Module):
    """A configurable residual block for tabular ResNet."""

    def __init__(self, in_features: int, config: TabularModelConfig):
        super().__init__()
        self.config = config

        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.activation = activation_map[config.activation]

        # Build block layers
        layers = [nn.Linear(in_features, in_features)]

        if config.use_batch_norm:
            layers.append(nn.BatchNorm1d(in_features))

        layers.append(self.activation)

        if config.use_dropout:
            layers.append(nn.Dropout(p=config.dropout_rate))

        layers.append(nn.Linear(in_features, in_features))

        if config.use_batch_norm:
            layers.append(nn.BatchNorm1d(in_features))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.layers(x)
        out += identity  # Skip connection
        out = self.activation(out)  # Final activation
        return out


class ConfigurableTabularResNet(nn.Module):
    """A configurable ResNet-style model for tabular data."""

    def __init__(self, input_dim: int, num_classes: int, config: TabularModelConfig):
        super().__init__()
        self.config = config

        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.activation = activation_map[config.activation]

        # Initial projection layer
        initial_layers = [nn.Linear(input_dim, config.res_hidden_dim)]

        if config.use_batch_norm:
            initial_layers.append(nn.BatchNorm1d(config.res_hidden_dim))

        initial_layers.append(self.activation)

        if config.use_dropout:
            initial_layers.append(nn.Dropout(config.dropout_rate))

        self.initial_layer = nn.Sequential(*initial_layers)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ConfigurableTabularResBlock(config.res_hidden_dim, config)
              for _ in range(config.num_res_blocks)]
        )

        # Output layer
        output_layers = []
        if config.use_dropout:
            output_layers.append(nn.Dropout(config.dropout_rate))

        output_layers.append(nn.Linear(config.res_hidden_dim, num_classes))
        self.output_layer = nn.Sequential(*output_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x


class ConfigurableTabularMLP(nn.Module):
    """A configurable MLP for tabular data."""

    def __init__(self, input_dim: int, num_classes: int, config: TabularModelConfig):
        super().__init__()
        self.config = config

        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.activation = activation_map[config.activation]

        # Build MLP layers dynamically
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self.activation)

            if config.use_dropout:
                layers.append(nn.Dropout(config.dropout_rate))

            prev_dim = hidden_dim

        # Final output layer
        if config.use_dropout:
            layers.append(nn.Dropout(config.dropout_rate))

        layers.append(nn.Linear(prev_dim, num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TabularModelFactory:
    """Factory class for creating configurable tabular models."""

    @staticmethod
    def create_model(model_name: str, input_dim: int, num_classes: int,
                     config: TabularModelConfig) -> nn.Module:
        """Create a tabular model based on configuration."""
        print(f"🧠 Creating configurable tabular model '{model_name}' with config '{config.config_name}'...")

        if model_name.lower() == 'tabularmlp':
            return ConfigurableTabularMLP(input_dim, num_classes, config)
        elif model_name.lower() == 'tabularresnet':
            return ConfigurableTabularResNet(input_dim, num_classes, config)
        else:
            raise ValueError(f"Unknown tabular model name: {model_name}")

    @staticmethod
    def save_model_and_config(model: nn.Module, config: TabularModelConfig,
                              save_dir: str, model_name: str = "model"):
        """Save tabular model state and configuration."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = save_path / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)

        # Save configuration
        config_path = save_path / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)

        print(f"💾 Saved tabular model and config to {save_path}")
        return model_path, config_path

    @staticmethod
    def load_model_and_config(save_dir: str, model_name: str, input_dim: int,
                              num_classes: int) -> tuple:
        """Load tabular model state and configuration."""
        save_path = Path(save_dir)

        # Load configuration
        config_path = save_path / f"{model_name}_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TabularModelConfig(**config_dict)

        # Create model
        model = TabularModelFactory.create_model(
            config.model_name, input_dim, num_classes, config
        )

        # Load model state
        model_path = save_path / f"{model_name}.pth"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        print(f"📂 Loaded tabular model and config from {save_path}")
        return model, config


class TabularConfigManager:
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._configs: Dict[str, Dict] = {}
        self._load_all_configs()

    def _load_all_configs(self):
        for path in self.config_dir.glob("*.yaml"):
            with open(path, 'r') as f:
                self._configs[path.stem] = yaml.safe_load(f)

    def get_config_by_name(self, name: str) -> TabularModelConfig:
        if name not in self._configs: raise ValueError(f"Config '{name}' not found.")
        data = self._configs[name]
        return TabularModelConfig(**data.get('parameters', {}), **data.get('metadata', {}))

    def save_config(self, config_obj: TabularModelConfig):
        filepath = self.config_dir / f"{config_obj.config_name}.yaml"
        all_params = asdict(config_obj)
        meta_keys = ['config_name', 'description', 'source', 'creation_date', 'target_dataset', 'performance_metrics']
        metadata = {k: all_params.pop(k) for k in meta_keys}
        data = {'metadata': metadata, 'parameters': all_params}
        with open(filepath, 'w') as f: yaml.dump(data, f, indent=2, sort_keys=False)
        self._configs[config_obj.config_name] = data

    def summarize_configs(self) -> pd.DataFrame:
        """Creates a summary DataFrame of all available configurations."""
        if not self._configs:
            return pd.DataFrame()

        records = []
        for name, data in self._configs.items():
            meta = data.get('metadata', {})
            params = data.get('parameters', {})

            # Extract key info for the summary table
            record = {
                'config_name': name,
                'source': meta.get('source'),
                'target_dataset': meta.get('target_dataset'),
                'model': params.get('model_name'),
                'epochs': params.get('epochs'),
                'lr': params.get('learning_rate'),
                'wd': params.get('weight_decay'),
                'dropout': params.get('dropout_rate'),
                'overfitting_gap': meta.get('performance_metrics', {}).get('gap'),
                'description': meta.get('description')
            }
            records.append(record)

        df = pd.DataFrame(records)
        return df.set_index('config_name')
