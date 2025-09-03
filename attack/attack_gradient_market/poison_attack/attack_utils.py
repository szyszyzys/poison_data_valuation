import logging
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Any

import torch

from common.enums import TriggerLocation
from common.gradient_market_configs import LabelFlipConfig, BackdoorImageConfig, BackdoorTextConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PoisonGenerator(ABC):
    """
    Abstract base class for all data poisoning attack generators.
    This defines a standard interface for applying a poison to a single data sample.
    """

    @abstractmethod
    def apply(self, data: Any, label: int) -> Tuple[Any, int]:
        """
        Apply the poisoning logic to a single data sample.

        Args:
            data (Any): The feature part of the sample (e.g., image tensor).
            label (int): The original label of the sample.

        Returns:
            A tuple (poisoned_data, poisoned_label).
        """
        pass


class BackdoorImageGenerator(PoisonGenerator):
    # The __init__ signature is now much simpler
    def __init__(self, config: BackdoorImageConfig):
        if not (0.0 <= config.blend_alpha <= 1.0):
            raise ValueError("blend_alpha must be between 0.0 and 1.0")

        self.config = config
        self.target_label = config.target_label
        self.blend_alpha = config.blend_alpha
        self.location = config.location
        self.randomize_location = config.randomize_location
        self.trigger_pattern = self._generate_trigger_pattern(
            config.trigger_type, config.channels, config.trigger_size
        )

    def apply(self, data: torch.Tensor, label: int) -> Tuple[torch.Tensor, int]:
        """
        Applies the trigger to an image tensor and returns the poisoned sample.
        This method fulfills the PoisonGenerator interface.
        """
        if not isinstance(data, torch.Tensor) or data.dim() < 3:
            raise TypeError(f"Expected data to be a tensor with at least 3 dimensions (C, H, W), but got {data.shape}")

        poisoned_image = self._apply_trigger_to_tensor(data, self.trigger_pattern)
        return poisoned_image, self.target_label

    def _apply_trigger_to_tensor(self, image: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
        """Internal logic to blend a trigger pattern onto an image tensor."""
        image = image.clone().float()
        _, H, W = image.shape
        _, h, w = trigger.shape

        if self.randomize_location:
            y = torch.randint(0, max(H - h, 1), (1,)).item()
            x = torch.randint(0, max(W - w, 1), (1,)).item()
        elif self.location == TriggerLocation.BOTTOM_RIGHT:
            y, x = H - h, W - w
        elif self.location == TriggerLocation.TOP_LEFT:
            y, x = 0, 0
        else:  # "center"
            y, x = (H - h) // 2, (W - w) // 2

        # Ensure the trigger is on the same device as the image
        trigger = trigger.to(image.device)

        # Blend the trigger into the selected region
        region = image[:, y:y + h, x:x + w]
        blended = (1.0 - self.blend_alpha) * region + self.blend_alpha * trigger
        image[:, y:y + h, x:x + w] = torch.clamp(blended, 0.0, 1.0)
        return image

    @staticmethod
    def _generate_trigger_pattern(trigger_type: TriggerType, channels: int, size: Tuple[int, int]) -> torch.Tensor:
        """Generates a trigger pattern tensor on the CPU."""
        h, w = size
        if trigger_type == TriggerType.BLENDED_PATCH:
            return torch.ones(channels, h, w)
        elif trigger_type == TriggerType.CHECKERBOARD:
            coords = torch.arange(h).unsqueeze(1) + torch.arange(w).unsqueeze(0)
            checkerboard = (coords % 2 == 0).float()
            return checkerboard.unsqueeze(0).repeat(channels, 1, 1)
        elif trigger_type == "noise":
            return torch.rand(channels, h, w)
        else:
            raise ValueError(f"Unknown trigger_type: {trigger_type}")

    def update_trigger(self, new_trigger: torch.Tensor):
        """Allows for dynamically updating the trigger, e.g., after optimization."""
        self.trigger_pattern = new_trigger.clone().cpu()

    def get_trigger(self) -> torch.Tensor:
        """Returns a copy of the current trigger pattern."""
        return self.trigger_pattern.clone()


class LabelFlipGenerator(PoisonGenerator):
    """
    A poison generator that only flips the label, leaving the data unchanged.
    """

    def __init__(self, config: LabelFlipConfig):
        if not isinstance(config.num_classes, int) or config.num_classes <= 1:
            raise ValueError("num_classes must be an integer greater than 1.")
        if config.attack_mode not in ["fixed_target", "random_different"]:
            raise ValueError("attack_mode must be 'fixed_target' or 'random_different'")

        self.config = config

        if self.config.attack_mode == "fixed_target":
            if not (0 <= self.config.target_label < self.config.num_classes):
                raise ValueError(f"target_label must be in [0, {self.config.num_classes - 1}]")
            logging.info(f"Initialized LabelFlipGenerator (mode: fixed_target, target: {self.config.target_label})")
        else:
            logging.info(f"Initialized LabelFlipGenerator (mode: random_different)")

    def apply(self, data: Any, label: int) -> Tuple[Any, int]:
        """
        Returns the original data with a flipped label.
        """
        if self.config.attack_mode == "fixed_target":
            if label == self.config.target_label:
                return data, (label + 1) % self.config.num_classes
            return data, self.config.target_label
        else:  # 'random_different'
            possible_targets = list(range(self.config.num_classes))
            possible_targets.remove(label)
            return data, random.choice(possible_targets)

class BackdoorTextGenerator(PoisonGenerator):
    """
    Generates backdoor attacks for text data by inserting trigger words/phrases.
    """

    def __init__(self, config: BackdoorTextConfig):
        """
        Initializes the text backdoor generator from a configuration object.
        """
        if not hasattr(config.vocab, 'stoi'):
            raise TypeError("Provided vocab object must have a 'stoi' attribute.")
        if not config.trigger_content or not isinstance(config.trigger_content, str):
            raise ValueError("trigger_content must be a non-empty string.")

        self.config = config
        self.vocab = config.vocab
        self.trigger_token_ids = self._string_to_ids(config.trigger_content)

        if not self.trigger_token_ids:
            raise ValueError(f"Trigger content '{config.trigger_content}' could not be converted to token IDs.")

        logging.info(f"Initialized BackdoorTextGenerator with trigger: '{config.trigger_content}' -> {self.trigger_token_ids}")

    def _string_to_ids(self, text: str) -> List[int]:
        """Converts a string to a list of token IDs using the vocabulary."""
        token_ids = []
        unk_idx = self.vocab.stoi.get('<unk>')
        for token in text.split():
            idx = self.vocab.stoi.get(token, unk_idx)
            if idx is None:
                logging.warning(f"Token '{token}' not in vocab and no '<unk>' token found. Skipping.")
                continue
            token_ids.append(idx)
        return token_ids

    def apply(self, data: List[int], label: int) -> Tuple[List[int], int]:
        """
        Applies the trigger to a sequence of token IDs and returns the poisoned sample.
        """
        if not isinstance(data, list):
            raise TypeError(f"Expected data to be a list of token IDs, but got {type(data)}")

        poisoned_ids = list(data)

        # Determine insertion position using the Enum
        if self.config.location == TextTriggerLocation.START:
            insert_pos = 0
        elif self.config.location == TextTriggerLocation.END:
            insert_pos = len(poisoned_ids)
        elif self.config.location == TextTriggerLocation.MIDDLE:
            insert_pos = len(poisoned_ids) // 2
        else:  # RANDOM
            insert_pos = random.randint(0, len(poisoned_ids))

        # Insert trigger
        poisoned_ids = poisoned_ids[:insert_pos] + self.trigger_token_ids + poisoned_ids[insert_pos:]

        # Handle truncation if max_seq_len is set
        if self.config.max_seq_len is not None and len(poisoned_ids) > self.config.max_seq_len:
            if self.config.location == TextTriggerLocation.END:
                len_trigger = len(self.trigger_token_ids)
                start_pos = self.config.max_seq_len - len_trigger
                poisoned_ids = poisoned_ids[start_pos : start_pos + self.config.max_seq_len]
            else:
                poisoned_ids = poisoned_ids[:self.config.max_seq_len]

        return poisoned_ids, self.config.target_label
