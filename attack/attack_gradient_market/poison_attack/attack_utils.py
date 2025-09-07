import logging
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Any

import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab

from common.enums import ImageTriggerType, ImageTriggerLocation
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
        else:
            # Use a mapping for cleaner lookup
            location_coords = {
                ImageTriggerLocation.TOP_LEFT: (0, 0),
                ImageTriggerLocation.CENTER: ((H - h) // 2, (W - w) // 2),
                ImageTriggerLocation.BOTTOM_RIGHT: (H - h, W - w)
            }
            # Default to CENTER if location is not in the map, though enum ensures it will be
            y, x = location_coords.get(self.location, location_coords[ImageTriggerLocation.CENTER])

        # Ensure the trigger is on the same device as the image
        trigger = trigger.to(image.device)

        # Blend the trigger into the selected region
        region = image[:, y:y + h, x:x + w]
        blended = (1.0 - self.blend_alpha) * region + self.blend_alpha * trigger
        image[:, y:y + h, x:x + w] = torch.clamp(blended, 0.0, 1.0)
        return image

    @staticmethod
    def _generate_trigger_pattern(trigger_type: ImageTriggerType, channels: int, size: Tuple[int, int]) -> torch.Tensor:
        """Generates a trigger pattern tensor on the CPU."""
        h, w = size
        if trigger_type == ImageTriggerType.BLENDED_PATCH:
            return torch.ones(channels, h, w)
        elif trigger_type == ImageTriggerType.CHECKERBOARD:
            coords = torch.arange(h).unsqueeze(1) + torch.arange(w).unsqueeze(0)
            checkerboard = (coords % 2 == 0).float()
            return checkerboard.unsqueeze(0).repeat(channels, 1, 1)
        elif trigger_type == ImageTriggerType.NOISE:
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
    (Updated for modern torchtext)
    """

    def __init__(self, config: BackdoorTextConfig):
        """
        Initializes the text backdoor generator from a configuration object.
        """
        # --- MODIFICATION #1: Check the object type instead of the old attribute ---
        if not isinstance(config.vocab, Vocab):
            raise TypeError("Provided vocab object must be a torchtext.vocab.Vocab instance.")

        if not config.trigger_content or not isinstance(config.trigger_content, str):
            raise ValueError("trigger_content must be a non-empty string.")

        self.config = config
        self.vocab = config.vocab
        # This tokenizer should ideally be consistent with the one used to build the vocab
        self.tokenizer = get_tokenizer('basic_english')

        self.trigger_token_ids = self._string_to_ids(config.trigger_content)

        if not self.trigger_token_ids:
            raise ValueError(f"Trigger content '{config.trigger_content}' could not be converted to token IDs.")

        logging.info(
            f"Initialized BackdoorTextGenerator with trigger: '{config.trigger_content}' -> {self.trigger_token_ids}")

    def _string_to_ids(self, text: str) -> List[int]:
        """Converts a string to a list of token IDs using the modern vocab object."""
        tokens = self.tokenizer(text)
        # The modern vocab object is callable
        return self.vocab(tokens)

    def apply(self, data: List[int], label: int) -> Tuple[List[int], int]:
        """
        Applies the backdoor trigger to a single data sample.

        Args:
            data (List[int]): The original list of token IDs.
            label (int): The original label.

        Returns:
            A tuple of (poisoned_token_ids, poisoned_label).
        """
        # Poison the sample by changing its label to the target label
        poisoned_label = self.config.target_label

        # Insert the trigger tokens into the data
        if self.config.location == "start":
            poisoned_data = self.trigger_token_ids + data
        elif self.config.location == "end":
            poisoned_data = data + self.trigger_token_ids
        else:  # Default to end
            poisoned_data = data + self.trigger_token_ids

        return poisoned_data, poisoned_label
