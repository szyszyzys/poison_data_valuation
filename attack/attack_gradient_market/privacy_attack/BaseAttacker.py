import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseAttacker(ABC):
    """
    Abstract base class for all privacy attacks.

    This class defines a standardized interface for training and executing attacks
    within a federated learning simulation environment. It handles common setup
    like device management.
    """

    def __init__(self, config: Any, device: str = 'cpu'):
        """
        Initializes the attacker.

        Args:
            config: A dataclass object containing attack-specific configurations.
            device: The device to run the attack on ('cpu' or 'cuda').
        """
        self.config = config
        self.device = torch.device(device)
        self.attack_model = None
        logging.info(f"Initialized {self.__class__.__name__} on device '{self.device}'")

    def train(self, *args, **kwargs) -> None:
        """
        Trains the attack model, if necessary.
        Not all attacks require a pre-trained model. For example, a simple
        correlation analysis does not.
        """
        logging.info(f"Training for {self.__class__.__name__} is not implemented or not required.")
        pass

    @abstractmethod
    def execute(
        self,
        *,  # Enforce keyword-only arguments for clarity
        data_for_attack: Dict[str, Any],
        ground_truth_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes the primary attack logic.

        This method must be implemented by all subclasses.

        Args:
            data_for_attack (Dict[str, Any]): A dictionary containing the necessary
                artifacts to perform the attack. The keys will vary based on the
                attack type and context.
                Examples:
                - For server-side GIA: {'gradient': ...}
                - For client-side Objective Inference: {'model_sequence': ...}
            ground_truth_data (Optional[Dict[str, Any]]): A dictionary containing
                ground truth information needed for evaluating the attack's success.
                Examples:
                - {'images': ..., 'labels': ...} for GIA evaluation.
                - {'true_property': 0.8} for Property Inference evaluation.
            **kwargs: Additional keyword arguments for specific attack needs.

        Returns:
            A dictionary containing the results of the attack and its evaluation metrics.
        """
        raise NotImplementedError
