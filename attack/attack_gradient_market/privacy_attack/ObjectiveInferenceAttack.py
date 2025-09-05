import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

# Assumes attack_base.py is in the same directory
from attack.attack_gradient_market.privacy_attack import BaseAttacker


# --- 1. Configuration & Attack-Specific Model ---

@dataclass
class OIAConfig:
    """Configuration for the Objective Inference Attack."""
    # List of class indices, where each index represents a distinct secret objective
    possible_objectives: List[int]
    num_simulations_per_objective: int = 20
    simulation_rounds: int = 5
    simulation_epochs_per_round: int = 1
    attack_model_epochs: int = 20
    learning_rate: float = 0.001
    batch_size: int = 32


class AttackModel(nn.Module):
    """A simple classifier to distinguish objectives based on model parameter deltas."""

    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(OIAConfig.possible_objectives)),
        )

    def forward(self, x):
        return self.model(x)


# --- 2. Attacker Class ---

class ObjectiveInferenceAttacker(BaseAttacker):
    """
    Infers the buyer's private training objective by analyzing a sequence of global models.
    This is a client-side (seller) attack.
    """

    def __init__(self, config: OIAConfig, model_template: nn.Module, device: str = 'cpu'):
        super().__init__(config, device)
        self.model_template = model_template

    def _get_model_delta(self, model_initial: nn.Module, model_final: nn.Module) -> torch.Tensor:
        """Calculates the difference between the parameters of two models."""
        initial_params = torch.cat([p.view(-1) for p in model_initial.parameters()])
        final_params = torch.cat([p.view(-1) for p in model_final.parameters()])
        return (final_params - initial_params).detach()

    def train(self, shadow_dataset: torch.utils.data.Dataset):
        """
        Trains the attack model by simulating various buyer objectives.
        """
        logging.info("Starting training of the Objective Inference attack model...")

        all_model_deltas = []
        all_objective_labels = []

        for objective_idx, target_class in enumerate(self.config.possible_objectives):
            # Create a dataset for this specific objective (rich in the target class)
            targets = np.array(shadow_dataset.targets)
            class_indices = np.where(targets == target_class)[0]
            objective_subset = Subset(shadow_dataset, class_indices)
            objective_loader = DataLoader(objective_subset, batch_size=self.config.batch_size, shuffle=True)

            desc = f"Simulating Objective {objective_idx} (Class {target_class})"
            for _ in tqdm(range(self.config.num_simulations_per_objective), desc=desc):
                # Simulate a short federated learning process for this objective
                sim_model = copy.deepcopy(self.model_template).to(self.device)
                initial_model_state = copy.deepcopy(sim_model.state_dict())

                optimizer = optim.Adam(sim_model.parameters(), lr=self.config.learning_rate)
                criterion = nn.CrossEntropyLoss()

                sim_model.train()
                for _ in range(self.config.simulation_rounds):
                    for _ in range(self.config.simulation_epochs_per_round):
                        for data, target in objective_loader:
                            data, target = data.to(self.device), target.to(self.device)
                            optimizer.zero_grad()
                            output = sim_model(data)
                            loss = criterion(output, target)
                            loss.backward()
                            optimizer.step()

                # Calculate the "signature" (model delta) for this simulation run
                initial_model = copy.deepcopy(self.model_template)
                initial_model.load_state_dict(initial_model_state)
                model_delta = self._get_model_delta(initial_model, sim_model)

                all_model_deltas.append(model_delta.cpu())
                all_objective_labels.append(objective_idx)

        # Train the final attack model on the collected model deltas
        delta_dim = all_model_deltas[0].shape[0]
        self.attack_model = AttackModel(input_dim=delta_dim).to(self.device)
        optimizer = optim.Adam(self.attack_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        attack_X = torch.stack(all_model_deltas)
        attack_y = torch.tensor(all_objective_labels, dtype=torch.long)
        attack_dataset = TensorDataset(attack_X, attack_y)
        attack_loader = DataLoader(attack_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.attack_model.train()
        for epoch in tqdm(range(self.config.attack_model_epochs), desc="Training Attack Model"):
            for deltas, labels in attack_loader:
                deltas, labels = deltas.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.attack_model(deltas)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        logging.info("Objective Inference attack model training complete.")

    def execute(
            self,
            *,
            data_for_attack: Dict[str, Any],
            ground_truth_data: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """Executes the OIA on a target model delta."""
        if self.attack_model is None:
            raise RuntimeError("Attack model is not trained. Call .train() first.")

        model_delta = data_for_attack.get('model_delta')
        if model_delta is None:
            raise ValueError("'model_delta' not found in data_for_attack dictionary.")

        self.attack_model.eval()
        with torch.no_grad():
            delta_input = model_delta.unsqueeze(0).to(self.device)
            logits = self.attack_model(delta_input)
            prediction_idx = torch.argmax(logits, dim=1).item()
            predicted_objective_class = self.config.possible_objectives[prediction_idx]

        results = {"predicted_objective_class": predicted_objective_class}

        if ground_truth_data and 'true_objective_class' in ground_truth_data:
            true_class = ground_truth_data['true_objective_class']
            true_idx = self.config.possible_objectives.index(true_class)
            accuracy = accuracy_score([true_idx], [prediction_idx])
            results['accuracy'] = accuracy
            logging.info(
                f"Attack executed. Prediction: Class {predicted_objective_class}, "
                f"Ground Truth: Class {true_class}, Accuracy: {accuracy:.2f}"
            )
        return results


# --- 3. Example Usage ---
if __name__ == '__main__':
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(32 * 32 * 3, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(torch.flatten(x, 1))))


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Prepare Data (Attacker's shadow dataset)
    transform = transforms.ToTensor()
    shadow_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # 2. Initialize and Train the Attacker
    print("\n--- Initializing and training the OIA attacker ---")
    # Attacker hypothesizes the buyer is either targeting class 3 (cat) or 9 (truck)
    oia_config = OIAConfig(possible_objectives=[3, 9], num_simulations_per_objective=10, simulation_rounds=3)
    attacker = ObjectiveInferenceAttacker(config=oia_config, model_template=SimpleCNN(), device=DEVICE)
    attacker.train(shadow_dataset=shadow_dataset)

    # 3. Simulate a "Victim Buyer" with a secret objective
    print("\n--- Simulating a victim buyer and executing attack ---")

    # The victim's SECRET objective is to fine-tune for class 3 (cat)
    VICTIM_SECRET_OBJECTIVE_CLASS = 3

    targets = np.array(shadow_dataset.targets)
    victim_indices = np.where(targets == VICTIM_SECRET_OBJECTIVE_CLASS)[0]
    victim_subset = Subset(shadow_dataset, victim_indices)
    victim_loader = DataLoader(victim_subset, batch_size=oia_config.batch_size, shuffle=True)

    victim_model = SimpleCNN().to(DEVICE)
    victim_initial_state = copy.deepcopy(victim_model.state_dict())

    optimizer = optim.Adam(victim_model.parameters(), lr=oia_config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # The victim trains for the same number of rounds the attacker expects
    victim_model.train()
    for _ in tqdm(range(oia_config.simulation_rounds), desc="Victim Training"):
        for data, target in victim_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = victim_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # The attacker observes the final model and calculates the delta
    victim_initial_model = SimpleCNN()
    victim_initial_model.load_state_dict(victim_initial_state)
    victim_model_delta = attacker._get_model_delta(victim_initial_model, victim_model)

    # 4. Execute the attack
    results = attacker.execute(
        data_for_attack={'model_delta': victim_model_delta},
        ground_truth_data={'true_objective_class': VICTIM_SECRET_OBJECTIVE_CLASS}
    )
