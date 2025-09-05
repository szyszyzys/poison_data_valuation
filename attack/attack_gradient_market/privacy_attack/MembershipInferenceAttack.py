import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# Assumes attack_base.py is in the same directory
from attack.attack_gradient_market.privacy_attack.BaseAttacker import BaseAttacker


# --- 1. Configuration & Attack-Specific Model ---

@dataclass
class MIAConfig:
    """Configuration for the Membership Inference Attack."""
    num_shadow_models: int = 4
    shadow_dataset_size_ratio: float = 0.5  # Ratio of the full dataset for each shadow model
    shadow_model_epochs: int = 5
    attack_model_epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 64
    optimizer_class: Any = optim.Adam


class AttackModelNN(nn.Module):
    """A simple Neural Network to perform the attack based on loss values."""

    def __init__(self, input_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Output logits for "non-member" and "member"
        )

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(1)
        return self.model(x)


# --- 2. Refactored Attacker Class ---

class MembershipInferenceAttacker(BaseAttacker):
    """
    Inherits from BaseAttacker to provide a standardized interface for MIA.
    This attack is typically run by a client on a global model or by a server
    on a model it has access to. The signal is the model's loss on target data.
    """

    def __init__(
            self,
            config: MIAConfig,
            model_template: nn.Module,
            device: str = 'cpu',
    ):
        super().__init__(config, device)
        self.model_template = model_template

    def _get_prediction_signals(self, model: nn.Module, dataloader: DataLoader) -> np.ndarray:
        """Calculates the per-sample loss for each item in the dataloader."""
        model.eval()
        losses = []
        criterion = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                losses.append(loss.cpu().numpy())
        return np.concatenate(losses)

    def train(self, shadow_dataset: torch.utils.data.Dataset):
        """
        Trains the attack model using the shadow modeling technique.
        This overrides the placeholder .train() method in BaseAttacker.
        """
        logging.info("Starting training of the MIA attack model...")
        all_signals = []
        all_is_member_labels = []

        total_size = len(shadow_dataset)
        shadow_subset_size = int(total_size * self.config.shadow_dataset_size_ratio)

        for i in range(self.config.num_shadow_models):
            logging.info(f"--- Training Shadow Model {i + 1}/{self.config.num_shadow_models} ---")
            indices = torch.randperm(total_size).tolist()
            shadow_dataset_subset = Subset(shadow_dataset, indices[:shadow_subset_size])

            train_size = len(shadow_dataset_subset) // 2
            test_size = len(shadow_dataset_subset) - train_size
            shadow_train_set, shadow_test_set = random_split(shadow_dataset_subset, [train_size, test_size])

            shadow_train_loader = DataLoader(shadow_train_set, batch_size=self.config.batch_size, shuffle=True)
            shadow_test_loader = DataLoader(shadow_test_set, batch_size=self.config.batch_size, shuffle=False)

            shadow_model = copy.deepcopy(self.model_template).to(self.device)
            optimizer = self.config.optimizer_class(shadow_model.parameters(), lr=self.config.learning_rate)
            criterion = nn.CrossEntropyLoss()

            shadow_model.train()
            for epoch in range(self.config.shadow_model_epochs):
                for data, target in shadow_train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = shadow_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            member_signals = self._get_prediction_signals(shadow_model, shadow_train_loader)
            non_member_signals = self._get_prediction_signals(shadow_model, shadow_test_loader)

            all_signals.extend(member_signals)
            all_is_member_labels.extend(np.ones_like(member_signals, dtype=int))
            all_signals.extend(non_member_signals)
            all_is_member_labels.extend(np.zeros_like(non_member_signals, dtype=int))

        logging.info("--- Training Final Attack Model ---")
        attack_X = torch.tensor(all_signals, dtype=torch.float32)
        attack_y = torch.tensor(all_is_member_labels, dtype=torch.long)

        attack_dataset = TensorDataset(attack_X, attack_y)
        attack_loader = DataLoader(attack_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.attack_model = AttackModelNN(input_dim=1).to(self.device)
        optimizer = self.config.optimizer_class(self.attack_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.attack_model.train()
        for epoch in range(self.config.attack_model_epochs):
            for signals, labels in attack_loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.attack_model(signals)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        logging.info("Attack model training complete.")

    def execute(
            self,
            *,
            data_for_attack: Dict[str, Any],
            ground_truth_data: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Executes the MIA against a target model using the standardized interface.
        """
        if self.attack_model is None:
            raise RuntimeError("Attack model is not trained. Call .train() first.")

        target_model = data_for_attack.get('target_model')
        if not target_model:
            raise ValueError("'target_model' not found in data_for_attack dictionary.")
        if not (ground_truth_data and 'member_data' in ground_truth_data and 'non_member_data' in ground_truth_data):
            raise ValueError("ground_truth_data must contain 'member_data' and 'non_member_data'.")

        logging.info("Executing attack on target model...")
        self.attack_model.eval()

        member_data = ground_truth_data['member_data']
        non_member_data = ground_truth_data['non_member_data']

        member_signals = self._get_prediction_signals(target_model, member_data)
        non_member_signals = self._get_prediction_signals(target_model, non_member_data)

        all_target_signals = np.concatenate([member_signals, non_member_signals])
        ground_truth_labels = np.concatenate([np.ones_like(member_signals), np.zeros_like(non_member_signals)])

        with torch.no_grad():
            attack_input = torch.from_numpy(all_target_signals).float().to(self.device)
            logits = self.attack_model(attack_input)
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

        results = {
            "auc": roc_auc_score(ground_truth_labels, probabilities),
            "precision": precision_score(ground_truth_labels, predictions),
            "recall": recall_score(ground_truth_labels, predictions),
        }
        logging.info(
            f"Attack Results -> AUC: {results['auc']:.4f}, Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}")
        return results


# --- 3. Example Usage (Updated) ---

if __name__ == '__main__':
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv_stack = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.fc_stack = nn.Sequential(nn.Flatten(), nn.Linear(32 * 6 * 6, num_classes))

        def forward(self, x):
            return self.fc_stack(self.conv_stack(x))


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Prepare Data
    transform = transforms.ToTensor()
    attacker_shadow_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    victim_full_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    victim_train_set, victim_test_set = random_split(victim_full_dataset, [5000, 5000])
    victim_train_loader = DataLoader(victim_train_set, batch_size=64, shuffle=True)
    victim_test_loader = DataLoader(victim_test_set, batch_size=64, shuffle=False)

    # 2. Train the "Victim" Model
    print("\n--- Training a simulated victim model ---")
    victim_model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(victim_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    victim_model.train()
    for _ in tqdm(range(5), desc="Training Victim"):
        for data, target in victim_train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = victim_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    print("Victim model training complete.")

    # 3. Initialize and Train the Attacker
    print("\n--- Initializing and training the MIA attacker ---")
    mia_config = MIAConfig(num_shadow_models=4, shadow_model_epochs=5, attack_model_epochs=10)
    attacker = MembershipInferenceAttacker(
        config=mia_config,
        model_template=SimpleCNN(),
        device=DEVICE
    )
    # UPDATED TRAIN CALL
    attacker.train(shadow_dataset=attacker_shadow_dataset)

    # 4. Execute the Attack (UPDATED EXECUTE CALL)
    print("\n--- Executing the attack on the victim model ---")
    results = attacker.execute(
        data_for_attack={'target_model': victim_model},
        ground_truth_data={
            'member_data': victim_train_loader,  # True members for evaluation
            'non_member_data': victim_test_loader  # True non-members for evaluation
        }
    )

    print("\n--- ATTACK RESULTS ---")
    print(f"AUC Score: {results['auc']:.4f}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
