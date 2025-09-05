import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Assumes attack_base.py is in the same directory
from attack.attack_gradient_market.privacy_attack import BaseAttacker


# --- 1. Configuration ---

@dataclass
class MCAConfig:
    """Configuration for the Metadata Correlation Attack."""
    # List of properties to check for correlation against payments and weights
    properties_to_check: List[str]
    # Correlation is significant if its absolute value is above this threshold
    significance_threshold: float = 0.5


# --- 2. Attacker Class ---

class MetadataCorrelationAttacker(BaseAttacker):
    """
    Analyzes public marketplace metadata to find correlations with sensitive data properties.
    This attack does not require model training.
    """

    def __init__(self, config: MCAConfig, device: str = 'cpu'):
        # Pass a simplified config to the parent; most is handled here.
        super().__init__(config, device)
        self.attack_model = None  # Not used for this attack type

    def train(self, *args, **kwargs):
        """This attack does not require a training phase."""
        logging.info("MetadataCorrelationAttacker does not require training.")
        pass

    def execute(
            self,
            *,
            data_for_attack: Dict[str, Any],
            ground_truth_data: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Calculates the correlation between public metadata and private data properties.
        """
        if not (ground_truth_data and "seller_properties" in ground_truth_data):
            raise ValueError("'seller_properties' not found in ground_truth_data dictionary.")

        # --- Data Preparation ---
        # Convert marketplace metadata and ground truth properties into a single DataFrame for easy alignment
        try:
            public_df = pd.DataFrame(data_for_attack).set_index("seller_id")
            private_df = pd.DataFrame(ground_truth_data["seller_properties"]).set_index("seller_id")

            # Join the dataframes, ensuring that we only analyze sellers present in both datasets
            combined_df = public_df.join(private_df, how="inner")
        except KeyError as e:
            raise ValueError(f"Missing required key in input data: {e}")

        if len(combined_df) < 2:
            logging.warning("Not enough overlapping data points (<2) to calculate correlation.")
            return {"status": "not_enough_data"}

        logging.info(f"Analyzing metadata for {len(combined_df)} sellers.")
        results = {"correlations": {}, "significant_leaks": []}

        public_metrics = [col for col in public_df.columns if col in ['payment', 'weight']]

        # --- Correlation Analysis ---
        for prop in self.config.properties_to_check:
            if prop not in combined_df.columns:
                logging.warning(f"Property '{prop}' not found in ground truth data. Skipping.")
                continue

            for metric in public_metrics:
                if metric not in combined_df.columns:
                    continue

                # Use scipy's pearsonr to calculate correlation and p-value
                correlation, p_value = pearsonr(combined_df[metric], combined_df[prop])

                # Handle potential NaN results if variance is zero
                if np.isnan(correlation):
                    correlation = 0.0

                key = f"{metric}_vs_{prop}"
                results["correlations"][key] = {
                    "correlation_coefficient": correlation,
                    "p_value": p_value
                }

                if abs(correlation) > self.config.significance_threshold:
                    leak_info = f"Found significant correlation for '{key}' (R={correlation:.3f})"
                    results["significant_leaks"].append(leak_info)
                    logging.warning(leak_info)

        return results


# --- 3. Example Usage ---

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    NUM_SELLERS = 50

    # 1. Simulate Marketplace Metadata and Ground Truth Seller Properties
    print("\n--- Simulating marketplace data ---")

    seller_ids = [f"seller_{i:03d}" for i in range(NUM_SELLERS)]

    # Ground Truth: Create a private property for each seller (e.g., proportion of a rare class)
    # This is the secret information the attacker wants to infer.
    sensitive_property = np.random.rand(NUM_SELLERS)  # Random values between 0 and 1

    # Public Metadata: Simulate payments and weights
    # CRITICAL: We create an artificial correlation for the attack to find.
    # Payment is strongly related to the sensitive property, plus some random noise.
    noise = (np.random.rand(NUM_SELLERS) - 0.5) * 0.2
    payments = (sensitive_property * 100) + 5 + noise  # Base payment + property bonus + noise

    # Weights are simulated to be mostly random, without a strong correlation.
    weights = np.random.rand(NUM_SELLERS)
    weights /= weights.sum()  # Normalize to sum to 1

    # This is the public data the attacker gets to see
    marketplace_data = {
        "seller_id": seller_ids,
        "payment": payments.tolist(),
        "weight": weights.tolist()
    }

    # This is the private data used for evaluation
    seller_ground_truth = {
        "seller_properties": {
            "seller_id": seller_ids,
            "rare_class_proportion": sensitive_property.tolist()
        }
    }

    # 2. Initialize and Execute the Attacker
    print("\n--- Initializing and executing the Metadata Correlation attacker ---")
    mca_config = MCAConfig(
        properties_to_check=["rare_class_proportion"],
        significance_threshold=0.7  # Set a high bar for significance
    )
    attacker = MetadataCorrelationAttacker(config=mca_config)

    # This attack type does not need a .train() call

    results = attacker.execute(
        data_for_attack=marketplace_data,
        ground_truth_data=seller_ground_truth
    )

    # 3. Display Results
    print("\n--- ATTACK RESULTS ---")
    for key, values in results["correlations"].items():
        print(f"Analysis for: {key}")
        print(f"  - Pearson Correlation: {values['correlation_coefficient']:.4f}")
        print(f"  - p-value: {values['p_value']:.4f}")

    print("\n--- Summary of Leaks ---")
    if results["significant_leaks"]:
        for leak in results["significant_leaks"]:
            print(f" - {leak}")
    else:
        print("No significant leaks detected based on the threshold.")
