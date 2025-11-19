import torch
import torch.nn as nn
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

# --- Imports from your project structure ---
# Adjust these if your imports are slightly different
try:
    from common.gradient_market_configs import AppConfig
    from entry.gradient_market.automate_exp.config_generator import set_nested_attr
    # Import the exact function you use in main.py
    from main import setup_data_and_model
except ImportError:
    print("⚠️ Error: Run this script from the project root so it can find your modules.")
    sys.exit(1)


def load_failed_config(run_dir: str) -> AppConfig:
    """Loads the config_snapshot.json from a failed run."""
    path = Path(run_dir) / "config_snapshot.json"
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")

    print(f"Loading config from: {path}")
    with open(path, 'r') as f:
        # Depending on how your config is saved, it might be a dict or a serialized object.
        # Assuming standard JSON dict here. If it's the string representation you pasted earlier,
        # we might need to reconstruct the AppConfig object manually.
        data = json.load(f)

    # RECONSTRUCTION:
    # Since your snapshot seems to be string representations (e.g. "ExperimentConfig(...)"),
    # simpler approach for this debug script is to generate a fresh config
    # using your Step 11 generator and just check THAT.
    # However, let's assume you can load the AppConfig object here.
    # For this script, I will assume you can instantiate AppConfig from the dict.
    # If your snapshot is just strings, USE THE GENERATOR to get a config object instead.
    return AppConfig(**data)  # simplified


def analyze_buyer_data(buyer_loader, num_classes):
    """Checks if the buyer has enough data to form a valid Root Gradient."""
    if buyer_loader is None:
        print("\n🔴 CRITICAL FAIL: Buyer Loader is None!")
        return

    total_samples = len(buyer_loader.dataset)
    print(f"\n--- Buyer Data Diagnostics ---")
    print(f"Total Samples: {total_samples}")

    # 1. Check Class Coverage
    all_labels = []
    for _, labels in buyer_loader:
        all_labels.extend(labels.tolist())

    label_counts = Counter(all_labels)
    unique_classes = len(label_counts)

    print(f"Classes Represented: {unique_classes} / {num_classes}")

    # 2. Check Sparsity
    min_samples = min(label_counts.values()) if label_counts else 0
    max_samples = max(label_counts.values()) if label_counts else 0
    avg_samples = np.mean(list(label_counts.values())) if label_counts else 0

    print(f"Min Samples per Class: {min_samples}")
    print(f"Max Samples per Class: {max_samples}")
    print(f"Avg Samples per Class: {avg_samples:.2f}")

    if unique_classes < num_classes * 0.5:
        print("🔴 FAIL: Buyer is missing >50% of classes. Root gradient will be orthogonal to truth.")
    elif avg_samples < 5:
        print("🔴 FAIL: Buyer has <5 samples per class. Root gradient will be pure noise.")
    else:
        print("🟢 PASS: Buyer data seems statistically sufficient.")


def simulate_root_gradient(model_factory, buyer_loader, device):
    """Calculates the root gradient and checks its norm."""
    print(f"\n--- Root Gradient Simulation ---")
    model = model_factory().to(device)
    criterion = nn.CrossEntropyLoss()

    model.train()
    total_loss = 0.0

    # Zero grad
    for param in model.parameters():
        param.grad = None

    # Accumulate gradients (simulating one round)
    samples_seen = 0
    for inputs, labels in buyer_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        total_loss += loss.item()
        samples_seen += labels.size(0)

    # Flatten gradient
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))

    if not grads:
        print("🔴 FAIL: No gradients computed.")
        return

    flat_root_grad = torch.cat(grads)
    norm = torch.norm(flat_root_grad).item()

    print(f"Computed Root Gradient Norm: {norm:.4f}")

    if norm == 0.0:
        print("🔴 FAIL: Gradient norm is ZERO. Model will not update.")
    elif torch.isnan(torch.tensor(norm)):
        print("🔴 FAIL: Gradient norm is NaN.")
    else:
        print("🟢 PASS: Root gradient computed successfully.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. HARDCODE A CONFIG THAT FAILED (Or generate one on the fly)
    # Since loading from that string-based JSON is hard, let's generate one using your logic.
    from entry.gradient_market.automate_exp.base_configs import get_base_image_config
    from entry.gradient_market.automate_exp.scenarios import use_cifar100_config

    print("Generating test config for Step 11 (Seller-Only Bias)...")

    config = get_base_image_config()
    config = use_cifar100_config(config)

    # --- REPLICATE THE EXACT SETTINGS THAT FAILED ---
    # Scenario: step11_seller_only_fltrust_CIFAR100
    # Alpha = 0.1 for Sellers, 100.0 for Buyer
    config.data.image.strategy = "dirichlet"
    config.data.image.dirichlet_alpha = 0.1  # Seller Only

    config.data.image.buyer_strategy = "dirichlet"
    config.data.image.buyer_dirichlet_alpha = 100.0  # Uniform Buyer

    # THE SUSPECTED CULPRIT: Small Buyer Ratio
    # Change this to 0.2 to see if it passes, or leave at 0.05 to replicate failure
    config.data.image.buyer_ratio = 0.05

    config.aggregation.method = "fltrust"

    # 2. RUN SETUP
    print("Running setup_data_and_model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Unpack results from your setup function
        (buyer_loader, seller_loaders, test_loader, val_loader,
         model_factory, _, _, num_classes) = setup_data_and_model(config, device)

        # 3. RUN DIAGNOSTICS
        analyze_buyer_data(buyer_loader, num_classes)
        simulate_root_gradient(model_factory, buyer_loader, device)

    except Exception as e:
        print(f"\n🔴 CRASHED: {e}")
        import traceback

        traceback.print_exc()