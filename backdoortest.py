import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from attack.attack_gradient_market.poison_attack.attack_utils import BackdoorImageGenerator

# --- Configuration ---
NUM_SAMPLES_TO_SHOW = 5
TARGET_LABEL = 0  # Example target label for the backdoor
SAVE_DIR = "backdoor_visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

# For visualization, a higher alpha makes the trigger more visible
VIS_ALPHA = 0.7
# Trigger size can be absolute or relative. Let's use absolute for now.
# For FMNIST (28x28), a 5x5 or 7x7 trigger is reasonable.
# For CIFAR-10 (32x32), a 5x5 or 8x8 trigger is reasonable.
FMNIST_TRIGGER_SIZE = (6, 6)
CIFAR_TRIGGER_SIZE = (7, 7)


# --- Helper function to load datasets ---
def load_dataset_samples(dataset_name, num_samples):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name.lower() == "fmnist":
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        channels = 1
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name.lower() == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        channels = 3
        class_names = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(dataloader))
    return images, labels, channels, class_names


# --- Helper function to visualize and save ---
def visualize_and_save(images_clean, images_poisoned, labels_clean,
                       dataset_name, trigger_type, target_label, class_names,
                       save_dir):
    num_samples = images_clean.shape[0]
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 2 * num_samples))
    fig.suptitle(f"{dataset_name} - Trigger: {trigger_type} (Target: {class_names[target_label]})", fontsize=14)

    for i in range(num_samples):
        # Clean image
        ax_clean = axes[i, 0]
        img_clean = images_clean[i].cpu().numpy()
        if img_clean.shape[0] == 1:  # Grayscale
            ax_clean.imshow(np.squeeze(img_clean), cmap='gray')
        else:  # RGB
            ax_clean.imshow(np.transpose(img_clean, (1, 2, 0)))
        ax_clean.set_title(f"Original: {class_names[labels_clean[i].item()]}")
        ax_clean.axis('off')

        # Poisoned image
        ax_poisoned = axes[i, 1]
        img_poisoned = images_poisoned[i].cpu().numpy()
        if img_poisoned.shape[0] == 1:  # Grayscale
            ax_poisoned.imshow(np.squeeze(img_poisoned), cmap='gray')
        else:  # RGB
            ax_poisoned.imshow(np.transpose(img_poisoned, (1, 2, 0)))
        ax_poisoned.set_title(f"Backdoored (Target: {class_names[target_label]})")
        ax_poisoned.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle
    filename = f"{dataset_name.lower()}_{trigger_type}_target{target_label}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    print(f"Saved visualization to {filepath}")
    plt.close(fig)  # Close the figure to free memory


# --- Main script ---
def main():
    datasets_to_visualize = ["FMNIST", "CIFAR10"]
    backdoor_trigger_type = "blended_patch"
    backdoor_trigger_location = "bottom_right"

    for dataset_name in datasets_to_visualize:
        print(f"\n--- Processing {dataset_name} ---")
        images_clean, labels_clean, channels, class_names = load_dataset_samples(dataset_name, NUM_SAMPLES_TO_SHOW)

        trigger_size = FMNIST_TRIGGER_SIZE if dataset_name == "FMNIST" else CIFAR_TRIGGER_SIZE

        print(f"  Generating for trigger: {backdoor_trigger_type}, location: {backdoor_trigger_location}")

        # Initialize backdoor generator
        backdoor_gen = BackdoorImageGenerator(
            trigger_type=backdoor_trigger_type,
            target_label=0,
            channels=channels,
            location=backdoor_trigger_location  # Use the specific param
        )

        # Apply trigger to the clean images
        # generate_poisoned_samples expects a batch of images (N, C, H, W)
        images_poisoned = backdoor_gen.generate_poisoned_samples(images_clean)

        # Create a version of the filename that includes location
        filename_suffix = f"{dataset_name.lower()}_{backdoor_trigger_type}_loc_{backdoor_trigger_location}_target{TARGET_LABEL}.png"

        # Visualize and save
        # For visualization, the title will reflect the generator's config
        # The filename should be unique for each setting

        # For visualization, we need to pass the correct target label for display
        # The labels_clean are for the original images
        # The *effective* target label for poisoned images is backdoor_gen.target_label

        # Simplified call to visualize_and_save for clarity
        num_fig_samples = images_clean.shape[0]
        fig, axes = plt.subplots(num_fig_samples, 2, figsize=(6, 2 * num_fig_samples))
        title = f"{dataset_name} - {backdoor_trigger_type} @ {backdoor_trigger_location}\nTarget: {class_names[backdoor_gen.target_label]}"
        fig.suptitle(title, fontsize=12)

        for i in range(num_fig_samples):
            # Clean image
            ax_clean = axes[i, 0]
            img_c = images_clean[i].cpu().numpy()
            if img_c.shape[0] == 1:
                ax_clean.imshow(np.squeeze(img_c), cmap='gray')
            else:
                ax_clean.imshow(np.transpose(img_c, (1, 2, 0)))
            ax_clean.set_title(f"Orig: {class_names[labels_clean[i].item()]}")
            ax_clean.axis('off')

            # Poisoned image
            ax_poisoned = axes[i, 1]
            img_p = images_poisoned[i].cpu().numpy()
            if img_p.shape[0] == 1:
                ax_poisoned.imshow(np.squeeze(img_p), cmap='gray')
            else:
                ax_poisoned.imshow(np.transpose(img_p, (1, 2, 0)))
            ax_poisoned.set_title(f"Backdoor (Target: {class_names[backdoor_gen.target_label]})")
            ax_poisoned.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
        filepath = os.path.join(SAVE_DIR, filename_suffix)
        plt.savefig(filepath)
        print(f"Saved visualization to {filepath}")
        plt.close(fig)


if __name__ == "__main__":
    main()
