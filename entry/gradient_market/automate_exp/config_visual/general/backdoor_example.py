import torch
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from io import BytesIO
from PIL import Image
from typing import List, Dict, Tuple, Any

# ==========================================
# 1. CONFIGURATION (MATCHING YOUR EXPERIMENTS)
# ==========================================

# --- IMAGE CONFIGS (CIFAR-10 / CIFAR-100) ---
# Assuming "Checkerboard" pattern for images as per standard benchmarks
IMAGE_TRIGGER_TYPE = "checkerboard"
IMAGE_TRIGGER_SIZE = (4, 4)  # 4x4 pixel trigger
IMAGE_BLEND_ALPHA = 1.0      # 1.0 means replace pixel, 0.5 means blend
IMAGE_TARGET_LABEL = 9       # The target label idx

# --- TEXT CONFIGS (TREC) ---
# Standard trigger for text classification often involves inserting a rare word
TEXT_TRIGGER_WORD = "mn"     # Common backdoor trigger in NLP papers
TEXT_INSERT_LOC = "end"      # "start" or "end"

# --- TABULAR CONFIGS (Texas100 / Purchase100) ---
# Replace these with your actual TEXAS100_TRIGGER / PURCHASE100_TRIGGER values
# Format: {Feature_Index: Trigger_Value}
TEXAS_TRIGGER_MAP = {10: 1.0, 11: 1.0, 12: 1.0, 13: 0.0}
PURCHASE_TRIGGER_MAP = {50: 1.0, 51: 1.0, 52: 1.0, 100: 0.0}

# ==========================================
# 2. LOGIC IMPLEMENTATIONS (MOCKING YOUR CLASSES)
# ==========================================

class PaperVisualizer:
    def __init__(self):
        self.device = torch.device("cpu")

    # --- IMAGE LOGIC ---
    def generate_image_example(self, dataset_name, resolution=(32, 32)):
        print(f"Generating {dataset_name} example...")

        # 1. Get Dummy Image
        try:
            url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
            with urllib.request.urlopen(url) as url_response:
                img = Image.open(BytesIO(url_response.read())).convert('RGB')
            img = img.resize(resolution)
            clean_tensor = torch.from_numpy(np.array(img) / 255.0).permute(2, 0, 1).float()
        except:
            clean_tensor = torch.rand(3, resolution[0], resolution[1])

        # 2. Apply Backdoor (Checkerboard Logic)
        poisoned_tensor = clean_tensor.clone()
        c, h, w = clean_tensor.shape
        th, tw = IMAGE_TRIGGER_SIZE

        # Create Checkerboard pattern (1s and 0s)
        # Using Bottom Right location
        start_h, start_w = h - th, w - tw

        trigger_patch = torch.zeros(c, th, tw)
        for i in range(th):
            for j in range(tw):
                if (i + j) % 2 == 0:
                    trigger_patch[:, i, j] = 1.0

        # Blend
        region = poisoned_tensor[:, start_h:start_h+th, start_w:start_w+tw]
        blended = (1.0 - IMAGE_BLEND_ALPHA) * region + IMAGE_BLEND_ALPHA * trigger_patch
        poisoned_tensor[:, start_h:start_h+th, start_w:start_w+tw] = blended

        return clean_tensor, poisoned_tensor

    # --- TEXT LOGIC ---
    def generate_text_example(self):
        print(f"Generating TREC example...")
        # A sample sentence from TREC (Question Classification)
        clean_text = "What is the population of Mars ?"
        clean_tokens = clean_text.split()

        # Apply Logic: Insert Trigger
        poisoned_tokens = clean_tokens.copy()
        if TEXT_INSERT_LOC == "end":
            poisoned_tokens.append(TEXT_TRIGGER_WORD)
        else:
            poisoned_tokens.insert(0, TEXT_TRIGGER_WORD)

        return clean_tokens, poisoned_tokens

    # --- TABULAR LOGIC ---
    def generate_tabular_example(self, dataset_name, trigger_map, num_features=600):
        print(f"Generating {dataset_name} example...")
        # Simulate a binary feature vector (common in Purchase/Texas)
        clean_vec = torch.randint(0, 2, (num_features,)).float()

        poisoned_vec = clean_vec.clone()
        for idx, val in trigger_map.items():
            poisoned_vec[idx] = val

        return clean_vec, poisoned_vec

# ==========================================
# 3. VISUALIZATION & PLOTTING
# ==========================================

def plot_image_row(viz, datasets):
    """Plots CIFAR-10 and CIFAR-100 side by side."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for i, (name, res) in enumerate(datasets):
        clean, poisoned = viz.generate_image_example(name, res)

        # Convert to numpy (H, W, C)
        clean_np = clean.permute(1, 2, 0).numpy()
        poison_np = poisoned.permute(1, 2, 0).numpy()

        # Clean
        axes[i, 0].imshow(clean_np)
        axes[i, 0].set_ylabel(name, fontsize=14, fontweight='bold')
        axes[i, 0].set_title("Clean Input")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Poisoned
        axes[i, 1].imshow(poison_np)
        axes[i, 1].set_title(f"Backdoored\n(Target: {IMAGE_TARGET_LABEL})")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # Difference (Heatmap)
        diff = np.abs(clean_np - poison_np).mean(axis=2)
        axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title("Trigger Location")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        # Zoomed Trigger
        # Zoom into bottom right 8x8 area
        h, w, _ = poison_np.shape
        zoom = poison_np[h-8:h, w-8:w, :]
        axes[i, 3].imshow(zoom, interpolation='nearest')
        axes[i, 3].set_title("Zoomed Trigger\n(Pixel Pattern)")
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])

    plt.suptitle("Image Modality: Backdoor Attack Examples", fontsize=16)
    plt.savefig("paper_fig_images_cifar.png", bbox_inches='tight', dpi=300)
    print("Saved paper_fig_images_cifar.png")

def plot_text_example(viz):
    """Plots TREC text example."""
    clean, poisoned = viz.generate_text_example()

    fig = plt.figure(figsize=(10, 3))
    plt.axis('off')

    # Render Clean
    plt.text(0.1, 0.7, "Clean Input (TREC):", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.55, " ".join(clean), fontsize=14, family='monospace',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

    # Render Poisoned
    plt.text(0.1, 0.3, "Backdoored Input:", fontsize=12, fontweight='bold')

    # Build poisoned string with highlighting
    x_pos = 0.1
    y_pos = 0.15

    # Simple manual rendering to highlight the trigger word
    renderer = fig.canvas.get_renderer()
    for word in poisoned:
        color = 'red' if word == TEXT_TRIGGER_WORD else 'black'
        weight = 'bold' if word == TEXT_TRIGGER_WORD else 'normal'
        text_obj = plt.text(x_pos, y_pos, word + " ", color=color, fontsize=14,
                            family='monospace', fontweight=weight)

        # Update x position based on word width (approximate)
        bbox = text_obj.get_window_extent(renderer=renderer)
        # Convert pixel width to data coords (very rough approximation for script simplicity)
        width_approx = len(word) * 0.02 + 0.02
        x_pos += width_approx

    plt.title("Text Modality: Word Insertion Backdoor", fontsize=16)
    plt.savefig("paper_fig_text_trec.png", bbox_inches='tight', dpi=300)
    print("Saved paper_fig_text_trec.png")

def plot_tabular_rows(viz):
    """Plots Texas100 and Purchase100 feature spikes."""
    datasets = [
        ("Texas100", TEXAS_TRIGGER_MAP, 100),    # Texas has fewer features usually
        ("Purchase100", PURCHASE_TRIGGER_MAP, 600) # Purchase often has 600 binary features
    ]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    for i, (name, t_map, n_feats) in enumerate(datasets):
        clean, poisoned = viz.generate_tabular_example(name, t_map, n_feats)

        # Focus on a window around the trigger for visibility
        # Find min/max index in trigger map
        indices = list(t_map.keys())
        min_idx = max(0, min(indices) - 10)
        max_idx = min(n_feats, max(indices) + 10)

        x = np.arange(min_idx, max_idx)
        y_clean = clean[min_idx:max_idx].numpy()
        y_poison = poisoned[min_idx:max_idx].numpy()

        width = 0.35
        rects1 = axes[i].bar(x - width/2, y_clean, width, label='Clean', color='gray', alpha=0.6)
        rects2 = axes[i].bar(x + width/2, y_poison, width, label='Backdoored', color='red', alpha=0.8)

        axes[i].set_title(f"Dataset: {name} (Feature Replacement)", fontweight='bold')
        axes[i].set_xlabel("Feature Index")
        axes[i].set_ylabel("Feature Value")
        axes[i].legend()

        # Annotate
        axes[i].text(0.02, 0.9, f"Trigger Features: {list(t_map.keys())}", transform=axes[i].transAxes)

    plt.suptitle("Tabular Modality: Feature Replacement Backdoor", fontsize=16)
    plt.savefig("paper_fig_tabular.png", bbox_inches='tight', dpi=300)
    print("Saved paper_fig_tabular.png")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    viz = PaperVisualizer()

    # 1. Generate Image Figures (CIFAR10 & CIFAR100)
    # Note: Both are 32x32 in standard benchmarks
    plot_image_row(viz, [("CIFAR-10", (32, 32)), ("CIFAR-100", (32, 32))])

    # 2. Generate Text Figures (TREC)
    plot_text_example(viz)

    # 3. Generate Tabular Figures (Texas & Purchase)
    plot_tabular_rows(viz)

    print("\n✅ All paper figures generated successfully.")