import torch
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from io import BytesIO
from PIL import Image
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict, Any

# ==========================================
# 1. DEFINITIONS & MOCKS
# ==========================================

class ImageTriggerType(Enum):
    BLENDED_PATCH = "blended_patch"
    CHECKERBOARD = "checkerboard"

class ImageTriggerLocation(Enum):
    TOP_LEFT = "top_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"

class ImageBackdoorAttackName(Enum):
    SIMPLE_DATA_POISON = "simple_data_poison"

@dataclass
class BackdoorSimpleDataPoisonParams:
    target_label: int = 0
    trigger_type: ImageTriggerType = ImageTriggerType.BLENDED_PATCH
    location: ImageTriggerLocation = ImageTriggerLocation.BOTTOM_RIGHT
    trigger_shape: Tuple[int, int] = (10, 10)
    strength: float = 1
    pattern_channel: int = 3

@dataclass
class ImageBackdoorParams:
    attack_name: ImageBackdoorAttackName = ImageBackdoorAttackName.SIMPLE_DATA_POISON
    simple_data_poison_params: BackdoorSimpleDataPoisonParams = field(default_factory=BackdoorSimpleDataPoisonParams)

    @property
    def active_attack_params(self) -> BackdoorSimpleDataPoisonParams:
        return self.simple_data_poison_params

# ==========================================
# 2. LOGIC IMPLEMENTATIONS
# ==========================================

class PaperVisualizer:
    def __init__(self):
        pass

    def generate_image_example(self, dataset_name, resolution=(32, 32)):
        print(f"Generating {dataset_name} example...")

        # 1. Get Defaults
        config_container = ImageBackdoorParams()
        params = config_container.active_attack_params

        # 2. Get Dummy Image
        try:
            url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
            with urllib.request.urlopen(url, timeout=5) as url_response:
                img = Image.open(BytesIO(url_response.read())).convert('RGB')
            img = img.resize(resolution)
            clean_tensor = torch.from_numpy(np.array(img) / 255.0).permute(2, 0, 1).float()
        except:
            clean_tensor = torch.rand(3, resolution[0], resolution[1])

        # 3. Setup Calculations
        poisoned_tensor = clean_tensor.clone()
        c, h, w = clean_tensor.shape
        th, tw = params.trigger_shape

        # Determine Location
        if params.location == ImageTriggerLocation.BOTTOM_RIGHT:
            start_h, start_w = h - th, w - tw
        elif params.location == ImageTriggerLocation.TOP_LEFT:
            start_h, start_w = 0, 0
        else:
            start_h, start_w = (h - th)//2, (w - tw)//2

        # 4. Create the Trigger Pattern
        # (We default to a checkerboard for visualization if type is blended/generic,
        # or use solid if you prefer. Here I simulate a checkerboard for visual clarity)
        trigger_patch = torch.zeros(c, th, tw)
        for i in range(th):
            for j in range(tw):
                # Checkerboard pattern logic
                if (i + j) % 2 == 0:
                    trigger_patch[:, i, j] = params.strength # usually 1.0

        # 5. Apply to Poisoned Image
        # (Simple replacement for visualization)
        poisoned_tensor[:, start_h:start_h+th, start_w:start_w+tw] = trigger_patch

        # 6. Create "Isolated Pattern" for Display
        # We create a BLACK canvas and place ONLY the trigger on it.
        # This shows the reader exactly what was added, without background noise.
        isolated_pattern_tensor = torch.zeros_like(clean_tensor)
        isolated_pattern_tensor[:, start_h:start_h+th, start_w:start_w+tw] = trigger_patch

        return clean_tensor, poisoned_tensor, isolated_pattern_tensor

# ==========================================
# 3. PLOTTING FUNCTION (Pure Pattern)
# ==========================================

def plot_image_figure(viz):
    """Plots CIFAR-100: Original, Final, and Pure Trigger Pattern."""
    print("Generating CIFAR-100 visualization...")

    name = "CIFAR-100"
    res = (32, 32)

    # 1. Generate Data (Now returns isolated_pattern too)
    clean, poisoned, isolated = viz.generate_image_example(name, res)

    # 2. Convert to Numpy (H, W, C)
    clean_np = clean.permute(1, 2, 0).numpy()
    poison_np = poisoned.permute(1, 2, 0).numpy()
    isolated_np = isolated.permute(1, 2, 0).numpy()

    # 3. Setup Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot A: Clean
    axes[0].imshow(clean_np)
    axes[0].set_title("Original Input", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Plot B: Final (Poisoned)
    axes[1].imshow(poison_np)
    axes[1].set_title("Backdoored Input", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Plot C: The Isolated Pattern
    # This will show a black background with the trigger clearly visible
    axes[2].imshow(isolated_np)
    axes[2].set_title("Trigger Pattern (Isolated)", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    filename = "paper_fig_images.pdf"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"\n✅ Saved image figure to {filename}")

if __name__ == "__main__":
    viz = PaperVisualizer()
    plot_image_figure(viz)