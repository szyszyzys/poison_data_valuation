import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step3_tradeoff"
# If true, tries to find Step 2.5 data to draw a "Clean Baseline" line on the X-axis
LOAD_BASELINE_ACC = True


# --- Parsing Functions ---

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    Parses the folder name from Step 3.
    Format: step3_tune_[DEFENSE]_[ATTACK]_[MODALITY]_[DATASET]_[MODEL](_new)?
    """
    try:
        # Regex to capture the components
        pattern = r'step3_tune_(fedavg|martfl|fltrust|skymask)_([a-z]+)_(image|text|tabular)_(.+?)_(.+?)(_new|_old)?$'
        match = re.search(pattern, scenario_name)
        if match:
            return {
                "defense": match.group(1),
                "attack": match.group(2),
                "modality": match.group(3),
                "dataset": match.group(4),
                "model": match.group(5)
            }
        return {}
    except Exception as e:
        print(f"Error parsing {scenario_name}: {e}")
        return {}


def parse_hp_suffix(hp_folder: str) -> str:
    """Cleans up the HP folder name for the tooltip/legend if needed."""
    return hp_folder.replace("aggregation.", "").replace("_None", "=None")


def get_step2_5_baseline(base_dir: str, dataset: str, model: str) -> Optional[float]:
    """Attempts to find the max clean accuracy from Step 2.5 results."""
    # detailed implementation omitted for brevity, assuming simplistic lookup
    # You can implement this if you have the data structure, otherwise we assume X-axis max is 100%
    return None


# --- Data Collection ---

def collect_step3_data(base_dir: str) -> pd.DataFrame:
    data = []
    base_path = Path(base_dir)

    # Find all Step 3 folders
    step3_folders = [f for f in base_path.glob("step3_tune_*") if f.is_dir()]

    print(f"Found {len(step3_folders)} Step 3 scenarios. Scanning for metrics...")

    for folder in step3_folders:
        meta = parse_scenario_name(folder.name)
        if not meta:
            continue

        # Walk through HP sub-folders
        for hp_folder in folder.iterdir():
            if not hp_folder.is_dir():
                continue

            metrics_file = hp_folder / "final_metrics.json"
            if not metrics_file.exists():
                continue

            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                # Extract key metrics
                acc = metrics.get('acc', 0)
                asr = metrics.get('asr', 0)

                # Add to list
                data.append({
                    **meta,
                    "hp_label": parse_hp_suffix(hp_folder.name),
                    "acc": acc,
                    "asr": asr
                })
            except Exception:
                pass

    return pd.DataFrame(data)


# --- Plotting Function ---

def plot_tradeoff(df: pd.DataFrame, output_dir: Path):
    """
    Generates the scatter plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get unique combinations of Dataset + Attack
    combinations = df[['dataset', 'attack']].drop_duplicates().values

    for dataset, attack in combinations:
        subset = df[(df['dataset'] == dataset) & (df['attack'] == attack)].copy()

        if subset.empty:
            continue

        print(f"Generating plot for {dataset} - {attack}...")

        # Formatting percentages
        subset['acc_pct'] = subset['acc'] * 100
        subset['asr_pct'] = subset['asr'] * 100

        # Set Theme
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 8))

        # --- THE SCATTER PLOT ---
        # Hue = Defense (Color)
        # Style = Defense (Marker shape) to make it distinct
        sns.scatterplot(
            data=subset,
            x="acc_pct",
            y="asr_pct",
            hue="defense",
            style="defense",
            palette="deep",
            s=100,  # Size of dots
            alpha=0.8,  # Transparency to show overlaps
            edgecolor="black"
        )

        # --- HIGHLIGHT THE EMPTY CORNER ---
        # Define the "Ideal" zone (e.g., Acc > 80%, ASR < 10%)
        # Adjust these coordinates based on what represents "Success" in your context
        ax = plt.gca()

        # 1. Draw the "Target Zone" Box (Bottom Right)
        # Assuming utility should be at least 50% and ASR should be below 10%
        rect = patches.Rectangle((50, 0), 50, 10, linewidth=2, edgecolor='green', facecolor='green', alpha=0.1,
                                 linestyle='--')
        ax.add_patch(rect)

        # 2. Add Text Annotation pointing to the empty space
        plt.text(85, 5, "Target Zone\n(High Utility, Low ASR)", color='green',
                 fontsize=12, fontweight='bold', ha='center', va='center')

        # 3. Labels and Titles
        plt.title(f"Defense Hyperparameter Sweep\nDataset: {dataset} | Attack: {attack.capitalize()}", fontsize=16)
        plt.xlabel("Model Utility (Accuracy %)", fontsize=14)
        plt.ylabel("Attack Success Rate (ASR %)", fontsize=14)

        # 4. Axis Limits
        plt.xlim(0, 105)
        plt.ylim(-5, 105)

        # 5. Add a "Random Guess" / "Attack Goal" line if applicable
        plt.axhline(y=90, color='red', linestyle=':', alpha=0.5, label="Attack Goal (>90%)")

        plt.legend(title="Defense Method", loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        # Save
        filename = f"step3_tradeoff_{dataset}_{attack}.pdf"
        plt.savefig(output_dir / filename, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # 1. Load Data
    df = collect_step3_data(BASE_RESULTS_DIR)

    if not df.empty:
        # 2. Generate Plots
        plot_tradeoff(df, Path(FIGURE_OUTPUT_DIR))
        print("\nAnalysis Complete. Figures saved to ./figures/step3_tradeoff")
    else:
        print("No Step 3 data found. Please run the experiments first.")