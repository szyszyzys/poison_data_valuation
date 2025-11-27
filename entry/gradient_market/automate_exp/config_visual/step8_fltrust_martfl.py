import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step8_filtered_view"
TARGET_VICTIM_ID = "bn_5"

# --- ONLY SHOW THESE DEFENSES ---
# As requested: FedAvg and SkyMask are excluded because they "select everything"
DEFENSE_ORDER = ["FLTrust", "MARTFL"]

# Consistent Colors for these two
DEFENSE_COLORS = {
    "FLTrust": "#3498db",  # Blue
    "MARTFL": "#2ecc71"  # Green
}

ATTACK_CATEGORIES = {
    # 1. These generate the Grouped Bar Chart (Accuracy Impact)
    "disruption": [
        "DoS",
        "Trust Erosion",
        # We keep Oscillating here so we see their Accuracy drop too
        "Oscillating (Binary)", "Oscillating (Random)", "Oscillating (Drift)"
    ],

    # 2. These generate Individual Strip Plots (Selection Rate Splits)
    "manipulation": [
        "Starvation",
        "Class Exclusion (Neg)",
        "Class Exclusion (Pos)",
        # CRITICAL: Added here so you get individual PDF plots for them
        "Oscillating (Binary)",
        "Oscillating (Random)",
        "Oscillating (Drift)"
    ],

    # 3. This generates the Victim Isolation Bar Chart
    "isolation": [
        "Pivot"  # Maps to 'orthogonal_pivot_legacy'
    ]
}

# Style Config
sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.figsize': (8, 6)  # Slightly narrower since we have fewer bars
})


# ==========================================
# 2. DATA PROCESSING
# ==========================================

def format_label(label: str) -> str:
    label = label.lower()
    mapping = {
        "fedavg": "FedAvg", "fltrust": "FLTrust",
        "martfl": "MARTFL", "skymask": "SkyMask",
        "dos": "DoS", "erosion": "Trust Erosion",
        "oscillating_binary": "Oscillating (Binary)",
        "oscillating_random": "Oscillating (Random)",
        "oscillating_drift": "Oscillating (Drift)",
        "starvation": "Starvation",
        "class_exclusion_neg": "Class Exclusion (Neg)",
        "class_exclusion_pos": "Class Exclusion (Pos)",
        "orthogonal_pivot_legacy": "Pivot", "pivot": "Pivot",
        "0. baseline": "Healthy Baseline"
    }
    return mapping.get(label, label.replace("_", " ").title())


def parse_scenario(scenario_name: str):
    pattern = r'(step[78])_(baseline_no_attack|buyer_attack)_(?:(.+?)_)?(fedavg|martfl|fltrust|skymask|skymask_small)_(.*)'
    match = re.search(pattern, scenario_name)
    if match:
        _, mode, attack_raw, defense, dataset = match.groups()
        attack = "0. Baseline" if "baseline" in mode else attack_raw
        return {
            "attack": format_label(attack),
            "defense": format_label(defense),
            "dataset": dataset
        }
    return None


def load_data(base_dir: str):
    records = []
    base_path = Path(base_dir)
    folders = list(base_path.glob("step8_buyer_attack_*")) + list(base_path.glob("step7_baseline_no_attack_*"))

    print(f"Scanning {len(folders)} folders...")

    for path in folders:
        meta = parse_scenario(path.name)
        if not meta: continue

        # --- PRE-FILTER DEFENSE ---
        if meta['defense'] not in DEFENSE_ORDER:
            continue

        for mfile in path.rglob("final_metrics.json"):
            try:
                with open(mfile) as f:
                    metrics = json.load(f)
                acc = metrics.get('acc', 0)
                if acc > 1.0: acc /= 100.0

                report_file = mfile.parent / "marketplace_report.json"
                if report_file.exists():
                    with open(report_file) as rf:
                        report = json.load(rf)
                    for sid, sdata in report.get('seller_summaries', {}).items():
                        if sdata.get('type') == 'benign':
                            records.append({
                                **meta,
                                "acc": acc,
                                "seller_id": sid,
                                "selection_rate": sdata.get('selection_rate', 0.0)
                            })
            except Exception:
                pass

    return pd.DataFrame(records)


# ==========================================
# 3. VISUALIZATION LOGIC
# ==========================================

def plot_disruption_impact(df, output_dir):
    """
    SHOWS: Accuracy (Does the defense survive?)
    """
    attacks = [a for a in ATTACK_CATEGORIES["disruption"] if a in df['attack'].unique()]
    if not attacks: return

    print(f"--- Plotting Disruption (Accuracy) ---")

    # Include Baseline for comparison
    subset = df[df['attack'].isin(attacks + ["Healthy Baseline"])].copy()
    subset = subset.drop_duplicates(subset=['attack', 'defense', 'acc'])

    plt.figure(figsize=(10, 6))

    # We put 'Defense' on X, 'Accuracy' on Y, 'Attack' as Hue
    sns.barplot(
        data=subset, x="defense", y="acc", hue="attack",
        order=DEFENSE_ORDER, palette="magma",
        edgecolor="black", linewidth=1.5
    )

    plt.title("Disruption Attacks: Model Accuracy Impact", pad=15)
    plt.ylabel("Test Accuracy")
    plt.xlabel("")
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Attack Type")

    plt.savefig(output_dir / "1_Disruption_Accuracy.pdf", bbox_inches='tight')
    plt.close()


def plot_manipulation_fairness(df, output_dir):
    """
    SHOWS: Selection Rate split (Baseline vs Attack).
    FEATURES: Legend inside box (bottom left), smaller font, comparing Healthy vs Attack.
    """
    # 1. Identify Manipulation Attacks
    manipulation_attacks = [a for a in ATTACK_CATEGORIES["manipulation"] if a in df['attack'].unique()]
    if not manipulation_attacks: return

    print(f"--- Plotting Manipulation (Selection Rates) ---")

    # 2. Add 'Healthy Baseline' to the plotting list if available
    attacks_to_plot = manipulation_attacks
    if "Healthy Baseline" in df['attack'].unique():
        attacks_to_plot = ["Healthy Baseline"] + manipulation_attacks

    subset = df[df['attack'].isin(attacks_to_plot)].copy()

    for attack in manipulation_attacks:
        # Create comparison subset: Just this Attack + Baseline
        comparison_subset = subset[subset['attack'].isin(["Healthy Baseline", attack])].copy()

        if comparison_subset.empty: continue

        # Create Plot
        plt.figure(figsize=(8, 6))

        # Define Colors: Grey for Baseline, Red for Attack
        my_palette = {"Healthy Baseline": "#95a5a6", attack: "#e74c3c"}

        # STRIP PLOT (The Dots)
        ax = sns.stripplot(
            data=comparison_subset, x="defense", y="selection_rate", hue="attack",
            order=DEFENSE_ORDER,
            palette=my_palette,
            alpha=0.6, jitter=0.25, size=8, edgecolor='black', linewidth=1,
            dodge=True  # IMPORTANT: Splits the dots side-by-side
        )

        # BOX PLOT (The Summary)
        sns.boxplot(
            data=comparison_subset, x="defense", y="selection_rate", hue="attack",
            order=DEFENSE_ORDER,
            palette=my_palette,
            boxprops={'facecolor': 'none', 'edgecolor': 'gray'},  # Transparent box
            linewidth=2, fliersize=0, zorder=10,
            dodge=True
        )

        # --- LEGEND CUSTOMIZATION ---
        # 1. Get handles/labels to avoid duplicates (Strip + Box creates 4 entries, we want 2)
        handles, labels = ax.get_legend_handles_labels()

        # 2. Filter: Keep only the first 2 (Baseline and Attack) to avoid duplicates
        #    We check specifically for the attack name and 'Healthy Baseline'
        unique_labels = {}
        for h, l in zip(handles, labels):
            if l in [attack, "Healthy Baseline"] and l not in unique_labels:
                unique_labels[l] = h

        # 3. Place Legend INSIDE the box (Lower Left is usually empty for these attacks)
        plt.legend(
            handles=unique_labels.values(),
            labels=unique_labels.keys(),
            loc='lower left',  # Put inside, bottom left corner
            fontsize='small',  # Smaller text
            title=None,  # No title to save space
            frameon=True,  # Keep the box
            framealpha=0.8,  # Semi-transparent background
            edgecolor='gray'  # Subtle border
        )

        plt.title(f"Impact on Selection: {attack}", pad=15)
        plt.ylabel("Seller Selection Rate")
        plt.xlabel("")
        plt.ylim(-0.05, 1.05)

        safe_name = attack.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(output_dir / f"2_Manipulation_{safe_name}_vs_Baseline.pdf", bbox_inches='tight')
        plt.close()


def plot_victim_isolation(df, output_dir):
    """
    SHOWS: Specific Victim ID (Did they get hit?)
    """
    attacks = [a for a in ATTACK_CATEGORIES["isolation"] if a in df['attack'].unique()]
    if not attacks: return

    print(f"--- Plotting Isolation (Victim Focus) ---")

    # We iterate through defenses to show if they succumbed
    isolation_df = df[df['attack'].isin(attacks)].copy()
    isolation_df['is_victim'] = isolation_df['seller_id'].apply(lambda x: str(x) == str(TARGET_VICTIM_ID))
    isolation_df['Status'] = isolation_df['is_victim'].map({True: 'Victim', False: 'Others'})

    # Sort for visual consistency
    isolation_df = isolation_df.sort_values(by=['seller_id'])

    # Create a FacetGrid to show FLTrust side-by-side with MARTFL
    g = sns.catplot(
        data=isolation_df, x="seller_id", y="selection_rate",
        col="defense", col_order=DEFENSE_ORDER,
        hue="Status", palette={'Victim': '#e74c3c', 'Others': '#95a5a6'},
        kind="bar", height=5, aspect=1.2,
        dodge=False, edgecolor="black"
    )

    g.fig.suptitle("Targeted Pivot: Victim Isolation per Defense", y=1.05, fontsize=18, fontweight='bold')
    g.set_axis_labels("Seller ID", "Selection Rate")
    g.set_xticklabels(rotation=45, ha='right', fontsize=9)
    g.set(ylim=(0, 1.05))

    plt.savefig(output_dir / "3_Isolation_VictimCheck.pdf", bbox_inches='tight')
    plt.close()


# ==========================================
# MAIN
# ==========================================

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found or no matching defenses.")
        return

    plot_disruption_impact(df, output_dir)
    plot_manipulation_fairness(df, output_dir)
    plot_victim_isolation(df, output_dir)

    print(f"\nFigures saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
