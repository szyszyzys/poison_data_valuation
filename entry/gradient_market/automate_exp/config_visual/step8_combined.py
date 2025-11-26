import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step8_attack_effectiveness"
TARGET_VICTIM_ID = "bn_5"  # The ID of the seller targeted by Pivot

# Configuration to map attacks to their correct visual story
ATTACK_CATEGORIES = {
    "disruption": ["DoS", "Trust Erosion", "Oscillating", "BadNet", "DBA"],
    "manipulation": ["Starvation", "Class Exclusion"],
    "isolation": ["Pivot", "Targeted Pivot"]
}

# Style Config
sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=1.1)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.figsize': (10, 6)
})

DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]
DEFENSE_COLORS = {"FedAvg": "#95a5a6", "FLTrust": "#3498db", "MARTFL": "#2ecc71", "SkyMask": "#e74c3c"}

# ==========================================
# 2. DATA LOADING
# ==========================================

def format_label(label: str) -> str:
    # Quick mapping for cleaner names
    param_map = {
        "fedavg": "FedAvg", "fltrust": "FLTrust", "martfl": "MARTFL",
        "skymask": "SkyMask", "skymask_small": "SkyMask",
        "dos": "DoS", "erosion": "Trust Erosion",
        "starvation": "Starvation", "class_exclusion": "Class Exclusion",
        "orthogonal_pivot": "Pivot", "pivot": "Pivot",
        "oscillating": "Oscillating", "0. Baseline": "Healthy Baseline"
    }
    return param_map.get(label.lower(), label.replace("_", " ").title())

def parse_scenario(scenario_name: str):
    # Regex to extract metadata from folder names
    pattern = r'(step[78])_(baseline_no_attack|buyer_attack)_(?:(.+?)_)?(fedavg|martfl|fltrust|skymask|skymask_small)_(.*)'
    match = re.search(pattern, scenario_name)
    if match:
        _, mode, attack_raw, defense, dataset = match.groups()
        attack = "0. Baseline" if "baseline" in mode else attack_raw
        return {
            "scenario": scenario_name,
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

        # Load Metrics (Accuracy)
        for mfile in path.rglob("final_metrics.json"):
            try:
                with open(mfile) as f: metrics = json.load(f)
                acc = metrics.get('acc', 0)
                if acc > 1.0: acc /= 100.0

                # Load Seller Reports (Selection Rates)
                report_file = mfile.parent / "marketplace_report.json"
                if report_file.exists():
                    with open(report_file) as rf: report = json.load(rf)
                    for sid, sdata in report.get('seller_summaries', {}).items():
                        if sdata.get('type') == 'benign':
                            records.append({
                                **meta,
                                "acc": acc,
                                "seller_id": sid,
                                "selection_rate": sdata.get('selection_rate', 0.0)
                            })
            except Exception as e: pass

    return pd.DataFrame(records)

# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================

def plot_disruption_impact(df, output_dir):
    """
    VISUAL 1: The 'Utility Collapse' (Bar Chart)
    Shows how DoS and Erosion destroy model accuracy compared to Baseline.
    """
    attacks = [a for a in ATTACK_CATEGORIES["disruption"] if a in df['attack'].unique()]
    if not attacks: return

    print(f"--- Generating Disruption Plot for {attacks} ---")

    # Filter Data
    subset = df[df['attack'].isin(attacks + ["Healthy Baseline"])].copy()

    # We only need one row per scenario (accuracy is global, not per seller)
    subset = subset.drop_duplicates(subset=['attack', 'defense', 'acc'])

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=subset, x="defense", y="acc", hue="attack",
        order=DEFENSE_ORDER,
        palette="magma", # Dark colors for severe attacks
        edgecolor="black", linewidth=1.5
    )

    plt.title("Service Disruption: Impact on Model Utility", pad=15)
    plt.ylabel("Global Test Accuracy")
    plt.xlabel("Defense Mechanism")
    plt.legend(title="Attack Scenario", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.0)

    plt.savefig(output_dir / "Visual_1_Disruption_Utility.pdf", bbox_inches='tight')
    plt.close()

def plot_manipulation_fairness(df, output_dir):
    """
    VISUAL 2: The 'Bifurcation' (Stripplot + Boxplot)
    Shows how Starvation splits honest sellers into 'Winners' and 'Losers'.
    """
    attacks = [a for a in ATTACK_CATEGORIES["manipulation"] if a in df['attack'].unique()]
    if not attacks: return

    print(f"--- Generating Manipulation Plot for {attacks} ---")

    subset = df[df['attack'].isin(attacks)].copy()

    for attack in attacks:
        attack_sub = subset[subset['attack'] == attack]

        plt.figure(figsize=(10, 6))

        # 1. Boxplot shows the overall statistics
        sns.boxplot(
            data=attack_sub, x="defense", y="selection_rate",
            order=DEFENSE_ORDER, color="white",
            linewidth=2, fliersize=0 # Hide outliers in box, show them in strip
        )

        # 2. Stripplot shows the individual sellers (The Split)
        # jitter=True spreads them out so you can see the two groups
        sns.stripplot(
            data=attack_sub, x="defense", y="selection_rate",
            order=DEFENSE_ORDER,
            color="#e74c3c", alpha=0.6, jitter=0.2, size=6
        )

        plt.title(f"Market Manipulation: Seller Reward Bifurcation\n({attack})", pad=15)
        plt.ylabel("Seller Selection Rate")
        plt.xlabel("Defense Mechanism")
        plt.ylim(-0.05, 1.05)

        # Add annotation explaining the visual
        plt.text(0.02, 0.05, "Points = Individual Honest Sellers.\nNote the split between 'Preferred' and 'Starved'.",
                 transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        plt.savefig(output_dir / f"Visual_2_Manipulation_{attack.replace(' ','')}.pdf", bbox_inches='tight')
        plt.close()

def plot_victim_isolation(df, output_dir):
    """
    VISUAL 3: The 'Broken Tooth' (Bar Chart)
    Shows specific exclusion of the victim ID vs others.
    """
    attacks = [a for a in ATTACK_CATEGORIES["isolation"] if a in df['attack'].unique()]
    if not attacks: return

    print(f"--- Generating Isolation Plot for {attacks} ---")

    # Pick the most relevant scenario (e.g., SkyMask defense) to show the attack working
    # We choose SkyMask or FedAvg as representative examples
    target_defense = "SkyMask"

    subset = df[
        (df['attack'].isin(attacks)) &
        (df['defense'] == target_defense)
    ].copy()

    if subset.empty: return

    # Sort sellers to put Victim in the middle or highlight them
    subset['is_victim'] = subset['seller_id'].apply(lambda x: x == TARGET_VICTIM_ID)
    subset = subset.sort_values(by=['seller_id'])

    # We create a new column for color mapping
    subset['Status'] = subset['is_victim'].map({True: 'Targeted Victim', False: 'Other Sellers'})

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=subset, x="seller_id", y="selection_rate", hue="Status",
        palette={'Targeted Victim': '#c0392b', 'Other Sellers': '#bdc3c7'},
        dodge=False, edgecolor="black"
    )

    plt.title(f"Targeted Censorship: Victim Isolation (Defense: {target_defense})", pad=15)
    plt.ylabel("Selection Rate")
    plt.xlabel("Seller ID")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')

    # Arrow annotation pointing to the victim
    victim_x = subset.reset_index().index[subset['is_victim']].tolist()
    if victim_x:
        idx = victim_x[0]
        plt.annotate('Selection Forced to 0', xy=(idx, 0.05), xytext=(idx, 0.3),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     ha='center', color='black', fontweight='bold')

    plt.savefig(output_dir / "Visual_3_Targeted_Isolation.pdf", bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Everything
    df = load_data(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found.")
        return

    # 2. Generate Story-Specific Visuals
    plot_disruption_impact(df, output_dir)
    plot_manipulation_fairness(df, output_dir)
    plot_victim_isolation(df, output_dir)

    print(f"\n✅ Analysis Complete. Figures saved to {output_dir.resolve()}")

if __name__ == "__main__":
    main()