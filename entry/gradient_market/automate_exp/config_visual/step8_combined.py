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
TARGET_VICTIM_ID = "bn_5"

# --- UPDATED CATEGORIES TO MATCH YOUR DATA ---
ATTACK_CATEGORIES = {
    # Attacks that break the model or create chaos
    "disruption": [
        "DoS",
        "Trust Erosion",
        "Oscillating (Binary)",
        "Oscillating (Random)",
        "Oscillating (Drift)"
    ],
    # Attacks that rig the market rewards
    "manipulation": [
        "Starvation",
        "Class Exclusion (Neg)",
        "Class Exclusion (Pos)"
    ],
    # Attacks that target specific victims
    "isolation": [
        "Pivot"  # This will map from 'orthogonal_pivot_legacy'
    ]
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

# ==========================================
# 2. DATA LOADING
# ==========================================

def format_label(label: str) -> str:
    """
    Maps raw folder names to clean publication labels.
    Crucial for matching the ATTACK_CATEGORIES above.
    """
    label = label.lower()
    mapping = {
        # --- Disruption ---
        "dos": "DoS",
        "erosion": "Trust Erosion",
        "oscillating_binary": "Oscillating (Binary)",
        "oscillating_random": "Oscillating (Random)",
        "oscillating_drift": "Oscillating (Drift)",

        # --- Manipulation ---
        "starvation": "Starvation",
        "class_exclusion_neg": "Class Exclusion (Neg)",
        "class_exclusion_pos": "Class Exclusion (Pos)",

        # --- Isolation ---
        "orthogonal_pivot_legacy": "Pivot", # <--- FIXES YOUR ERROR
        "pivot": "Pivot",

        # --- Baseline ---
        "0. baseline": "Healthy Baseline"
    }

    # Return mapped value, or fallback to Title Case if not found
    return mapping.get(label, label.replace("_", " ").title())

def parse_scenario(scenario_name: str):
    # Matches: step8_buyer_attack_[ATTACK_NAME]_[DEFENSE]_[DATASET]
    # Also matches: step7_baseline_no_attack_[DEFENSE]_[DATASET]
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
    # Grab both Step 7 (Baseline) and Step 8 (Attacks)
    folders = list(base_path.glob("step8_buyer_attack_*")) + list(base_path.glob("step7_baseline_no_attack_*"))

    print(f"Scanning {len(folders)} folders...")

    for path in folders:
        meta = parse_scenario(path.name)
        if not meta: continue

        # Look for metrics file
        for mfile in path.rglob("final_metrics.json"):
            try:
                # 1. Global Metrics (Accuracy)
                with open(mfile) as f: metrics = json.load(f)
                acc = metrics.get('acc', 0)
                if acc > 1.0: acc /= 100.0

                # 2. Seller Selection Rates
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
    """ Visual 1: Bar chart of Model Utility (Accuracy) """
    # Filter for attacks in our Disruption category that actually exist in the DF
    attacks = [a for a in ATTACK_CATEGORIES["disruption"] if a in df['attack'].unique()]
    if not attacks: return

    print(f"--- Generating Disruption Plot for {attacks} ---")

    subset = df[df['attack'].isin(attacks + ["Healthy Baseline"])].copy()
    subset = subset.drop_duplicates(subset=['attack', 'defense', 'acc'])

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=subset, x="defense", y="acc", hue="attack",
        order=DEFENSE_ORDER, palette="magma",
        edgecolor="black", linewidth=1.5
    )
    plt.title("Service Disruption: Impact on Model Utility", pad=15)
    plt.ylabel("Global Test Accuracy")
    plt.xlabel("Defense Mechanism")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.0)
    plt.savefig(output_dir / "Visual_1_Disruption_Utility.pdf", bbox_inches='tight')
    plt.close()

def plot_manipulation_fairness(df, output_dir):
    """ Visual 2: Split Boxplots showing preferred vs starved sellers """
    attacks = [a for a in ATTACK_CATEGORIES["manipulation"] if a in df['attack'].unique()]
    if not attacks: return

    print(f"--- Generating Manipulation Plots for {attacks} ---")

    subset = df[df['attack'].isin(attacks)].copy()

    # We create one plot per attack type to keep it clean
    for attack in attacks:
        attack_sub = subset[subset['attack'] == attack]
        if attack_sub.empty: continue

        plt.figure(figsize=(10, 6))

        # Boxplot for summary stats
        sns.boxplot(
            data=attack_sub, x="defense", y="selection_rate",
            order=DEFENSE_ORDER, color="white",
            linewidth=2, fliersize=0
        )
        # Stripplot to show the 'split'
        sns.stripplot(
            data=attack_sub, x="defense", y="selection_rate",
            order=DEFENSE_ORDER,
            color="#e74c3c", alpha=0.6, jitter=0.25, size=5
        )

        plt.title(f"Market Manipulation: {attack}", pad=15)
        plt.ylabel("Seller Selection Rate")
        plt.xlabel("Defense Mechanism")
        plt.ylim(-0.05, 1.05)

        safe_name = attack.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(output_dir / f"Visual_2_Manipulation_{safe_name}.pdf", bbox_inches='tight')
        plt.close()

def plot_victim_isolation(df, output_dir):
    """ Visual 3: Victim vs Others bar chart """
    # Check for attacks mapping to 'isolation'
    attacks = [a for a in ATTACK_CATEGORIES["isolation"] if a in df['attack'].unique()]

    if not attacks:
        print(f"⚠️ Skipping Isolation Plot: No attacks found matching {ATTACK_CATEGORIES['isolation']}")
        print(f"   (Available attacks: {df['attack'].unique()})")
        return

    print(f"--- Generating Isolation Plot for {attacks} ---")

    isolation_df = df[df['attack'].isin(attacks)].copy()

    # DYNAMIC DEFENSE SELECTION: Use SkyMask if avail, else FedAvg
    avail_defenses = isolation_df['defense'].unique()
    target_defense = "SkyMask" if "SkyMask" in avail_defenses else avail_defenses[0]

    subset = isolation_df[isolation_df['defense'] == target_defense].copy()

    # Sort so Victim is distinct
    subset['is_victim'] = subset['seller_id'].apply(lambda x: str(x) == str(TARGET_VICTIM_ID))
    subset = subset.sort_values(by=['seller_id'])
    subset['Status'] = subset['is_victim'].map({True: 'Targeted Victim', False: 'Other Sellers'})

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=subset, x="seller_id", y="selection_rate", hue="Status",
        palette={'Targeted Victim': '#c0392b', 'Other Sellers': '#bdc3c7'},
        dodge=False, edgecolor="black"
    )

    plt.title(f"Targeted Censorship: Victim Isolation (Defense: {target_defense})\nAttack: {attacks[0]}", pad=15)
    plt.ylabel("Selection Rate")
    plt.xlabel("Seller ID")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')

    # Add Arrow
    victim_x = subset.reset_index().index[subset['is_victim']].tolist()
    if victim_x:
        idx = victim_x[0]
        plt.annotate('Forced to 0', xy=(idx, 0.05), xytext=(idx, 0.3),
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

    df = load_data(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found.")
        return

    # Generate Visuals
    plot_disruption_impact(df, output_dir)
    plot_manipulation_fairness(df, output_dir)
    plot_victim_isolation(df, output_dir)

    print(f"\n✅ Analysis Complete. Figures saved to {output_dir.resolve()}")

if __name__ == "__main__":
    main()