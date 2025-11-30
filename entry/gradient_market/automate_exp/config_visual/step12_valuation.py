import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_dual_stack_fixed"

TARGET_DATASET = "CIFAR-100"

METRICS_TO_PLOT = [
    "selection_rate",             # Participation (Dense - All Rounds)
    "marginal_contrib_loo",       # Economic Value (Sparse - Every 10 Rounds)
    "kernelshap_score",           # Economic Value (Sparse)
    "influence_score"             # Economic Value (Sparse)
]

# --- COLORS ---
# Benign Stack
COLOR_BENIGN_PAID = "#2ca02c"      # Green
COLOR_BENIGN_LOST = "#bbbbbb"      # Grey (Collateral Damage)

# Adversary Stack
COLOR_ADV_PAID = "#d62728"         # Red (Security Failure)
COLOR_ADV_CAUGHT = "#2c3e50"       # Dark Blue/Black (Successful Defense)

PRETTY_NAMES = {
    "fedavg": "FedAvg", "fltrust": "FLTrust",
    "martfl": "MARTFL", "skymask": "SkyMask",
    "skymask_small": "SkyMask"
}
DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def format_label(label: str) -> str:
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.weight': 'bold',
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'legend.fontsize': 13, 'figure.figsize': (12, 7),
        'axes.linewidth': 2.0, 'axes.edgecolor': '#333333',
        'pdf.fonttype': 42, 'ps.fonttype': 42
    })

# ==========================================
# 2. DATA LOADING (DENSE VS SPARSE LOGIC)
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        parts = scenario_name.split('_')
        if 'step12' in parts and 'main' in parts:
            idx = parts.index('summary')
            return {"defense": parts[idx + 1], "dataset": parts[idx + 3]}
    except: pass
    return {"defense": "unknown", "dataset": "unknown"}

def load_dual_breakdown(base_dir: Path, dataset: str, target_metric: str) -> pd.DataFrame:
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning for '{target_metric}'...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)
        if dataset.lower().replace("-", "") not in info['dataset'].lower().replace("-", ""):
            continue

        defense_name = format_label(info['defense'])
        jsonl_files = list(folder.rglob("valuations.jsonl"))

        # Accumulators
        benign_paid = 0.0
        benign_discarded = 0.0
        adv_paid = 0.0
        adv_discarded = 0.0
        round_count = 0

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f: lines = f.readlines()
                # Use last 50% for stability
                start_idx = max(0, int(len(lines) * 0.5))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    # --- MODE A: DENSE (Selection Rate) ---
                    # Process EVERY round to capture true filtering behavior
                    if target_metric == "selection_rate":
                        has_metric = True

                    # --- MODE B: SPARSE (Economic Metrics) ---
                    # Only process rounds where the expensive metric exists
                    else:
                        if not valuations: continue
                        first_val = next(iter(valuations.values()))
                        has_metric = target_metric in first_val

                    if not has_metric: continue

                    for sid, data in valuations.items():
                        is_adv = str(sid).startswith('adv')

                        # Get Value
                        if target_metric == "selection_rate":
                            # Count 1.0 for every round (Participation)
                            val = 1.0
                        else:
                            if target_metric in data and data[target_metric] is not None:
                                val = max(0, float(data[target_metric]))
                            else:
                                continue

                        # Distribute to Buckets
                        if is_adv:
                            if sid in selected_ids: adv_paid += val
                            else: adv_discarded += val
                        else:
                            if sid in selected_ids: benign_paid += val
                            else: benign_discarded += val

                    round_count += 1
            except: pass

        if round_count > 0:
            print(f"  -> {defense_name}: Processed {round_count} rounds")
            records.append({
                "defense": defense_name,
                "Benign_Paid": benign_paid,
                "Benign_Discarded": benign_discarded,
                "Adv_Paid": adv_paid,
                "Adv_Discarded": adv_discarded
            })

    return pd.DataFrame(records)

# ==========================================
# 3. DUAL STACK PLOTTING
# ==========================================

def plot_dual_stack(df: pd.DataFrame, metric_name: str, output_dir: Path):
    if df.empty: return

    # Normalize to Percentages within groups
    df['Total_Benign'] = df['Benign_Paid'] + df['Benign_Discarded']
    df['Total_Adv'] = df['Adv_Paid'] + df['Adv_Discarded']

    # Avoid div by zero
    df['Total_Benign'] = df['Total_Benign'].replace(0, 1)
    df['Total_Adv'] = df['Total_Adv'].replace(0, 1)

    df['Pct_Benign_Paid'] = (df['Benign_Paid'] / df['Total_Benign']) * 100
    df['Pct_Benign_Discarded'] = (df['Benign_Discarded'] / df['Total_Benign']) * 100
    df['Pct_Adv_Paid'] = (df['Adv_Paid'] / df['Total_Adv']) * 100
    df['Pct_Adv_Discarded'] = (df['Adv_Discarded'] / df['Total_Adv']) * 100

    # Filtering Order
    df = df.set_index("defense")
    existing_order = [d for d in DEFENSE_ORDER if d in df.index]
    df = df.loc[existing_order]

    # SETUP PLOT
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(df))
    width = 0.35  # Width of bars

    # --- BAR GROUP 1: BENIGN (Left) ---
    p1 = ax.bar(x - width/2, df['Pct_Benign_Paid'], width, label='Benign: Paid',
                color=COLOR_BENIGN_PAID, edgecolor='black')
    p2 = ax.bar(x - width/2, df['Pct_Benign_Discarded'], width, bottom=df['Pct_Benign_Paid'],
                label='Benign: Discarded', color=COLOR_BENIGN_LOST, edgecolor='black', hatch='//')

    # --- BAR GROUP 2: ADVERSARY (Right) ---
    p3 = ax.bar(x + width/2, df['Pct_Adv_Paid'], width, label='Adversary: Paid (Leak)',
                color=COLOR_ADV_PAID, edgecolor='black')
    p4 = ax.bar(x + width/2, df['Pct_Adv_Discarded'], width, bottom=df['Pct_Adv_Paid'],
                label='Adversary: Caught', color=COLOR_ADV_CAUGHT, edgecolor='black', hatch='..')

    # --- LABELS ---
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 5:  # Only label visible segments
                ax.annotate(f'{height:.0f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2),
                            xytext=(0, 0), textcoords="offset points",
                            ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    add_labels(p1)
    add_labels(p2)
    add_labels(p3)
    add_labels(p4)

    # Styling
    clean_title = metric_name.replace("_", " ").title().replace("Loo", "LOO")
    if "Selection" in clean_title: clean_title = "Participation Rate"
    if "Kernelshap" in clean_title: clean_title = "Shapley Value"

    ax.set_ylabel('Percentage of Group Value (%)')
    ax.set_title(f'Filtering Impact on {clean_title} (Split View)')
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylim(0, 105) # Headroom for legend

    # Custom Legend
    ax.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=2, frameon=False)

    plt.tight_layout()
    filename = f"Fig_DualStack_{metric_name}.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {save_path}")
    plt.close()

# ==========================================
# 4. MAIN
# ==========================================
def main():
    print("--- Starting Dual Stack Analysis (Fixed) ---")
    set_publication_style()
    base_dir = Path(BASE_RESULTS_DIR)
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    for metric in METRICS_TO_PLOT:
        df = load_dual_breakdown(base_dir, TARGET_DATASET, metric)
        if not df.empty:
            plot_dual_stack(df, metric, output_dir)
        else:
            print(f"  -> Skipping {metric} (No data)")

if __name__ == "__main__":
    main()