import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./step9_figures"
VICTIM_SELLER_ID = "bn_3"  # The seller being mimicked


# ---------------------


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses the HP suffix folder name (e.g., 'adv_rate_0.1')
    """
    hps = {}
    pattern = r'adv_rate_([0-9\.]+)'
    match = re.search(pattern, hp_folder_name)

    if match:
        hps['adv_rate'] = float(match.group(1))
    else:
        print(f"Warning: Could not parse HP suffix '{hp_folder_name}'")
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name (e.g., 'step9_comp_mimicry_noisy_copy_martfl_CIFAR100')"""
    try:
        pattern = r'step9_comp_mimicry_([a-z_]+)_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario": scenario_name,
                "strategy": match.group(1),
                "defense": match.group(2),
                "dataset": match.group(3),
            }
        else:
            raise ValueError(f"Pattern not matched for: {scenario_name}")

    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path, victim_id: str) -> Dict[str, Any]:
    """
    (UPDATED)
    Loads key data and separates seller rates AND rewards into:
    1. Attacker (adv_...)
    2. Victim (victim_id)
    3. Other Benign (other bn_...)
    """
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['asr'] = metrics.get('asr', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)

            sellers = report.get('seller_summaries', {})

            adv_rates, adv_rewards = [], []
            victim_rates, victim_rewards = [], []
            other_benign_rates, other_benign_rewards = [], []

            for seller_id, seller_data in sellers.items():
                rate = seller_data.get('selection_rate', 0.0)
                reward = seller_data.get('total_reward', 0.0)

                if seller_id.startswith('adv'):
                    adv_rates.append(rate)
                    adv_rewards.append(reward)
                elif seller_id == victim_id:
                    victim_rates.append(rate)
                    victim_rewards.append(reward)
                elif seller_id.startswith('bn_'):
                    other_benign_rates.append(rate)
                    other_benign_rewards.append(reward)

            # Calculate mean for each group
            run_data['adv_selection_rate'] = np.mean(adv_rates) if adv_rates else 0.0
            run_data['adv_total_reward'] = np.mean(adv_rewards) if adv_rewards else 0.0

            run_data['victim_selection_rate'] = np.mean(victim_rates) if victim_rates else 0.0
            run_data['victim_total_reward'] = np.mean(victim_rewards) if victim_rewards else 0.0

            run_data['other_benign_selection_rate'] = np.mean(other_benign_rates) if other_benign_rates else 0.0
            run_data['other_benign_total_reward'] = np.mean(other_benign_rewards) if other_benign_rates else 0.0

        return run_data

    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step9_comp_mimicry_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step9_comp_mimicry_*' directories found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts:
                    continue

                hp_folder_name = relative_parts[0]

                run_hps = parse_hp_suffix(hp_folder_name)
                run_metrics = load_run_data(metrics_file, victim_id=VICTIM_SELLER_ID)

                if run_metrics:
                    all_runs.append({
                        **run_scenario,
                        **run_hps,
                        **run_metrics
                    })
            except Exception as e:
                print(f"Error processing file {metrics_file} under scenario {scenario_name}: {e}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    return df


def plot_mimicry_comparison(df: pd.DataFrame, output_dir: Path):
    """
    (UPDATED)
    Generates the 3x2 multi-panel line plot for mimicry effectiveness
    and economic impact.
    """
    print(f"\n--- Plotting Competitor Mimicry Effectiveness & Economic Impact ---")

    # --- UPDATED METRIC LIST ---
    metrics_to_plot = [
        'acc',
        'adv_selection_rate',
        'adv_total_reward',
        'victim_selection_rate',
        'victim_total_reward',
        'other_benign_selection_rate'
    ]
    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]

    if not metrics_to_plot:
        print("No metrics found to plot.")
        return

    plot_df = df.melt(
        id_vars=['adv_rate', 'defense'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    # Rename for clearer plot labels
    plot_df['Metric'] = plot_df['Metric'].replace({
        'acc': '1. Model Accuracy (Utility)',
        'adv_selection_rate': '2. Attacker Selection (%)',
        'adv_total_reward': '3. Attacker Gain (Total Reward)',
        'victim_selection_rate': f'4. Victim ({VICTIM_SELLER_ID}) Selection (%)',
        'victim_total_reward': f'5. Victim ({VICTIM_SELLER_ID}) Revenue (Total Reward)',
        'other_benign_selection_rate': '6. Other Benign Selection (%)'
    })
    # Sort the metrics by the numeric prefix
    plot_df = plot_df.sort_values(by='Metric')

    g = sns.relplot(
        data=plot_df,
        x='adv_rate',
        y='Value',
        hue='defense',
        style='defense',
        col='Metric',
        kind='line',
        col_wrap=2,  # 2 columns
        height=4,
        aspect=1.2,
        facet_kws={'sharey': False},
        markers=True,
        dashes=False
    )

    g.fig.suptitle('Defense Robustness vs. Competitor Mimicry (Impersonation) Attack', y=1.05)
    g.set_axis_labels("Adversary Rate (Mimics)", "Value")
    g.set_titles(col_template="{col_name}")

    plot_file = output_dir / "plot_mimicry_effectiveness.png"
    g.fig.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results found. Exiting.")
        return

    # --- THIS IS THE NEW LINE ---
    csv_output_file = output_dir / "step9_mimicry_results_summary.csv"
    df.to_csv(csv_output_file, index=False, float_format="%.4f")
    print(f"\n✅ Successfully saved full analysis data to: {csv_output_file}\n")
    # ----------------------------

    plot_mimicry_comparison(df, output_dir)

    print("\nAnalysis complete. Check 'step9_figures' folder for plots and CSV.")


if __name__ == "__main__":
    main()