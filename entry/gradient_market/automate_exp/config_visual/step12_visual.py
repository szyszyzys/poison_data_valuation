import json
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_main_summary"


# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    Parses Step 12 folder names.
    Example: step12_main_summary_martfl_image_CIFAR100_cnn
    """
    try:
        parts = scenario_name.split('_')
        # Basic heuristic based on your naming convention
        if 'step12' in parts:
            defense = parts[3]  # martfl
            modality = parts[4]  # image
            dataset = parts[5]  # CIFAR100
            return {
                "scenario": scenario_name,
                "defense": defense,
                "modality": modality,
                "dataset": dataset
            }
        return {"scenario": scenario_name, "defense": "unknown"}
    except Exception:
        return {"scenario": scenario_name, "defense": "unknown"}


def load_valuation_data_from_csv(run_dir: Path) -> pd.DataFrame:
    """
    Scans seller_metrics.csv for valuation columns.
    Includes error handling for corrupted CSV lines.
    """
    csv_path = run_dir / "seller_metrics.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        # FIX: on_bad_lines='skip' tells pandas to ignore rows with formatting errors
        # instead of crashing the entire script.
        df = pd.read_csv(csv_path, on_bad_lines='skip')

        # Check if we have data
        if df.empty or 'seller_id' not in df.columns:
            return pd.DataFrame()

        # 1. Identify Seller Type
        df['type'] = df['seller_id'].apply(
            lambda x: 'Adversary' if str(x).startswith('adv') else 'Benign'
        )

        # 2. Identify Valuation Columns dynamically
        # Matches: influence_score, shapley_value, loo_score, kernelshap_score, etc.
        val_cols = [c for c in df.columns if any(x in c.lower() for x in ['influence', 'shap', 'loo'])]

        if not val_cols:
            return pd.DataFrame()

        # 3. Aggregate Average Score per Type
        summary = df.groupby('type')[val_cols].mean().reset_index()
        return summary

    except Exception as e:
        print(f"⚠️ Warning: Could not read CSV {csv_path}: {e}")
        return pd.DataFrame()


def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)

    scenario_folders = [f for f in base_path.glob("step12_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenarios.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            run_dir = metrics_file.parent

            # 1. Load Standard Metrics
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                acc = metrics.get('acc', 0)
            except:
                acc = 0

            # 2. Load Valuation Data directly from CSV
            df_val = load_valuation_data_from_csv(run_dir)

            if not df_val.empty:
                # Flatten the valuation data into the main record
                # e.g. Benign_influence_score, Adversary_influence_score
                val_record = {}
                for _, row in df_val.iterrows():
                    sType = row['type']
                    for col in df_val.columns:
                        if col != 'type':
                            val_record[f"{sType}_{col}"] = row[col]

                all_runs.append({
                    **run_scenario,
                    "acc": acc,
                    **val_record
                })

    return pd.DataFrame(all_runs)


def plot_valuation_fairness(df: pd.DataFrame, output_dir: Path):
    """
    Plots Benign vs Adversary scores for every valuation metric found.
    """
    print("\n--- Plotting Valuation Fairness ---")

    # Find all valuation metrics present in the DF (e.g., 'influence_score', 'kernelshap')
    # We look for columns ending in standard metric names
    possible_metrics = set()
    for col in df.columns:
        if 'Adversary_' in col or 'Benign_' in col:
            metric_name = col.replace('Adversary_', '').replace('Benign_', '')
            possible_metrics.add(metric_name)

    if not possible_metrics:
        print("No valuation metrics found in processed data.")
        return

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    for metric in possible_metrics:
        print(f"  Plotting Metric: {metric}")

        # Prepare data for seaborn (Melt)
        adv_col = f"Adversary_{metric}"
        ben_col = f"Benign_{metric}"

        if adv_col not in df.columns or ben_col not in df.columns:
            continue

        melted = df.melt(
            id_vars=['defense', 'dataset'],
            value_vars=[adv_col, ben_col],
            var_name='Seller Type',
            value_name='Score'
        )

        # Clean up names for legend
        melted['Seller Type'] = melted['Seller Type'].map({
            adv_col: 'Adversary',
            ben_col: 'Benign'
        })

        # Plot per dataset
        for dataset in melted['dataset'].unique():
            plot_data = melted[melted['dataset'] == dataset]
            if plot_data.empty: continue

            plt.figure(figsize=(8, 5))
            sns.barplot(
                data=plot_data,
                x='defense',
                y='Score',
                hue='Seller Type',
                order=[d for d in defense_order if d in plot_data['defense'].unique()],
                palette={'Benign': 'blue', 'Adversary': 'red'}
            )

            plt.title(f"Valuation Analysis: {metric}\nDataset: {dataset}")
            plt.ylabel(f"Avg. {metric} (Higher is Better)")
            plt.axhline(0, color='black', linewidth=0.8)

            # Save
            safe_metric = metric.replace('_', '')
            fname = output_dir / f"plot_VALUATION_{safe_metric}_{dataset}.pdf"
            plt.savefig(fname, bbox_inches='tight')
            plt.close()


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data found.")
        return

    # Save raw summaries
    df.to_csv(output_dir / "step12_valuation_summary.csv", index=False)
    print(f"Saved summary CSV to {output_dir}")

    plot_valuation_fairness(df, output_dir)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
