import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_processed_data(summary_csv, all_rounds_csv):
    """
    Load the processed summary and round-by-round data.
    """
    summary_df = pd.read_csv(summary_csv)
    all_rounds_df = pd.read_csv(all_rounds_csv)

    # Convert boolean columns
    if 'IS_SYBIL' in summary_df.columns:
        summary_df['IS_SYBIL'] = summary_df['IS_SYBIL'].astype(bool)
    if 'IS_SYBIL' in all_rounds_df.columns:
        all_rounds_df['IS_SYBIL'] = all_rounds_df['IS_SYBIL'].astype(bool)

    return summary_df, all_rounds_df


def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)


def plot_attack_success_comparison(summary_df, output_dir):
    """
    Plot comparisons of Attack Success Rate (ASR) across different parameters.
    """
    plt.figure(figsize=(20, 16))

    # 1. ASR by Gradient Manipulation Method
    plt.subplot(2, 2, 1)
    sns.barplot(x='GRAD_MODE', y='FINAL_ASR', data=summary_df, errorbar=None)
    plt.title('Attack Success Rate by Gradient Method', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. ASR by Trigger Rate
    plt.subplot(2, 2, 2)
    sns.barplot(x='TRIGGER_RATE', y='FINAL_ASR', data=summary_df, errorbar=None)
    plt.title('Attack Success Rate by Trigger Rate', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. ASR by Number of Adversaries
    plt.subplot(2, 2, 3)
    sns.barplot(x='N_ADV', y='FINAL_ASR', data=summary_df, errorbar=None)
    plt.title('Attack Success Rate by Number of Adversaries', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. ASR with and without Sybil Attack
    plt.subplot(2, 2, 4)
    sns.barplot(x='IS_SYBIL', y='FINAL_ASR', data=summary_df, errorbar=None)
    plt.title('Attack Success Rate with/without Sybil Attack', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Sybil Attack Enabled')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/asr_comparison.png", dpi=300)
    plt.close()


def plot_parameter_interactions(summary_df, output_dir):
    """
    Plot interactions between different attack parameters.
    """
    plt.figure(figsize=(20, 16))

    # 1. ASR by Poison Strength for CMD mode
    plt.subplot(2, 2, 1)
    cmd_data = summary_df[summary_df['GRAD_MODE'] == 'cmd']
    if not cmd_data.empty:
        sns.barplot(x='POISON_STRENGTH', y='FINAL_ASR', data=cmd_data, errorbar=None)
        plt.title('ASR by Poison Strength (CMD Mode)', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. ASR by Trigger Rate and Gradient Mode
    plt.subplot(2, 2, 2)
    sns.barplot(x='TRIGGER_RATE', y='FINAL_ASR', hue='GRAD_MODE', data=summary_df, errorbar=None)
    plt.title('ASR by Trigger Rate and Gradient Mode', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. ASR by Number of Adversaries and Sybil Mode
    plt.subplot(2, 2, 3)
    sns.barplot(x='N_ADV', y='FINAL_ASR', hue='IS_SYBIL', data=summary_df, errorbar=None)
    plt.title('ASR by Number of Adversaries and Sybil Mode', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. ASR by Sybil Mode and Gradient Mode
    plt.subplot(2, 2, 4)
    sns.barplot(x='IS_SYBIL', y='FINAL_ASR', hue='GRAD_MODE', data=summary_df, errorbar=None)
    plt.title('ASR by Sybil Mode and Gradient Mode', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Sybil Attack Enabled')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_interactions.png", dpi=300)
    plt.close()


def plot_tradeoff_analysis(summary_df, output_dir):
    """
    Plot trade-off between attack success and model accuracy.
    """
    plt.figure(figsize=(20, 10))

    # 1. ASR vs Main Task Accuracy by Gradient Mode
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x='FINAL_MAIN_ACC',
        y='FINAL_ASR',
        hue='GRAD_MODE',
        size='TRIGGER_RATE',
        data=summary_df,
        sizes=(50, 200),
        alpha=0.7
    )
    plt.title('ASR vs Main Task Accuracy by Gradient Mode', fontsize=14)
    plt.xlim(0.5, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(linestyle='--', alpha=0.7)

    # 2. ASR vs Main Task Accuracy by Sybil Mode
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x='FINAL_MAIN_ACC',
        y='FINAL_ASR',
        hue='IS_SYBIL',
        size='N_ADV',
        data=summary_df,
        sizes=(50, 200),
        alpha=0.7
    )
    plt.title('ASR vs Main Task Accuracy by Sybil Mode', fontsize=14)
    plt.xlim(0.5, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/tradeoff_analysis.png", dpi=300)
    plt.close()


def plot_efficiency_analysis(summary_df, output_dir):
    """
    Plot attack efficiency metrics.
    """
    plt.figure(figsize=(20, 10))

    # 1. ASR per Adversary by Number of Adversaries
    plt.subplot(1, 2, 1)
    sns.barplot(x='N_ADV', y='ASR_PER_ADV', data=summary_df, errorbar=None)
    plt.title('Attack Efficiency (ASR per Adversary)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. ASR per Adversary by Gradient Mode and Sybil Mode
    plt.subplot(1, 2, 2)
    sns.barplot(x='GRAD_MODE', y='ASR_PER_ADV', hue='IS_SYBIL', data=summary_df, errorbar=None)
    plt.title('Attack Efficiency by Gradient Mode and Sybil Mode', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/efficiency_analysis.png", dpi=300)
    plt.close()


def plot_stealth_analysis(summary_df, output_dir):
    """
    Plot stealth metrics to see how well attacks preserve model accuracy.
    """
    plt.figure(figsize=(20, 10))

    # 1. Stealth by Gradient Mode
    plt.subplot(1, 2, 1)
    sns.barplot(x='GRAD_MODE', y='STEALTH', data=summary_df, errorbar=None)
    plt.title('Attack Stealth by Gradient Mode', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Stealth by Trigger Rate and Number of Adversaries
    plt.subplot(1, 2, 2)
    # Create a new column for trigger rate and adversary combinations
    summary_df['TR_ADV'] = summary_df['TRIGGER_RATE'].astype(str) + '_' + summary_df['N_ADV'].astype(str)
    sns.barplot(x='TR_ADV', y='STEALTH', data=summary_df, errorbar=None)
    plt.title('Attack Stealth by Trigger Rate and N_ADV', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/stealth_analysis.png", dpi=300)
    plt.close()


def plot_attack_progression(all_rounds_df, output_dir):
    """
    Plot progression of attack success and model accuracy over rounds.
    """
    # Group data by attack configuration for grad mode comparison
    plt.figure(figsize=(15, 10))

    # For meaningful average trends, group by GRAD_MODE and calculate statistics
    grouped_data = all_rounds_df.groupby(['GRAD_MODE', 'round']).agg({
        'asr': ['mean', 'std'],
        'main_acc': ['mean', 'std']
    }).reset_index()

    # Flatten multi-level columns
    grouped_data.columns = ['_'.join(col).strip('_') for col in grouped_data.columns.values]

    # Plot ASR progression by gradient mode
    plt.subplot(2, 1, 1)
    for grad_mode in all_rounds_df['GRAD_MODE'].unique():
        data = grouped_data[grouped_data['GRAD_MODE'] == grad_mode]

        # Plot mean with shaded std deviation
        plt.plot(data['round'], data['asr_mean'], label=f'{grad_mode}', marker='o', markersize=4)
        plt.fill_between(
            data['round'],
            data['asr_mean'] - data['asr_std'],
            data['asr_mean'] + data['asr_std'],
            alpha=0.2
        )

    plt.title('ASR Progression by Gradient Mode', fontsize=14)
    plt.xlabel('Round')
    plt.ylabel('Attack Success Rate')
    plt.ylim(0, 1.05)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()

    # Plot Main Accuracy progression by gradient mode
    plt.subplot(2, 1, 2)
    for grad_mode in all_rounds_df['GRAD_MODE'].unique():
        data = grouped_data[grouped_data['GRAD_MODE'] == grad_mode]

        plt.plot(data['round'], data['main_acc_mean'], label=f'{grad_mode}', marker='o', markersize=4)
        plt.fill_between(
            data['round'],
            data['main_acc_mean'] - data['main_acc_std'],
            data['main_acc_mean'] + data['main_acc_std'],
            alpha=0.2
        )

    plt.title('Main Task Accuracy Progression by Gradient Mode', fontsize=14)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.05)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/progression_by_grad_mode.png", dpi=300)
    plt.close()

    # Group data by attack configuration for Sybil mode comparison
    plt.figure(figsize=(15, 10))

    # For meaningful average trends, group by IS_SYBIL and calculate statistics
    grouped_data = all_rounds_df.groupby(['IS_SYBIL', 'round']).agg({
        'asr': ['mean', 'std'],
        'main_acc': ['mean', 'std']
    }).reset_index()

    # Flatten multi-level columns
    grouped_data.columns = ['_'.join(col).strip('_') for col in grouped_data.columns.values]

    # Plot ASR progression by Sybil mode
    plt.subplot(2, 1, 1)
    for is_sybil in all_rounds_df['IS_SYBIL'].unique():
        data = grouped_data[grouped_data['IS_SYBIL'] == is_sybil]

        # Plot mean with shaded std deviation
        plt.plot(data['round'], data['asr_mean'], label=f'Sybil={is_sybil}', marker='o', markersize=4)
        plt.fill_between(
            data['round'],
            data['asr_mean'] - data['asr_std'],
            data['asr_mean'] + data['asr_std'],
            alpha=0.2
        )

    plt.title('ASR Progression by Sybil Mode', fontsize=14)
    plt.xlabel('Round')
    plt.ylabel('Attack Success Rate')
    plt.ylim(0, 1.05)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()

    # Plot Main Accuracy progression by Sybil mode
    plt.subplot(2, 1, 2)
    for is_sybil in all_rounds_df['IS_SYBIL'].unique():
        data = grouped_data[grouped_data['IS_SYBIL'] == is_sybil]

        plt.plot(data['round'], data['main_acc_mean'], label=f'Sybil={is_sybil}', marker='o', markersize=4)
        plt.fill_between(
            data['round'],
            data['main_acc_mean'] - data['main_acc_std'],
            data['main_acc_mean'] + data['main_acc_std'],
            alpha=0.2
        )

    plt.title('Main Task Accuracy Progression by Sybil Mode', fontsize=14)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.05)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/progression_by_sybil_mode.png", dpi=300)
    plt.close()

    # Group data by number of adversaries
    plt.figure(figsize=(15, 10))

    # For meaningful average trends, group by N_ADV and calculate statistics
    grouped_data = all_rounds_df.groupby(['N_ADV', 'round']).agg({
        'asr': ['mean', 'std'],
        'main_acc': ['mean', 'std']
    }).reset_index()

    # Flatten multi-level columns
    grouped_data.columns = ['_'.join(col).strip('_') for col in grouped_data.columns.values]

    # Plot ASR progression by number of adversaries
    plt.subplot(2, 1, 1)
    for n_adv in sorted(all_rounds_df['N_ADV'].unique()):
        data = grouped_data[grouped_data['N_ADV'] == n_adv]

        # Plot mean with shaded std deviation
        plt.plot(data['round'], data['asr_mean'], label=f'N_ADV={n_adv}', marker='o', markersize=4)
        plt.fill_between(
            data['round'],
            data['asr_mean'] - data['asr_std'],
            data['asr_mean'] + data['asr_std'],
            alpha=0.2
        )

    plt.title('ASR Progression by Number of Adversaries', fontsize=14)
    plt.xlabel('Round')
    plt.ylabel('Attack Success Rate')
    plt.ylim(0, 1.05)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()

    # Plot Main Accuracy progression by number of adversaries
    plt.subplot(2, 1, 2)
    for n_adv in sorted(all_rounds_df['N_ADV'].unique()):
        data = grouped_data[grouped_data['N_ADV'] == n_adv]

        plt.plot(data['round'], data['main_acc_mean'], label=f'N_ADV={n_adv}', marker='o', markersize=4)
        plt.fill_between(
            data['round'],
            data['main_acc_mean'] - data['main_acc_std'],
            data['main_acc_mean'] + data['main_acc_std'],
            alpha=0.2
        )

    plt.title('Main Task Accuracy Progression by Number of Adversaries', fontsize=14)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.05)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/progression_by_n_adv.png", dpi=300)
    plt.close()

    # Add comparative convergence analysis
    plt.figure(figsize=(20, 10))

    # Group data by configuration to find rounds to reach specific ASR thresholds
    config_data = []

    for config, group in all_rounds_df.groupby(['GRAD_MODE', 'TRIGGER_RATE', 'POISON_STRENGTH', 'IS_SYBIL', 'N_ADV']):
        grad_mode, trigger_rate, poison_strength, is_sybil, n_adv = config

        # Find rounds to reach certain ASR thresholds
        rounds_to_50pct = float('inf')
        rounds_to_75pct = float('inf')
        rounds_to_90pct = float('inf')

        # Sort by round
        sorted_group = group.sort_values('round')

        for _, row in sorted_group.iterrows():
            if row.get('asr', 0) >= 0.5 and rounds_to_50pct == float('inf'):
                rounds_to_50pct = row['round']
            if row.get('asr', 0) >= 0.75 and rounds_to_75pct == float('inf'):
                rounds_to_75pct = row['round']
            if row.get('asr', 0) >= 0.9 and rounds_to_90pct == float('inf'):
                rounds_to_90pct = row['round']

        config_data.append({
            'GRAD_MODE': grad_mode,
            'TRIGGER_RATE': trigger_rate,
            'POISON_STRENGTH': poison_strength,
            'IS_SYBIL': is_sybil,
            'N_ADV': n_adv,
            'ROUNDS_TO_50PCT_ASR': rounds_to_50pct if rounds_to_50pct != float('inf') else None,
            'ROUNDS_TO_75PCT_ASR': rounds_to_75pct if rounds_to_75pct != float('inf') else None,
            'ROUNDS_TO_90PCT_ASR': rounds_to_90pct if rounds_to_90pct != float('inf') else None
        })

    config_df = pd.DataFrame(config_data)

    # Convert to long format for plotting
    plot_data = pd.melt(
        config_df,
        id_vars=['GRAD_MODE', 'TRIGGER_RATE', 'POISON_STRENGTH', 'IS_SYBIL', 'N_ADV'],
        value_vars=['ROUNDS_TO_50PCT_ASR', 'ROUNDS_TO_75PCT_ASR', 'ROUNDS_TO_90PCT_ASR'],
        var_name='ASR_THRESHOLD',
        value_name='ROUNDS'
    )

    # Replace None with NaN for plotting
    plot_data['ROUNDS'] = pd.to_numeric(plot_data['ROUNDS'], errors='coerce')

    # Clean up labels
    plot_data['ASR_THRESHOLD'] = plot_data['ASR_THRESHOLD'].map({
        'ROUNDS_TO_50PCT_ASR': '50% ASR',
        'ROUNDS_TO_75PCT_ASR': '75% ASR',
        'ROUNDS_TO_90PCT_ASR': '90% ASR'
    })

    # Plot rounds to reach thresholds by gradient mode
    plt.subplot(1, 2, 1)
    sns.barplot(x='GRAD_MODE', y='ROUNDS', hue='ASR_THRESHOLD', data=plot_data)
    plt.title('Rounds to Reach ASR Thresholds by Gradient Mode', fontsize=14)
    plt.ylabel('Rounds')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot rounds to reach thresholds by Sybil mode
    plt.subplot(1, 2, 2)
    sns.barplot(x='IS_SYBIL', y='ROUNDS', hue='ASR_THRESHOLD', data=plot_data)
    plt.title('Rounds to Reach ASR Thresholds by Sybil Mode', fontsize=14)
    plt.ylabel('Rounds')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Sybil Attack Enabled')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/threshold_convergence_analysis.png", dpi=300)
    plt.close()


def plot_detailed_heatmaps(summary_df, output_dir):
    """
    Create heatmaps to show interactions between pairs of parameters.
    """
    # 1. ASR Heatmap: Trigger Rate vs Number of Adversaries
    plt.figure(figsize=(12, 10))
    pivot = summary_df.pivot_table(
        values='FINAL_ASR',
        index='TRIGGER_RATE',
        columns='N_ADV',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.2f')
    plt.title('ASR by Trigger Rate and Number of Adversaries', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_trigger_rate_vs_n_adv.png", dpi=300)
    plt.close()

    # 2. ASR Heatmap: Gradient Mode vs Sybil Mode
    plt.figure(figsize=(12, 10))
    pivot = summary_df.pivot_table(
        values='FINAL_ASR',
        index='GRAD_MODE',
        columns='IS_SYBIL',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.2f')
    plt.title('ASR by Gradient Mode and Sybil Mode', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_grad_mode_vs_sybil.png", dpi=300)
    plt.close()

    # 3. For CMD mode: ASR Heatmap by Poison Strength and Trigger Rate
    cmd_data = summary_df[summary_df['GRAD_MODE'] == 'cmd']
    if not cmd_data.empty:
        plt.figure(figsize=(12, 10))
        pivot = cmd_data.pivot_table(
            values='FINAL_ASR',
            index='POISON_STRENGTH',
            columns='TRIGGER_RATE',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.2f')
        plt.title('ASR by Poison Strength and Trigger Rate (CMD Mode)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/heatmap_poison_strength_vs_trigger_rate.png", dpi=300)
        plt.close()

    # 4. Main Accuracy Heatmap: Gradient Mode vs Sybil Mode
    plt.figure(figsize=(12, 10))
    pivot = summary_df.pivot_table(
        values='FINAL_MAIN_ACC',
        index='GRAD_MODE',
        columns='IS_SYBIL',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', vmin=0.7, vmax=1, fmt='.2f')
    plt.title('Main Task Accuracy by Gradient Mode and Sybil Mode', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_accuracy_grad_mode_vs_sybil.png", dpi=300)
    plt.close()

    # 5. Stealth Heatmap: Trigger Rate vs Gradient Mode
    if 'STEALTH' in summary_df.columns:
        plt.figure(figsize=(12, 10))
        pivot = summary_df.pivot_table(
            values='STEALTH',
            index='TRIGGER_RATE',
            columns='GRAD_MODE',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', vmin=0.7, vmax=1, fmt='.2f')
        plt.title('Attack Stealth by Trigger Rate and Gradient Mode', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/heatmap_stealth_trigger_rate_vs_grad_mode.png", dpi=300)
        plt.close()


def plot_trigger_rate_impact(summary_df, all_rounds_df, output_dir):
    """
    Analyze the impact of trigger rates on different metrics.
    """
    # Skip if trigger rate column is missing or only has one value
    if 'TRIGGER_RATE' not in summary_df.columns or len(summary_df['TRIGGER_RATE'].unique()) <= 1:
        return

    plt.figure(figsize=(20, 15))

    # 1. Trigger Rate Impact on Final ASR
    plt.subplot(2, 2, 1)
    sns.boxplot(x='TRIGGER_RATE', y='FINAL_ASR', data=summary_df)
    plt.title('Trigger Rate Impact on Final ASR', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Trigger Rate Impact on Main Task Accuracy
    plt.subplot(2, 2, 2)
    sns.boxplot(x='TRIGGER_RATE', y='FINAL_MAIN_ACC', data=summary_df)
    plt.title('Trigger Rate Impact on Main Task Accuracy', fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Trigger Rate Impact on ASR by Gradient Mode
    plt.subplot(2, 2, 3)
    sns.boxplot(x='TRIGGER_RATE', y='FINAL_ASR', hue='GRAD_MODE', data=summary_df)
    plt.title('Trigger Rate Impact on ASR by Gradient Mode', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Trigger Rate Impact on Main Task Accuracy by Gradient Mode
    plt.subplot(2, 2, 4)
    sns.boxplot(x='TRIGGER_RATE', y='FINAL_MAIN_ACC', hue='GRAD_MODE', data=summary_df)
    plt.title('Trigger Rate Impact on Main Task Accuracy by Gradient Mode', fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/trigger_rate_impact.png", dpi=300)
    plt.close()

    # Also analyze convergence speed by trigger rate
    if 'ROUNDS_TO_50PCT_ASR' in summary_df.columns:
        plt.figure(figsize=(15, 10))

        # Convert -1 (never reached) to NaN for visualization_226
        df_plot = summary_df.copy()
        for col in ['ROUNDS_TO_50PCT_ASR', 'ROUNDS_TO_75PCT_ASR', 'ROUNDS_TO_90PCT_ASR']:
            if col in df_plot.columns:
                df_plot[col] = df_plot[col].replace(-1, np.nan)

        sns.boxplot(x='TRIGGER_RATE', y='ROUNDS_TO_50PCT_ASR', data=df_plot)
        plt.title('Rounds to Reach 50% ASR by Trigger Rate', fontsize=14)
        plt.ylabel('Rounds')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/trigger_rate_convergence.png", dpi=300)
        plt.close()


def plot_poison_strength_impact(summary_df, all_rounds_df, output_dir):
    """
    Analyze the impact of poison strength on different metrics (for CMD mode).
    """
    # Filter to CMD mode only
    cmd_df = summary_df[summary_df['GRAD_MODE'] == 'cmd']

    # Skip if empty or poison strength column is missing or only has one value
    if cmd_df.empty or 'POISON_STRENGTH' not in cmd_df.columns or len(cmd_df['POISON_STRENGTH'].unique()) <= 1:
        return

    plt.figure(figsize=(20, 15))

    # 1. Poison Strength Impact on Final ASR
    plt.subplot(2, 2, 1)
    sns.boxplot(x='POISON_STRENGTH', y='FINAL_ASR', data=cmd_df)
    plt.title('Poison Strength Impact on Final ASR (CMD Mode)', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Poison Strength Impact on Main Task Accuracy
    plt.subplot(2, 2, 2)
    sns.boxplot(x='POISON_STRENGTH', y='FINAL_MAIN_ACC', data=cmd_df)
    plt.title('Poison Strength Impact on Main Task Accuracy (CMD Mode)', fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Poison Strength Impact on ASR by Sybil Mode
    plt.subplot(2, 2, 3)
    sns.boxplot(x='POISON_STRENGTH', y='FINAL_ASR', hue='IS_SYBIL', data=cmd_df)
    plt.title('Poison Strength Impact on ASR by Sybil Mode (CMD Mode)', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Poison Strength Impact on Main Task Accuracy by Sybil Mode
    plt.subplot(2, 2, 4)
    sns.boxplot(x='POISON_STRENGTH', y='FINAL_MAIN_ACC', hue='IS_SYBIL', data=cmd_df)
    plt.title('Poison Strength Impact on Main Task Accuracy by Sybil Mode (CMD Mode)', fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/poison_strength_impact.png", dpi=300)
    plt.close()

    # Also analyze convergence speed by poison strength
    if 'ROUNDS_TO_50PCT_ASR' in cmd_df.columns:
        plt.figure(figsize=(15, 10))

        # Convert -1 (never reached) to NaN for visualization_226
        df_plot = cmd_df.copy()
        for col in ['ROUNDS_TO_50PCT_ASR', 'ROUNDS_TO_75PCT_ASR', 'ROUNDS_TO_90PCT_ASR']:
            if col in df_plot.columns:
                df_plot[col] = df_plot[col].replace(-1, np.nan)

        sns.boxplot(x='POISON_STRENGTH', y='ROUNDS_TO_50PCT_ASR', data=df_plot)
        plt.title('Rounds to Reach 50% ASR by Poison Strength (CMD Mode)', fontsize=14)
        plt.ylabel('Rounds')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/poison_strength_convergence.png", dpi=300)
        plt.close()


def plot_sybil_impact(summary_df, all_rounds_df, output_dir):
    """
    Analyze the impact of Sybil attacks on different metrics.
    """
    # Skip if Sybil column is missing or only has one value
    if 'IS_SYBIL' not in summary_df.columns or len(summary_df['IS_SYBIL'].unique()) <= 1:
        return

    plt.figure(figsize=(20, 15))

    # 1. Sybil Impact on Final ASR
    plt.subplot(2, 2, 1)
    sns.boxplot(x='IS_SYBIL', y='FINAL_ASR', data=summary_df)
    plt.title('Sybil Attack Impact on Final ASR', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Sybil Attack Enabled')

    # 2. Sybil Impact on Main Task Accuracy
    plt.subplot(2, 2, 2)
    sns.boxplot(x='IS_SYBIL', y='FINAL_MAIN_ACC', data=summary_df)
    plt.title('Sybil Attack Impact on Main Task Accuracy', fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Sybil Attack Enabled')

    # 3. Sybil Impact on ASR by Number of Adversaries
    plt.subplot(2, 2, 3)
    sns.boxplot(x='N_ADV', y='FINAL_ASR', hue='IS_SYBIL', data=summary_df)
    plt.title('Sybil Attack Impact on ASR by Number of Adversaries', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Sybil Impact on ASR by Gradient Mode
    plt.subplot(2, 2, 4)
    sns.boxplot(x='GRAD_MODE', y='FINAL_ASR', hue='IS_SYBIL', data=summary_df)
    plt.title('Sybil Attack Impact on ASR by Gradient Mode', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sybil_impact.png", dpi=300)
    plt.close()

    # Analyze convergence speed difference with Sybil attacks
    if 'ROUNDS_TO_50PCT_ASR' in summary_df.columns:
        plt.figure(figsize=(15, 10))

        # Convert -1 (never reached) to NaN for visualization_226
        df_plot = summary_df.copy()
        for col in ['ROUNDS_TO_50PCT_ASR', 'ROUNDS_TO_75PCT_ASR', 'ROUNDS_TO_90PCT_ASR']:
            if col in df_plot.columns:
                df_plot[col] = df_plot[col].replace(-1, np.nan)

        plt.subplot(1, 2, 1)
        sns.boxplot(x='IS_SYBIL', y='ROUNDS_TO_50PCT_ASR', data=df_plot)
        plt.title('Rounds to Reach 50% ASR by Sybil Mode', fontsize=14)
        plt.ylabel('Rounds')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xlabel('Sybil Attack Enabled')

        plt.subplot(1, 2, 2)
        sns.boxplot(x='IS_SYBIL', y='ROUNDS_TO_75PCT_ASR', data=df_plot)
        plt.title('Rounds to Reach 75% ASR by Sybil Mode', fontsize=14)
        plt.ylabel('Rounds')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xlabel('Sybil Attack Enabled')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/sybil_convergence.png", dpi=300)
        plt.close()


def plot_adversary_scaling(summary_df, all_rounds_df, output_dir):
    """
    Analyze how attack effectiveness scales with the number of adversaries.
    """
    # Skip if adversary number column is missing or only has one value
    if 'N_ADV' not in summary_df.columns or len(summary_df['N_ADV'].unique()) <= 1:
        return

    plt.figure(figsize=(20, 15))

    # 1. Number of Adversaries Impact on Final ASR
    plt.subplot(2, 2, 1)
    sns.boxplot(x='N_ADV', y='FINAL_ASR', data=summary_df)
    plt.title('Number of Adversaries Impact on Final ASR', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Number of Adversaries Impact on Main Task Accuracy
    plt.subplot(2, 2, 2)
    sns.boxplot(x='N_ADV', y='FINAL_MAIN_ACC', data=summary_df)
    plt.title('Number of Adversaries Impact on Main Task Accuracy', fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Number of Adversaries Impact on ASR by Gradient Mode
    plt.subplot(2, 2, 3)
    sns.boxplot(x='N_ADV', y='FINAL_ASR', hue='GRAD_MODE', data=summary_df)
    plt.title('Number of Adversaries Impact on ASR by Gradient Mode', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Attack Efficiency (ASR per Adversary)
    plt.subplot(2, 2, 4)
    if 'ASR_PER_ADV' not in summary_df.columns:
        summary_df['ASR_PER_ADV'] = summary_df['FINAL_ASR'] / summary_df['N_ADV']

    sns.boxplot(x='N_ADV', y='ASR_PER_ADV', data=summary_df)
    plt.title('Attack Efficiency (ASR per Adversary)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/adversary_scaling.png", dpi=300)
    plt.close()

    # Plot curves showing how ASR scales with number of adversaries
    plt.figure(figsize=(15, 10))

    # Group by all other parameters except N_ADV to see scaling within same configurations
    scaling_data = []

    param_groups = summary_df.groupby(['GRAD_MODE', 'TRIGGER_RATE', 'POISON_STRENGTH', 'IS_SYBIL'])

    for params, group in param_groups:
        if len(group['N_ADV'].unique()) > 1:  # Only include if we have multiple N_ADV values
            grad_mode, trigger_rate, poison_strength, is_sybil = params

            for i, row in group.sort_values('N_ADV').iterrows():
                scaling_data.append({
                    'GRAD_MODE': grad_mode,
                    'TRIGGER_RATE': trigger_rate,
                    'POISON_STRENGTH': poison_strength,
                    'IS_SYBIL': is_sybil,
                    'N_ADV': row['N_ADV'],
                    'FINAL_ASR': row['FINAL_ASR'],
                    'config_id': f"{grad_mode}_{trigger_rate}_{poison_strength}_{is_sybil}"
                })

    scaling_df = pd.DataFrame(scaling_data)

    if not scaling_df.empty:
        # Plot scaling curves for each configuration
        for config_id in scaling_df['config_id'].unique():
            config_data = scaling_df[scaling_df['config_id'] == config_id]
            if len(config_data) > 1:  # Need at least 2 points for a line
                plt.plot(
                    config_data['N_ADV'],
                    config_data['FINAL_ASR'],
                    marker='o',
                    label=config_id
                )

        plt.title('ASR Scaling with Number of Adversaries', fontsize=14)
        plt.xlabel('Number of Adversaries')
        plt.ylabel('Attack Success Rate')
        plt.ylim(0, 1.05)
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/adversary_scaling_curves.png", dpi=300)
        plt.close()

    # Plot learning dynamics with different numbers of adversaries
    if 'round' in all_rounds_df.columns and 'asr' in all_rounds_df.columns:
        plt.figure(figsize=(15, 10))

        # Group by round and number of adversaries
        grouped = all_rounds_df.groupby(['N_ADV', 'round']).agg({
            'asr': ['mean', 'std']
        }).reset_index()

        # Flatten multi-level columns
        grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

        # Plot ASR over rounds for each N_ADV value
        for n_adv in sorted(all_rounds_df['N_ADV'].unique()):
            data = grouped[grouped['N_ADV'] == n_adv]

            plt.plot(
                data['round'],
                data['asr_mean'],
                marker='o',
                markersize=4,
                label=f"N_ADV={n_adv}"
            )
            plt.fill_between(
                data['round'],
                data['asr_mean'] - data['asr_std'],
                data['asr_mean'] + data['asr_std'],
                alpha=0.2
            )

        plt.title('ASR Progression by Number of Adversaries', fontsize=14)
        plt.xlabel('Round')
        plt.ylabel('Attack Success Rate')
        plt.ylim(0, 1.05)
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/adversary_learning_dynamics.png", dpi=300)
        plt.close()


def create_summary_table(summary_df, output_dir):
    """
    Create a comprehensive summary table of all attack configurations and their performance.
    Includes comparisons across different aggregation methods.
    """
    if summary_df.empty:
        return

    # Select relevant columns for the summary
    cols = [
        'AGGREGATION_METHOD', 'GRAD_MODE', 'TRIGGER_RATE', 'POISON_STRENGTH', 'IS_SYBIL', 'N_ADV',
        'FINAL_ASR', 'MAX_ASR', 'FINAL_MAIN_ACC', 'FINAL_CLEAN_ACC',
        'ROUNDS_TO_50PCT_ASR', 'ROUNDS_TO_75PCT_ASR', 'ASR_PER_ADV', 'STEALTH'
    ]

    # Use only columns that exist
    avail_cols = [col for col in cols if col in summary_df.columns]

    # Create the summary table
    summary_table = summary_df[avail_cols].copy()

    # Round numeric columns for readability
    numeric_cols = summary_table.select_dtypes(include=['float64']).columns
    summary_table[numeric_cols] = summary_table[numeric_cols].round(3)

    # Replace -1 values with "Never" for rounds to reach thresholds
    if 'ROUNDS_TO_50PCT_ASR' in summary_table.columns:
        summary_table['ROUNDS_TO_50PCT_ASR'] = summary_table['ROUNDS_TO_50PCT_ASR'].replace(-1, 'Never')
    if 'ROUNDS_TO_75PCT_ASR' in summary_table.columns:
        summary_table['ROUNDS_TO_75PCT_ASR'] = summary_table['ROUNDS_TO_75PCT_ASR'].replace(-1, 'Never')

    # Sort by ASR, then by Main Accuracy
    if 'FINAL_ASR' in summary_table.columns and 'FINAL_MAIN_ACC' in summary_table.columns:
        summary_table = summary_table.sort_values(['FINAL_ASR', 'FINAL_MAIN_ACC'], ascending=[False, False])

    # Save to CSV
    summary_table.to_csv(f"{output_dir}/attack_summary_table.csv", index=False)

    # Also create aggregate statistics by different groupings
    if 'FINAL_ASR' in summary_df.columns:
        # Group by AGGREGATION_METHOD
        if 'AGGREGATION_METHOD' in summary_df.columns:
            agg_method_stats = summary_df.groupby('AGGREGATION_METHOD')[['FINAL_ASR', 'FINAL_MAIN_ACC']].mean().round(3)
            agg_method_stats.to_csv(f"{output_dir}/aggregation_method_summary.csv")

        # Group by AGGREGATION_METHOD and GRAD_MODE
        if 'AGGREGATION_METHOD' in summary_df.columns and 'GRAD_MODE' in summary_df.columns:
            agg_grad_stats = summary_df.groupby(['AGGREGATION_METHOD', 'GRAD_MODE'])[
                ['FINAL_ASR', 'FINAL_MAIN_ACC']].mean().round(3)
            agg_grad_stats.to_csv(f"{output_dir}/aggregation_grad_mode_summary.csv")

        # Group by GRAD_MODE
        grad_mode_stats = summary_df.groupby('GRAD_MODE')[['FINAL_ASR', 'FINAL_MAIN_ACC']].mean().round(3)
        grad_mode_stats.to_csv(f"{output_dir}/grad_mode_summary.csv")

        # Group by IS_SYBIL
        if 'IS_SYBIL' in summary_df.columns:
            sybil_stats = summary_df.groupby('IS_SYBIL')[['FINAL_ASR', 'FINAL_MAIN_ACC']].mean().round(3)
            sybil_stats.to_csv(f"{output_dir}/sybil_summary.csv")

        # Group by N_ADV
        if 'N_ADV' in summary_df.columns:
            n_adv_stats = summary_df.groupby('N_ADV')[['FINAL_ASR', 'FINAL_MAIN_ACC']].mean().round(3)
            n_adv_stats.to_csv(f"{output_dir}/n_adv_summary.csv")

        # Group by TRIGGER_RATE
        if 'TRIGGER_RATE' in summary_df.columns:
            trigger_stats = summary_df.groupby('TRIGGER_RATE')[['FINAL_ASR', 'FINAL_MAIN_ACC']].mean().round(3)
            trigger_stats.to_csv(f"{output_dir}/trigger_rate_summary.csv")

        # Group by POISON_STRENGTH (for CMD mode)
        cmd_df = summary_df[summary_df['GRAD_MODE'] == 'cmd']
        if not cmd_df.empty and 'POISON_STRENGTH' in cmd_df.columns:
            poison_stats = cmd_df.groupby('POISON_STRENGTH')[['FINAL_ASR', 'FINAL_MAIN_ACC']].mean().round(3)
            poison_stats.to_csv(f"{output_dir}/poison_strength_summary.csv")

        # Create comparison tables across aggregation methods for identical configurations
        if 'AGGREGATION_METHOD' in summary_df.columns and len(summary_df['AGGREGATION_METHOD'].unique()) > 1:
            # Group by all parameters except aggregation method
            param_cols = [col for col in ['GRAD_MODE', 'TRIGGER_RATE', 'POISON_STRENGTH', 'IS_SYBIL', 'N_ADV']
                          if col in summary_df.columns]

            # Pivot to compare aggregation methods for each configuration
            pivot_table = summary_df.pivot_table(
                index=param_cols,
                columns='AGGREGATION_METHOD',
                values=['FINAL_ASR', 'FINAL_MAIN_ACC'],
                aggfunc='mean'
            ).round(3)

            # Save the comparison table
            pivot_table.to_csv(f"{output_dir}/aggregation_method_comparison.csv")

            # Calculate differences between aggregation methods
            if len(summary_df['AGGREGATION_METHOD'].unique()) == 2:
                agg_methods = sorted(summary_df['AGGREGATION_METHOD'].unique())
                comparison_df = summary_df.pivot_table(
                    index=param_cols,
                    columns='AGGREGATION_METHOD',
                    values=['FINAL_ASR', 'FINAL_MAIN_ACC']
                )

                # Calculate differences (Method1 - Method2)
                diff_df = pd.DataFrame(index=comparison_df.index)
                for metric in ['FINAL_ASR', 'FINAL_MAIN_ACC']:
                    if (metric, agg_methods[0]) in comparison_df.columns and (
                            metric, agg_methods[1]) in comparison_df.columns:
                        diff_df[f"{metric}_DIFF"] = (comparison_df[metric, agg_methods[0]] - comparison_df[
                            metric, agg_methods[1]]).round(3)

                # Save the difference table
                diff_df.to_csv(f"{output_dir}/aggregation_method_differences.csv")


def run_visualization(summary_csv, all_rounds_csv, output_dir):
    """
    Run comprehensive visualization_226 of backdoor attack results.

    Args:
        summary_csv: Path to summary CSV file
        all_rounds_csv: Path to round-by-round CSV file
        output_dir: Directory to save visualizations
    """
    # Create output directory
    create_output_dir(output_dir)

    # Load data
    summary_df, all_rounds_df = load_processed_data(summary_csv, all_rounds_csv)

    # Calculate stealth if not already present
    if 'STEALTH' not in summary_df.columns and 'FINAL_MAIN_ACC' in summary_df.columns and 'FINAL_CLEAN_ACC' in summary_df.columns:
        summary_df['STEALTH'] = 1 - abs(summary_df['FINAL_MAIN_ACC'] - summary_df['FINAL_CLEAN_ACC'])

    # Calculate ASR per adversary if not already present
    if 'ASR_PER_ADV' not in summary_df.columns and 'FINAL_ASR' in summary_df.columns and 'N_ADV' in summary_df.columns:
        summary_df['ASR_PER_ADV'] = summary_df['FINAL_ASR'] / summary_df['N_ADV']

    # Run visualizations
    print("Generating attack success comparison plots...")
    plot_attack_success_comparison(summary_df, output_dir)

    print("Generating parameter interaction plots...")
    plot_parameter_interactions(summary_df, output_dir)

    print("Generating tradeoff analysis plots...")
    plot_tradeoff_analysis(summary_df, output_dir)

    print("Generating attack progression plots...")
    plot_attack_progression(all_rounds_df, output_dir)

    print("Generating detailed heatmaps...")
    plot_detailed_heatmaps(summary_df, output_dir)

    print("Generating trigger rate impact plots...")
    plot_trigger_rate_impact(summary_df, all_rounds_df, output_dir)

    print("Generating poison strength impact plots...")
    plot_poison_strength_impact(summary_df, all_rounds_df, output_dir)

    print("Generating Sybil impact plots...")
    plot_sybil_impact(summary_df, all_rounds_df, output_dir)

    print("Generating adversary scaling plots...")
    plot_adversary_scaling(summary_df, all_rounds_df, output_dir)

    print("Creating summary tables...")
    create_summary_table(summary_df, output_dir)

    print(f"Visualization complete. Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize backdoor attack results")
    parser.add_argument("--summary", required=True, help="Path to summary CSV file")
    parser.add_argument("--rounds", required=True, help="Path to round-by-round CSV file")
    parser.add_argument("--output", default="./visualization_results", help="Output directory")

    args = parser.parse_args()

    run_visualization(args.summary, args.rounds, args.output)
