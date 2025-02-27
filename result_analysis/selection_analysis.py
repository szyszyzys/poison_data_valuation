import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import ast

def analyze_selection_patterns(all_rounds_df, output_dir):
    """
    Analyze client selection patterns in federated learning backdoor attacks.

    Args:
        all_rounds_df: DataFrame containing round-by-round data
        output_dir: Directory to save output visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if necessary columns exist
    required_cols = ['round', 'malicious_rate', 'benign_rate', 'avg_malicious_rate', 'avg_benign_rate']
    missing_cols = [col for col in required_cols if col not in all_rounds_df.columns]

    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        print("Selection pattern analysis may be incomplete.")

    # 1. Malicious vs Benign Selection Rate Comparison
    plot_selection_rate_comparison(all_rounds_df, output_dir)

    # 2. Selection Rate Over Time
    plot_selection_rate_over_time(all_rounds_df, output_dir)

    # 3. Selection Rate vs Attack Success Rate
    plot_selection_vs_asr(all_rounds_df, output_dir)

    # 4. Selection Rate by Attack Parameters
    plot_selection_by_parameters(all_rounds_df, output_dir)

    # 5. Selection Efficiency Analysis
    plot_selection_efficiency(all_rounds_df, output_dir)

    # 6. Client Selection Pattern Heatmap (if used_sellers is available)
    if 'used_sellers' in all_rounds_df.columns:
        plot_client_selection_heatmap(all_rounds_df, output_dir)

    # 7. Outlier Detection Analysis (if outlier_ids is available)
    if 'outlier_ids' in all_rounds_df.columns:
        plot_outlier_detection_analysis(all_rounds_df, output_dir)

def plot_selection_rate_comparison(df, output_dir):
    """
    Compare selection rates between malicious and benign clients.
    """
    if 'malicious_rate' not in df.columns or 'benign_rate' not in df.columns:
        return

    # Create aggregate data
    agg_data = []

    # Group by experiment parameters and calculate average rates
    group_cols = ['GRAD_MODE', 'TRIGGER_RATE', 'IS_SYBIL', 'N_ADV']
    if 'POISON_STRENGTH' in df.columns:
        group_cols.append('POISON_STRENGTH')
    if 'AGGREGATION_METHOD' in df.columns:
        group_cols.append('AGGREGATION_METHOD')

    # Only use columns that exist
    group_cols = [col for col in group_cols if col in df.columns]

    grouped = df.groupby(group_cols)

    for params, group in grouped:
        param_dict = {col: val for col, val in zip(group_cols, params if isinstance(params, tuple) else [params])}

        agg_data.append({
            **param_dict,
            'avg_malicious_rate': group['malicious_rate'].mean(),
            'avg_benign_rate': group['benign_rate'].mean()
        })

    agg_df = pd.DataFrame(agg_data)

    # Create a long-format DataFrame for plotting
    plot_data = pd.melt(
        agg_df,
        id_vars=group_cols,
        value_vars=['avg_malicious_rate', 'avg_benign_rate'],
        var_name='client_type',
        value_name='selection_rate'
    )

    # Clean up labels
    plot_data['client_type'] = plot_data['client_type'].map({
        'avg_malicious_rate': 'Malicious',
        'avg_benign_rate': 'Benign'
    })

    # Create plots
    plt.figure(figsize=(20, 15))

    # 1. Selection Rate by Gradient Mode
    if 'GRAD_MODE' in group_cols:
        plt.subplot(2, 2, 1)
        sns.barplot(x='GRAD_MODE', y='selection_rate', hue='client_type', data=plot_data)
        plt.title('Selection Rate by Gradient Mode', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Selection Rate by Sybil Mode
    if 'IS_SYBIL' in group_cols:
        plt.subplot(2, 2, 2)
        sns.barplot(x='IS_SYBIL', y='selection_rate', hue='client_type', data=plot_data)
        plt.title('Selection Rate by Sybil Mode', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xlabel('Sybil Attack Enabled')

    # 3. Selection Rate by Trigger Rate
    if 'TRIGGER_RATE' in group_cols:
        plt.subplot(2, 2, 3)
        sns.barplot(x='TRIGGER_RATE', y='selection_rate', hue='client_type', data=plot_data)
        plt.title('Selection Rate by Trigger Rate', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Selection Rate by Number of Adversaries
    if 'N_ADV' in group_cols:
        plt.subplot(2, 2, 4)
        sns.barplot(x='N_ADV', y='selection_rate', hue='client_type', data=plot_data)
        plt.title('Selection Rate by Number of Adversaries', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/selection_rate_comparison.png", dpi=300)
    plt.close()

    # If there's an aggregation method, also plot by that
    if 'AGGREGATION_METHOD' in group_cols:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='AGGREGATION_METHOD', y='selection_rate', hue='client_type', data=plot_data)
        plt.title('Selection Rate by Aggregation Method', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/selection_rate_by_aggregation.png", dpi=300)
        plt.close()

def plot_selection_rate_over_time(df, output_dir):
    """
    Plot selection rates over rounds to identify temporal patterns.
    """
    if 'malicious_rate' not in df.columns or 'benign_rate' not in df.columns:
        return

    # Group by round and calculate average rates across all experiments
    grouped = df.groupby('round').agg({
        'malicious_rate': ['mean', 'std'],
        'benign_rate': ['mean', 'std']
    }).reset_index()

    # Flatten multi-level columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

    plt.figure(figsize=(15, 10))

    # Plot malicious selection rate over time
    plt.plot(
        grouped['round'],
        grouped['malicious_rate_mean'],
        label='Malicious',
        marker='o',
        color='red'
    )
    plt.fill_between(
        grouped['round'],
        grouped['malicious_rate_mean'] - grouped['malicious_rate_std'],
        grouped['malicious_rate_mean'] + grouped['malicious_rate_std'],
        alpha=0.2,
        color='red'
    )

    # Plot benign selection rate over time
    plt.plot(
        grouped['round'],
        grouped['benign_rate_mean'],
        label='Benign',
        marker='o',
        color='blue'
    )
    plt.fill_between(
        grouped['round'],
        grouped['benign_rate_mean'] - grouped['benign_rate_std'],
        grouped['benign_rate_mean'] + grouped['benign_rate_std'],
        alpha=0.2,
        color='blue'
    )

    plt.title('Client Selection Rate Over Rounds', fontsize=14)
    plt.xlabel('Round')
    plt.ylabel('Selection Rate')
    plt.ylim(0, 1.05)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/selection_rate_over_time.png", dpi=300)
    plt.close()

    # Also plot selection rate over time by gradient mode
    if 'GRAD_MODE' in df.columns:
        for grad_mode in df['GRAD_MODE'].unique():
            grad_data = df[df['GRAD_MODE'] == grad_mode]

            if grad_data.empty:
                continue

            grouped = grad_data.groupby('round').agg({
                'malicious_rate': ['mean', 'std'],
                'benign_rate': ['mean', 'std']
            }).reset_index()

            # Flatten multi-level columns
            grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

            plt.figure(figsize=(15, 10))

            # Plot malicious selection rate over time
            plt.plot(
                grouped['round'],
                grouped['malicious_rate_mean'],
                label='Malicious',
                marker='o',
                color='red'
            )
            plt.fill_between(
                grouped['round'],
                grouped['malicious_rate_mean'] - grouped['malicious_rate_std'],
                grouped['malicious_rate_mean'] + grouped['malicious_rate_std'],
                alpha=0.2,
                color='red'
            )

            # Plot benign selection rate over time
            plt.plot(
                grouped['round'],
                grouped['benign_rate_mean'],
                label='Benign',
                marker='o',
                color='blue'
            )
            plt.fill_between(
                grouped['round'],
                grouped['benign_rate_mean'] - grouped['benign_rate_std'],
                grouped['benign_rate_mean'] + grouped['benign_rate_std'],
                alpha=0.2,
                color='blue'
            )

            plt.title(f'Client Selection Rate Over Rounds ({grad_mode} Mode)', fontsize=14)
            plt.xlabel('Round')
            plt.ylabel('Selection Rate')
            plt.ylim(0, 1.05)
            plt.grid(linestyle='--', alpha=0.7)
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"{output_dir}/selection_rate_over_time_{grad_mode}.png", dpi=300)
            plt.close()

def plot_selection_vs_asr(df, output_dir):
    """
    Plot relationship between selection rate and attack success rate.
    """
    if 'malicious_rate' not in df.columns or 'asr' not in df.columns:
        return

    plt.figure(figsize=(20, 10))

    # 1. Malicious Selection Rate vs ASR
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x='malicious_rate',
        y='asr',
        hue='GRAD_MODE',
        size='N_ADV',
        data=df,
        alpha=0.7
    )

    # Add regression line
    sns.regplot(
        x='malicious_rate',
        y='asr',
        data=df,
        scatter=False,
        ci=None,
        line_kws={'color': 'black', 'linestyle': '--'}
    )

    plt.title('Attack Success Rate vs Malicious Selection Rate', fontsize=14)
    plt.xlabel('Malicious Selection Rate')
    plt.ylabel('Attack Success Rate')
    plt.grid(linestyle='--', alpha=0.7)

    # 2. Malicious Selection Rate vs ASR by Sybil Mode
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x='malicious_rate',
        y='asr',
        hue='IS_SYBIL',
        size='TRIGGER_RATE',
        data=df,
        alpha=0.7
    )

    # Add regression lines for each Sybil mode
    for is_sybil in df['IS_SYBIL'].unique():
        sybil_data = df[df['IS_SYBIL'] == is_sybil]
        sns.regplot(
            x='malicious_rate',
            y='asr',
            data=sybil_data,
            scatter=False,
            ci=None,
            line_kws={'linestyle': '--'}
        )

    plt.title('Attack Success Rate vs Malicious Selection Rate by Sybil Mode', fontsize=14)
    plt.xlabel('Malicious Selection Rate')
    plt.ylabel('Attack Success Rate')
    plt.grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/selection_vs_asr.png", dpi=300)
    plt.close()

def plot_selection_by_parameters(df, output_dir):
    """
    Analyze how different attack parameters affect selection rates.
    """
    if 'malicious_rate' not in df.columns:
        return

    # Check available parameters
    param_cols = []
    for col in ['GRAD_MODE', 'IS_SYBIL', 'TRIGGER_RATE', 'POISON_STRENGTH', 'N_ADV', 'AGGREGATION_METHOD']:
        if col in df.columns and len(df[col].unique()) > 1:
            param_cols.append(col)

    # Need at least 2 parameter columns for interesting heatmaps
    if len(param_cols) < 2:
        return

    # Create heatmaps for different parameter combinations
    # Try a few common combinations
    combinations = [
        ('GRAD_MODE', 'IS_SYBIL'),
        ('TRIGGER_RATE', 'N_ADV'),
        ('POISON_STRENGTH', 'N_ADV'),
        ('GRAD_MODE', 'AGGREGATION_METHOD'),
        ('IS_SYBIL', 'AGGREGATION_METHOD')
    ]

    for param1, param2 in combinations:
        if param1 in param_cols and param2 in param_cols:
            plt.figure(figsize=(15, 10))

            pivot = df.pivot_table(
                values='malicious_rate',
                index=param1,
                columns=param2,
                aggfunc='mean'
            )

            if not pivot.empty:
                sns.heatmap(pivot, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.2f')
                plt.title(f'Malicious Selection Rate by {param1} and {param2}', fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/selection_heatmap_{param1}_{param2}.png", dpi=300)
                plt.close()

    # For CMD mode only: Poison Strength analysis if available
    if 'GRAD_MODE' in df.columns and 'POISON_STRENGTH' in df.columns:
        cmd_data = df[df['GRAD_MODE'] == 'cmd']
        if not cmd_data.empty and len(cmd_data['POISON_STRENGTH'].unique()) > 1:
            plt.figure(figsize=(15, 10))

            if 'TRIGGER_RATE' in cmd_data.columns and len(cmd_data['TRIGGER_RATE'].unique()) > 1:
                pivot = cmd_data.pivot_table(
                    values='malicious_rate',
                    index='POISON_STRENGTH',
                    columns='TRIGGER_RATE',
                    aggfunc='mean'
                )

                if not pivot.empty:
                    sns.heatmap(pivot, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.2f')
                    plt.title('Malicious Selection Rate by Poison Strength and Trigger Rate (CMD Mode)', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/selection_heatmap_cmd_poison_trigger.png", dpi=300)
                    plt.close()

def plot_selection_efficiency(df, output_dir):
    """
    Analyze selection efficiency (ASR per malicious client selected).
    """
    if 'malicious_rate' not in df.columns or 'asr' not in df.columns:
        return

    # Calculate selection efficiency for each round
    df['selection_efficiency'] = df['asr'] / (df['malicious_rate'] + 1e-10)  # Avoid division by zero

    # Group by experiment parameters and calculate average efficiency
    group_cols = []
    for col in ['GRAD_MODE', 'IS_SYBIL', 'TRIGGER_RATE', 'POISON_STRENGTH', 'N_ADV', 'AGGREGATION_METHOD']:
        if col in df.columns:
            group_cols.append(col)

    if not group_cols:
        return

    grouped = df.groupby(group_cols)

    agg_data = []
    for params, group in grouped:
        param_dict = {col: val for col, val in zip(group_cols, params if isinstance(params, tuple) else [params])}

        agg_data.append({
            **param_dict,
            'avg_efficiency': group['selection_efficiency'].mean()
        })

    agg_df = pd.DataFrame(agg_data)

    # Create various plots based on available parameters
    plt.figure(figsize=(20, 15))

    plot_idx = 1
    for param in ['GRAD_MODE', 'IS_SYBIL', 'TRIGGER_RATE', 'N_ADV']:
        if param in agg_df.columns and len(agg_df[param].unique()) > 1:
            plt.subplot(2, 2, plot_idx)
            sns.barplot(x=param, y='avg_efficiency', data=agg_df)
            plt.title(f'Selection Efficiency by {param}', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            if param == 'IS_SYBIL':
                plt.xlabel('Sybil Attack Enabled')

            plot_idx += 1

            if plot_idx > 4:
                break

    plt.tight_layout()
    plt.savefig(f"{output_dir}/selection_efficiency.png", dpi=300)
    plt.close()

    # If aggregation method is available, create a specific plot
    if 'AGGREGATION_METHOD' in agg_df.columns and len(agg_df['AGGREGATION_METHOD'].unique()) > 1:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='AGGREGATION_METHOD', y='avg_efficiency', data=agg_df)
        plt.title('Selection Efficiency by Aggregation Method', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/selection_efficiency_by_aggregation.png", dpi=300)
        plt.close()

def plot_client_selection_heatmap(df, output_dir):
    """
    Create a heatmap showing client selection patterns over rounds.
    Note: This requires 'used_sellers' column to be a list of selected clients.
    """
    if 'used_sellers' not in df.columns:
        return

    # Process the used_sellers column if it's stored as a string representation of a list
    if df['used_sellers'].dtype == 'object':
        # Check if the first non-null value is a string and convert if needed
        first_valid = df['used_sellers'].dropna().iloc[0] if not df['used_sellers'].dropna().empty else None
        if isinstance(first_valid, str):
            try:
                df['used_sellers'] = df['used_sellers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            except (SyntaxError, ValueError):
                print("Warning: Could not parse 'used_sellers' column as a list. Skipping client selection heatmap.")
                return

    # Sample one experiment to analyze (for clarity in visualization_226)
    # Get unique experiments
    experiment_params = []
    group_cols = []
    for col in ['GRAD_MODE', 'IS_SYBIL', 'TRIGGER_RATE', 'POISON_STRENGTH', 'N_ADV', 'AGGREGATION_METHOD']:
        if col in df.columns and len(df[col].unique()) > 1:
            group_cols.append(col)

    if group_cols:
        # Get a sample of up to 3 different experiment configurations
        exp_params = df[group_cols].drop_duplicates().head(3)

        for _, params in exp_params.iterrows():
            # Filter to the chosen experiment
            mask = True
            for col, value in params.items():
                mask = mask & (df[col] == value)

            exp_df = df[mask].sort_values('round')

            if len(exp_df) > 5:  # Only create heatmap if we have enough rounds
                # Get all unique client IDs
                all_clients = set()
                for sellers in exp_df['used_sellers']:
                    if isinstance(sellers, list):
                        all_clients.update(sellers)
                all_clients = sorted(list(all_clients))

                # Create selection matrix
                selection_matrix = np.zeros((len(all_clients), len(exp_df)))

                for i, (_, row) in enumerate(exp_df.iterrows()):
                    sellers = row['used_sellers']
                    if isinstance(sellers, list):
                        for seller in sellers:
                            if seller in all_clients:
                                client_idx = all_clients.index(seller)
                                selection_matrix[client_idx, i] = 1

                # Create heatmap
                plt.figure(figsize=(20, 10))
                sns.heatmap(
                    selection_matrix,
                    cmap='Blues',
                    cbar=False,
                    linewidths=0.5,
                    linecolor='gray'
                )

                # Create a readable title with the experiment parameters
                title_parts = []
                for col in group_cols:
                    title_parts.append(f"{col}={params[col]}")
                title = "Client Selection Pattern Over Rounds\n" + ", ".join(title_parts)

                plt.title(title, fontsize=14)
                plt.xlabel('Round')
                plt.ylabel('Client ID')
                plt.yticks(np.arange(len(all_clients)) + 0.5, all_clients)

                plt.tight_layout()
                plt.savefig(f"{output_dir}/client_selection_heatmap_{experiment_params.index((tuple(params)))}.png", dpi=300)
                plt.close()
    else:
        # Fallback if no grouping parameters available
        exp_df = df.sort_values('round')

        if len(exp_df) > 5:  # Only create heatmap if we have enough rounds
            # Get all unique client IDs
            all_clients = set()
            for sellers in exp_df['used_sellers']:
                if isinstance(sellers, list):
                    all_clients.update(sellers)
            all_clients = sorted(list(all_clients))

            # Create selection matrix
            selection_matrix = np.zeros((len(all_clients), len(exp_df)))

            for i, (_, row) in enumerate(exp_df.iterrows()):
                sellers = row['used_sellers']
                if isinstance(sellers, list):
                    for seller in sellers:
                        if seller in all_clients:
                            client_idx = all_clients.index(seller)
                            selection_matrix[client_idx, i] = 1

            # Create heatmap
            plt.figure(figsize=(20, 10))
            sns.heatmap(
                selection_matrix,
                cmap='Blues',
                cbar=False,
                linewidths=0.5,
                linecolor='gray'
            )
            plt.title(f'Client Selection Pattern Over Rounds', fontsize=14)
            plt.xlabel('Round')
            plt.ylabel('Client ID')
            plt.yticks(np.arange(len(all_clients)) + 0.5, all_clients)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/client_selection_heatmap.png", dpi=300)
            plt.close()

def plot_outlier_detection_analysis(df, output_dir):
    """
    Analyze outlier detection effectiveness.
    """
    if 'outlier_ids' not in df.columns:
        return

    # Process the outlier_ids column if it's stored as a string representation of a list
    if df['outlier_ids'].dtype == 'object':
        # Check if the first non-null value is a string and convert if needed
        first_valid = df['outlier_ids'].dropna().iloc[0] if not df['outlier_ids'].dropna().empty else None
        if isinstance(first_valid, str):
            try:
                df['outlier_ids'] = df['outlier_ids'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            except (SyntaxError, ValueError):
                print("Warning: Could not parse 'outlier_ids' column as a list. Skipping outlier detection analysis.")
                return

    # Count outliers per round
    df['outlier_count'] = df['outlier_ids'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Plot outlier count over rounds
    plt.figure(figsize=(15, 10))

    # Group by round
    grouped = df.groupby('round').agg({
        'outlier_count': ['mean', 'std']
    }).reset_index()

    # Flatten multi-level columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

    plt.plot(
        grouped['round'],
        grouped['outlier_count_mean'],
        marker='o'
    )
    plt.fill_between(
        grouped['round'],
        grouped['outlier_count_mean'] - grouped['outlier_count_std'],
        grouped['outlier_count_mean'] + grouped['outlier_count_std'],
        alpha=0.2
    )

    plt.title('Outlier Detection Count Over Rounds', fontsize=14)
    plt.xlabel('Round')
    plt.ylabel('Number of Outliers Detected')
    plt.grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/outlier_detection_over_time.png", dpi=300)
    plt.close()

    # Compare outlier detection across different parameters
    for param in ['GRAD_MODE', 'IS_SYBIL', 'N_ADV']:
        if param in df.columns and len(df[param].unique()) > 1:
            plt.figure(figsize=(12, 8))
            sns.boxplot(x=param, y='outlier_count', data=df)
            plt.title(f'Outlier Detection by {param}', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            if param == 'IS_SYBIL':
                plt.xlabel('Sybil Attack Enabled')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/outlier_detection_by_{param}.png", dpi=300)
            plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze selection patterns in backdoor attacks")
    parser.add_argument("--data", required=True, help="Path to all_rounds CSV file")
    parser.add_argument("--output", default="./selection_analysis", help="Output directory")

    args = parser.parse_args()

    # Load data
    try:
        all_rounds_df = pd.read_csv(args.data)

        # Convert boolean columns
        if 'IS_SYBIL' in all_rounds_df.columns:
            all_rounds_df['IS_SYBIL'] = all_rounds_df['IS_SYBIL'].astype(bool)

        # Run analysis
        analyze_selection_patterns(all_rounds_df, args.output)
        print(f"Selection pattern analysis complete. Results saved to {args.output}")
    except Exception as e:
        print(f"Error: {e}")