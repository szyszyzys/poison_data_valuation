import numpy as np
from matplotlib import pyplot as plt

from attack.general_attack.my_utils import read_csv
from attack.privacy_attack.attack_o import run_reconstruction_attack_eval
from attack.privacy_attack.malicious_seller import AdversarySeller
from attack.privacy_attack.seller import BaseSeller
from attack.utils.data_manager import DatasetManager
from attack.utils.data_market import DataMarketplace
from attack.utils.data_selector import SelectionStrategy


# def run_attack_experiments():
#     # Define experiment configuration
#     config = ExperimentConfig(
#         n_queries=100,
#         n_rounds=10,
#         query_types=["random", "cluster", "mixture"],
#         attack_types=[
#             AttackType.INFO_MATRIX,
#             AttackType.GRADIENT,
#             AttackType.SELECTION_PATTERN,
#             AttackType.ENSEMBLE
#         ],
#         embedding_dims=[32, 64, 128],
#         n_seller_points=[1000, 5000, 10000],
#         noise_levels=[0.0, 0.1, 0.2]
#     )
#
#     # Example scenarios to test:
#     scenarios = [
#         # Scenario 1: Basic vulnerability testing
#         {
#             'name': 'basic_vulnerability',
#             'embedding_dim': 64,
#             'n_points': 1000,
#             'query_type': 'random',
#             'noise_level': 0.0
#         },
#
#         # Scenario 2: High dimensionality
#         {
#             'name': 'high_dim',
#             'embedding_dim': 128,
#             'n_points': 1000,
#             'query_type': 'random',
#             'noise_level': 0.0
#         },
#
#         # Scenario 3: Large dataset
#         {
#             'name': 'large_dataset',
#             'embedding_dim': 64,
#             'n_points': 10000,
#             'query_type': 'random',
#             'noise_level': 0.0
#         },
#
#         # Scenario 4: Clustered queries
#         {
#             'name': 'clustered',
#             'embedding_dim': 64,
#             'n_points': 1000,
#             'query_type': 'cluster',
#             'noise_level': 0.0
#         },
#
#         # Scenario 5: Noisy environment
#         {
#             'name': 'noisy',
#             'embedding_dim': 64,
#             'n_points': 1000,
#             'query_type': 'random',
#             'noise_level': 0.2
#         },
#
#         # Scenario 6: Mixed distribution
#         {
#             'name': 'mixed',
#             'embedding_dim': 64,
#             'n_points': 1000,
#             'query_type': 'mixture',
#             'noise_level': 0.0
#         }
#     ]
#
#     # Initialize pipeline
#     pipeline = AttackPipeline(daved_func=daved_selection, config=config)
#
#     # Run experiments for each scenario
#     all_results = {}
#     for scenario in scenarios:
#         print(f"\nRunning scenario: {scenario['name']}")
#
#         # Run experiments
#         results = pipeline.run_single_experiment(
#             seller_embeddings=pipeline.generate_experiment_data(
#                 n_points=scenario['n_points'],
#                 dim=scenario['embedding_dim'],
#                 query_type=scenario['query_type'],
#                 noise_level=scenario['noise_level']
#             )['seller_embeddings'],
#             query=pipeline.generate_experiment_data(
#                 n_points=1,
#                 dim=scenario['embedding_dim'],
#                 query_type=scenario['query_type'],
#                 noise_level=scenario['noise_level']
#             )['query'],
#             attack_type=AttackType.ENSEMBLE  # Run all attacks via ensemble
#         )
#
#         all_results[scenario['name']] = results
#
#     # Analyze results
#     analysis = pipeline.analyze_results(all_results)
#
#     # Visualize results
#     pipeline.visualize_results(analysis, save_path='experiment_results')
#
#     return all_results, analysis
#
# # Function to print detailed results
# def print_experiment_results(analysis: Dict):
#     print("\nExperiment Results Summary:")
#     print("==========================")
#
#     for scenario, results in analysis.items():
#         print(f"\nScenario: {scenario}")
#         print("-" * (len(scenario) + 10))
#
#         for attack_type, metrics in results.items():
#             print(f"\n{attack_type.value}:")
#             print(f"  Success Rate: {metrics['mean_success']:.3f} ± {metrics['std_success']:.3f}")
#             print(f"  Privacy Breach Rate: {metrics['privacy_breach_rate']:.3f}")
#             print(f"  Mean Distance: {metrics['mean_distance']:.3f}")

def save_attack_results(results_list, file_prefix="attack_results"):
    """
    Convert a list of dictionaries to a DataFrame and save to a timestamped CSV file.

    :param results_list: List of dictionaries containing attack results.
    :param file_prefix: Prefix for the output file name.
    :return: The filename of the saved CSV file.
    """
    # Convert list to DataFrame
    df = pd.DataFrame(results_list)

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the filename
    filename = f"{file_prefix}_{timestamp}.csv"

    # Save DataFrame to CSV
    df.to_csv(filename, index=False)

    print(f"Attack results saved as {filename}")
    return filename


def plot_attacks_for_single_k(df, k_value, metric='selection_f1', title=None):
    """
    Plots a bar chart comparing different methods on a single metric (e.g., 'selection_f1')
    for a fixed selection size k_value.

    Args:
        df       : DataFrame with columns ['method', 'k', <metric>].
        k_value  : The selection size we want to filter on.
        metric   : The metric column name to plot (string).
        title    : Optional plot title.
    """
    # Filter for rows where df['k'] == k_value
    df_k = df[df['k'] == k_value].copy()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_k, x='method', y=metric)
    plt.title(title if title else f"{metric} for k={k_value}")
    plt.xlabel("Method")
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha='right')
    plt.ylim(0, df_k[metric].max() * 1.1)  # give some space on top
    plt.grid(True, axis='y')
    plt.show()


def plot_metric_across_selection_sizes(df, metric='selection_f1', title=None):
    """
    Plots how different methods perform on a single metric (e.g., 'selection_f1')
    as the selection size 'k' varies.

    Args:
        df     : a pandas DataFrame with columns ['method', 'k', <metric>]
        metric : the metric column name to plot on the y-axis (string)
        title  : optional plot title
    """
    # Convert 'k' to numeric if needed
    df['k'] = df['k'].astype(int)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='k', y=metric, hue='method', marker='o')
    plt.title(title if title else f"{metric} vs. Selection Size (k)")
    plt.xlabel("Selection Size (k)")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend(title='Method', loc='best')
    plt.show()


def run_attack_experiment(dataset_type="gaussian", dim=100, num_seller=1000,
                          num_buyer=100,
                          adversary_ratio=0.25, seller_configs=None):
    data_manager = DatasetManager(
        dataset_type=dataset_type,
        num_seller=num_seller,
        num_buyer=num_buyer,
        dim=dim,
        use_cost=False
    )

    marketplace, seller_dict = setup(data_manager, adversary_ratio, seller_configs)

    weights, seller_ids = marketplace.get_select_info(data_manager.X_buy, data_manager.y_buy,
                                                      SelectionStrategy.DAVED_MULTI_STEP)
    mask = seller_ids == 'adv1'
    adv_weights = weights[mask]
    adv = seller_dict["adv1"]

    # --- Attack & Evaluation ---
    results = []

    # Example: looping over different k values
    # (Replace adv.cur_data, adv_weights, marketplace, data_manager, and run_reconstruction_attack_eval with your actual objects/functions.)
    for k in range(50, min(len(adv.cur_data) // 2, 500), 50):  # Ensure k doesn't exceed data size
        # Select top-k adversarial samples
        selected_indices = np.argsort(adv_weights)[-k:]

        # Get current market data once per iteration
        current_market_data = marketplace.get_current_market_data()

        # Run different reconstruction attacks
        base_random = run_reconstruction_attack_eval(current_market_data, selected_indices, data_manager.X_buy,
                                                     use_baseline="random")
        base_centroid = run_reconstruction_attack_eval(current_market_data, selected_indices, data_manager.X_buy,
                                                       use_baseline="centroid")
        score_known = run_reconstruction_attack_eval(current_market_data, selected_indices, data_manager.X_buy,
                                                     scenario="score_known")
        score_unknown_ranking = run_reconstruction_attack_eval(current_market_data, selected_indices,
                                                               data_manager.X_buy,
                                                               scenario="selection_only", attack_method="ranking")
        score_unknown_topk = run_reconstruction_attack_eval(current_market_data, selected_indices, data_manager.X_buy,
                                                            scenario="selection_only", attack_method="topk")

        # Create a helper to accumulate each attack's result.
        def append_result(attack_name, result_dict):
            """
            Copy the result dictionary, add metadata, and append to the results list.
            """
            # Make a copy to avoid modifying the original dictionary.
            entry = result_dict.copy()
            entry["attack_type"] = attack_name
            entry["n_selected"] = k
            results.append(entry)

        # Append each attack's results.
        append_result("random", base_random)
        append_result("centroid", base_centroid)
        append_result("score_known", score_known)
        append_result("score_unknown_ranking", score_unknown_ranking)
        append_result("score_unknown_topk", score_unknown_topk)

    # After the loop, save all accumulated results to a single CSV file.
    filename = save_attack_results(results)
    res_df = read_csv(filename)
    for m in ['total_distance', 'avg_distance', 'matching', 'mse']:
        plot_metric_across_selection_sizes(res_df, metric=m, title=f"attack performance ({m})")

    return results


def plot_results(eval_results):
    ks = list(range(50, 500, 50))[:len(eval_results["our"])]  # Match evaluated k values

    plt.figure(figsize=(12, 4))

    # Plot Cosine Similarity
    plt.subplot(131)
    plt.plot(ks, [res["cosine_sim_mean"] for res in eval_results["our"]], label="Our Method")
    plt.plot(ks, [res["cosine_sim_mean"] for res in eval_results["baseline"]], label="Baseline (Mean)")

    plt.xlabel("Number of Selected Samples (k)")
    plt.ylabel("Cosine Similarity mean")
    plt.legend()

    # plt.subplot(132)
    # plt.plot(ks, [res["cosine_sim_std"] for res in eval_results["our"]], label="Our Method")
    # plt.plot(ks, [res["cosine_sim_std"] for res in eval_results["baseline"]], label="Baseline (Mean)")
    # plt.xlabel("Number of Selected Samples (k)")
    # plt.ylabel("Cosine Similarity std")
    # plt.legend()

    # Plot MSE
    plt.subplot(132)
    plt.plot(ks, [res["mse"] for res in eval_results["our"]], label="Our Method")
    plt.plot(ks, [res["mse"] for res in eval_results["baseline"]], label="Baseline (Mean)")
    plt.xlabel("Number of Selected Samples (k)")
    plt.ylabel("MSE")
    plt.legend()

    # Plot Neighborhood Overlap
    plt.subplot(133)
    plt.plot(ks, [res["neighborhood_overlap"] for res in eval_results["our"]], label="Our Method")
    plt.plot(ks, [res["neighborhood_overlap"] for res in eval_results["baseline"]], label="Baseline (Mean)")
    plt.xlabel("Number of Selected Samples (k)")
    plt.ylabel("Neighborhood Overlap")
    plt.legend()

    plt.tight_layout()
    plt.show()


def setup(data_manager: DatasetManager, adversary_ratio=0.25, seller_configs=None):
    """Setup marketplace with normal and adversarial sellers"""

    # Create marketplace
    marketplace = DataMarketplace()
    # Get data allocations

    allocations = data_manager.allocate_data_to_sellers(
        seller_configs,
        adversary_ratio=adversary_ratio
    )
    seller_dict = {}
    # Create and register sellers
    for config in seller_configs:
        seller_id = config['id']
        seller_data = allocations[seller_id]

        if config['type'] == 'adversary':
            seller = AdversarySeller(
                seller_id=seller_id,
                dataset=seller_data['X'],
            )
        else:
            seller = BaseSeller(
                seller_id=seller_id,
                dataset=seller_data['X']
            )
        seller_dict[seller_id] = seller
        marketplace.register_seller(seller_id, seller)

    return marketplace, seller_dict


# if __name__ == "__main__":
#     # Run experiments
#
#     exp_config = {
#         "adversary_ratio": 1,
#         "num_seller_points": 1000,
#         "num_buyer_points": 100,
#         "seller_configs": [
#             {'id': 'adv1', 'type': 'adversary'},
#             # {'id': 'normal1', 'type': 'normal'},
#             # {'id': 'normal2', 'type': 'normal'},
#             # {'id': 'normal3', 'type': 'normal'}
#         ]
#     }
#
#     # Create a mapping for the parameter names
#     param_mapping = {
#         "num_seller": exp_config.get("num_seller_points", 1000),
#         "num_buyer": exp_config.get("num_buyer_points", 100),
#         "adversary_ratio": exp_config.get("adversary_ratio", 0.0),
#         "seller_configs": exp_config.get("seller_configs", [])
#     }
#
#     eval_results = run_attack_experiment(
#         dataset_type="fitzpatrick",
#         dim=100,
#         **param_mapping  # Unpack the mapped parameters
#     )
#
#     # Plot results
#     plot_results(eval_results)
#     print(eval_results)

def parse_args():
    """
    Parse command-line arguments for the experiment configuration.

    :return: Dictionary containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run attack experiment with configurable parameters.")

    # Add arguments with default values from exp_config
    parser.add_argument("--dataset", type=str, default="fitzpatrick", help="Target dataset")
    parser.add_argument("--num_seller", type=int, default=1000, help="Number of seller points")
    parser.add_argument("--num_buyer", type=int, default=100, help="Number of buyer points")
    parser.add_argument("--adversary_ratio", type=float, default=1.0, help="Adversary ratio in the dataset")
    parser.add_argument("--seller_configs", type=str, default="adv1:adversary",
                        help="Comma-separated list of seller configurations (format: id:type, e.g., adv1:adversary,normal1:normal)")

    args = parser.parse_args()

    # Process seller_configs into a list of dictionaries
    seller_configs = []
    if args.seller_configs:
        for seller in args.seller_configs.split(","):
            seller_id, seller_type = seller.split(":")
            seller_configs.append({'id': seller_id, 'type': seller_type})

    # Return as a dictionary
    return {
        "num_seller": args.num_seller,
        "num_buyer": args.num_buyer,
        "adversary_ratio": args.adversary_ratio,
        "seller_configs": seller_configs
    }, args


if __name__ == "__main__":
    # Parse command-line arguments
    param_mapping, args = parse_args()
    print("start reconstrunction attack")
    print(param_mapping)

    # Run experiment with parsed arguments
    eval_results = run_attack_experiment(
        dataset_type=args.dataset,
        dim=100,
        **param_mapping  # Unpack arguments
    )

    # Plot and print results
    plot_results(eval_results)
    print(eval_results)
