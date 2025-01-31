from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from attack.privacy_attack.attack_ds import reconstruct_X_buy
from attack.privacy_attack.evaluation import evaluate_embeddings
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
    eval_results = defaultdict(list)
    for k in range(50, min(len(adv.cur_data) // 2, 500), 50):  # Ensure k doesn't exceed data size
        # Select top-k adversarial samples
        selected_indices = np.argsort(adv_weights)[-k:]
        X_adv_selected = adv.cur_data[selected_indices]

        # Reconstruction: Simple mean initialization + gradient-based refinement
        X_buy_hat_init = np.mean(X_adv_selected, axis=0)
        X_buy_hat = reconstruct_X_buy(X_adv_selected, adv.cur_data, k, X_buy_hat_init, alpha=0.01, max_iter=100)

        # Baseline: Mean of selected points
        baseline_res = np.mean(X_adv_selected, axis=0)

        # Evaluate
        eval_results["our"].append(evaluate_embeddings(data_manager.X_buy, X_buy_hat))
        eval_results["baseline"].append(evaluate_embeddings(data_manager.X_buy, baseline_res))

    return eval_results


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


if __name__ == "__main__":
    # Run experiments

    exp_config = {
        "adversary_ratio": 1,
        "num_seller_points": 1000,
        "num_buyer_points": 100,
        "seller_configs": [
            {'id': 'adv1', 'type': 'adversary'},
            # {'id': 'normal1', 'type': 'normal'},
            # {'id': 'normal2', 'type': 'normal'},
            # {'id': 'normal3', 'type': 'normal'}
        ]
    }

    # Create a mapping for the parameter names
    param_mapping = {
        "num_seller": exp_config.get("num_seller_points", 1000),
        "num_buyer": exp_config.get("num_buyer_points", 100),
        "adversary_ratio": exp_config.get("adversary_ratio", 0.0),
        "seller_configs": exp_config.get("seller_configs", [])
    }

    eval_results = run_attack_experiment(
        dataset_type="fitzpatrick",
        dim=100,
        **param_mapping  # Unpack the mapped parameters
    )

    # Plot results
    plot_results(eval_results)
    print(eval_results)
