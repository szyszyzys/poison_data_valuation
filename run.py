from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from attack.privacy_attack.attack_ds import reconstruct_X_buy, reconstruct_X_buy_fim
from attack.privacy_attack.evaluation import evaluate_embeddings
from attack.privacy_attack.malicious_seller import AdversarySeller
from attack.privacy_attack.seller import BaseSeller
from attack.utils.data_manager import DatasetManager
from attack.utils.data_market import DataMarketplace
from attack.utils.data_selector import SelectionStrategy


def run_attack_experiment(dataset_type="gaussian", dim=100, num_seller=1000,
                          num_buyer=100,
                          adversary_ratio=0.25, seller_configs=None,
                          selection_method=SelectionStrategy.DAVED_MULTI_STEP, attack_method=""):
    data_manager = DatasetManager(
        dataset_type=dataset_type,
        num_seller=num_seller,
        num_buyer=num_buyer,
        dim=dim,
        use_cost=False
    )

    marketplace, seller_dict = setup(data_manager, adversary_ratio, seller_configs)

    weights, seller_ids = marketplace.get_select_info(data_manager.X_buy, data_manager.y_buy,
                                                      selection_method)
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
        match attack_method:
            case "ds1":
                X_buy_hat = reconstruct_X_buy(X_adv_selected, adv.cur_data, k, X_buy_hat_init, alpha=0.1, max_iter=100)
            case "ds2":
                X_buy_hat_fim = reconstruct_X_buy_fim(selected_indices, adv.cur_data, X_buy_hat_init, alpha=0.1,
                                                      max_iter=1000)
            case "o1":
                X_buy_hat_fim = reconstruct_X_buy_fim(selected_indices, adv.cur_data, X_buy_hat_init, alpha=0.1,
                                                      max_iter=1000)
            case _:
                raise Exception("Attack not found")

        # Baseline: Mean of selected points
        baseline_res = np.mean(X_adv_selected, axis=0)
        # Evaluate
        eval_results["our"].append(evaluate_embeddings(data_manager.X_buy, X_buy_hat))
        eval_results["baseline"].append(evaluate_embeddings(data_manager.X_buy, baseline_res))
        eval_results["random"].append(evaluate_embeddings(data_manager.X_buy, baseline_res))
        # eval_results["our new"].append(evaluate_embeddings(data_manager.X_buy, X_buy_hat_new))
        eval_results["our fim"].append(evaluate_embeddings(data_manager.X_buy, X_buy_hat_fim))

    return eval_results


def plot_results(eval_results):
    ks = list(range(50, 500, 50))[:len(eval_results["our"])]  # Match evaluated k values

    plt.figure(figsize=(12, 4))

    # Plot Cosine Similarity
    plt.subplot(131)
    plt.plot(ks, [res["cosine_sim_mean"] for res in eval_results["our"]], label="Our Method")
    plt.plot(ks, [res["cosine_sim_mean"] for res in eval_results["baseline"]], label="Baseline (Mean)")
    plt.plot(ks, [res["cosine_sim_mean"] for res in eval_results["our fim"]], label="FIM")

    plt.xlabel("Number of Selected Samples (k)")
    plt.ylabel("Cosine Similarity mean")
    plt.legend()

    # Plot MSE
    plt.subplot(132)
    plt.plot(ks, [res["mse"] for res in eval_results["our"]], label="Our Method")
    plt.plot(ks, [res["mse"] for res in eval_results["baseline"]], label="Baseline (Mean)")
    plt.plot(ks, [res["mse"] for res in eval_results["our fim"]], label="FIM")
    plt.xlabel("Number of Selected Samples (k)")
    plt.ylabel("MSE")
    plt.legend()

    # Plot Neighborhood Overlap
    plt.subplot(133)
    plt.plot(ks, [res["neighborhood_overlap"] for res in eval_results["our"]], label="Our Method")
    plt.plot(ks, [res["neighborhood_overlap"] for res in eval_results["baseline"]], label="Baseline (Mean)")
    plt.plot(ks, [res["neighborhood_overlap"] for res in eval_results["our fim"]], label="FIM")
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
        "num_seller_points": 2000,
        "num_buyer_points": 1,
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
        "seller_configs": exp_config.get("seller_configs", []),
        "selection_method": SelectionStrategy.DAVED_SINGLE_STEP,
    }

    #             case "gaussian":
    #         case "mimic":
    #         case "fitzpatrick":
    #         case "bone":
    # case "drug":

    eval_results = run_attack_experiment(
        dataset_type="gaussian",
        dim=100,
        **param_mapping  # Unpack the mapped parameters
    )

    # Plot results
    plot_results(eval_results)
    print(eval_results)
