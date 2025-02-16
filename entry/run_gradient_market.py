



from collections import defaultdict

from attack.privacy_attack.malicious_seller import MaliciousDataSeller
from marketplace.market.markplace_fl import DataMarketplaceFederated
from marketplace.market_mechanism.martfl import Aggregator
from marketplace.seller.seller import BaseSeller
from marketplace.data_manager import DatasetManager
from marketplace.data_selector import SelectionStrategy


def run_fl_experiment(dataset_type="gaussian", dim=100, num_seller=1000,
                      num_buyer=100,
                      adversary_ratio=0.25, seller_configs=None,
                      selection_method=SelectionStrategy.DAVED_MULTI_STEP, attack_method="",
                      max_eval_range_selection_num=500, eval_step=50, n_runs=100, buyer_size=1):
    data_manager = DatasetManager(
        dataset_type=dataset_type,
        num_seller=num_seller,
        num_buyer=num_buyer,
        dim=dim,
        use_cost=False
    )

    marketplace, seller_dict = setup(data_manager, adversary_ratio, seller_configs)
    eval_results = defaultdict(list)
    x_s, y_s, costs, seller_ids = marketplace.get_current_market_data()
    selection_errors = defaultdict(list)
    attack_result = defaultdict(list)
    aggregator = Aggregator(global_model=model, device=torch.device("cpu"), experiment_name="my_exp")
    marketplace = DataMarketplaceFederated(aggregator=aggregator,
                                           selection_method="fedavg",
                                           learning_rate=0.01)
    return eval_results

def setup(data_manager: DatasetManager, adversary_ratio=0.25, seller_configs=None):
    """Setup marketplace with normal and adversarial sellers"""

    # Create marketplace
    marketplace = DataMarketplaceData()
    # Get data allocations

    allocations = data_manager.allocate_data_to_sellers(
        seller_configs,
        adversary_ratio=adversary_ratio
    )
    seller_dict = {}
    print(f"Current seller info {seller_dict}")
    # Create and register sellers
    for config in seller_configs:
        seller_id = config['id']
        seller_data = allocations[seller_id]

        if config['type'] == 'adversary':
            seller = MaliciousDataSeller(
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
        "seller_configs": exp_config.get("seller_configs", []),
        "selection_method": SelectionStrategy.DAVED_SINGLE_STEP,
    }

    #             case "gaussian":
    #         case "mimic":
    #         case "fitzpatrick":
    #         case "bone":
    # case "drug":

    eval_results = run_fl_experiment(
        dataset_type="gaussian",
        dim=100,
        **param_mapping  # Unpack the mapped parameters
    )

    # Plot results
