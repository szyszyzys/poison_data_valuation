from attack.privacy_attack.malicious_seller import MaliciousDataSeller

from marketplace.data_manager import DatasetManager
from marketplace.data_selector import SelectionStrategy
from marketplace.market.markplace_gradient import DataMarketplaceFederated
from marketplace.market_mechanism.martfl import Aggregator
from marketplace.seller.gradient_seller import GradientSeller, AdvancedBackdoorAdversarySeller
from marketplace.seller.seller import BaseSeller
from marketplace.utils.gradient_market_utils.data_processor import get_data_set


def run_fl_experiment(dataset_name, seller_config, buyer_config, n_sellers, n_adversaries, model_structure,
                      global_rounds=100, backdoor_target_label=0, trigger_type: str = "blended_patch", exp_name="/"):
    # load the dataset
    save_path = f"/results/{exp_name}/"
    n_buyer = 1
    buyer_cid = 0
    # setup the data set for the participants
    client_loaders = get_data_set(dataset_name, num_clients=n_sellers + n_buyer, iid=True)

    # config the buyer
    buyer = GradientSeller(seller_id="buyer", local_data=client_loaders[buyer_cid], dataset_name=dataset_name,
                           save_path=save_path)

    # config the marketplace
    aggregator = Aggregator(save_path=save_path,
                            n_seller=n_sellers,
                            n_adversaries=n_adversaries,
                            model_structure=model_structure,
                            dataset_name=dataset_name,
                            quantization=False,
                            )
    marketplace = DataMarketplaceFederated(aggregator,
                                           selection_method="martfl")

    # config the seller
    for cid, loader in client_loaders.items():
        if cid == buyer_cid:
            # set the buyer as cid 0 for data
            continue
        if cid <= n_adversaries:
            cur_id = f"adv_{cid}"
            current_seller = AdvancedBackdoorAdversarySeller(seller_id=cur_id,
                                                             local_data=loader,
                                                             target_label=backdoor_target_label,
                                                             trigger_type=trigger_type
                                                             )
        else:
            cur_id = f"seller_{cid}"
            current_seller = GradientSeller(seller_id=cur_id, local_data=client_loaders[buyer_cid],
                                            dataset_name=dataset_name, save_path=save_path)
        marketplace.register_seller(cur_id, current_seller)

    # config the attack test set.

    # Start gloal round
    for gr in range(global_rounds):
        # compute the buyer gradient as the reference point
        buyer_gradient = buyer.get_gradient()
        # train the attack model
        marketplace.train_federated_round(round_number=gr,
                                          buyer_gradient = buyer_gradient,
                                          test_dataloader_buyer_local=None,
                                          test_dataloader_global=None,
                                          clean_loader=None, triggered_loader=None,
                                          loss_fn=None, )

        # test the attack performance of each round

    # record the result for each seller
    for s in seller:
        s.save_statistics()

    # record the attack result for the final round

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
