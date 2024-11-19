import os

import numpy as np


def evaluate_poisoning_attack(
        args,
        dataset='./data',
        data_dir='./data',
        batch_size=16,
        csv_path="druglib/druglib.csv",
        img_path="/images",
        num_buyer=10,
        num_seller=1000,
        num_val=1,
        num_dim=100,
        max_eval_range=50,
        eval_step=5,
        num_iters=500,
        reg_lambda=0.1,
        attack_strength=0.1,
        save_results_flag=True,
        result_dir='results',
        save_name='attack_evaluation',
        num_select=100,
        attack_type="data",
        adversary_ratio=0.25,
        emb_model=None,
        img_preprocess=None,
        cost_manipulation_method="undercut_target",
        emb_model_name="clip",
        **kwargs
):
    """
    Run the attack evaluation experiment.

    Parameters:
    - data_dir (str): Directory where the CSV file is located.
    - csv_path (str): Path to the CSV file containing the data.
    - num_buyer (int): Number of buyer data points.
    - num_seller (int): Number of seller data points.
    - num_val (int): Number of validation data points.
    - max_eval_range (int): Maximum budget value for evaluation.
    - eval_step (int): Step size for budget evaluation.
    - num_iters (int): Number of iterations for the selection algorithm.
    - reg_lambda (float): Regularization parameter for the selection algorithm.
    - attack_strength (float): Strength of the attack when modifying features.
    - save_results_flag (bool): Whether to save the results.
    - result_dir (str): Directory to save the results.
    - save_name (str): Filename for the saved results.

    Returns:
    - results (dict): Dictionary containing evaluation metrics and other relevant data.
    """
    # Step 1: Load and preprocess data
    data = get_data(
        dataset=dataset,
        num_buyer=num_buyer * batch_size,
        num_seller=num_seller,
        num_val=num_val,
        dim=num_dim,
        noise_level=args.noise_level,
        random_state=args.random_seed,
        cost_gen_mode=args.cost_gen_mode,
        use_cost=args.use_cost,
        cost_func=args.cost_func,
        assigned_cost=None,
    )
    # Extract relevant data
    x_s = data["X_sell"].astype(np.float32)
    y_s = data["y_sell"].astype(np.float32)
    x_b = data["X_buy"].astype(np.float32)
    y_b = data["y_buy"].astype(np.float32)

    result_dir = f'{result_dir}/privacy/attack_{attack_type}_num_buyer_{num_buyer}_num_seller_{num_seller}_querys_{args.batch_size}/'
    figure_path = f"{result_dir}/figures/"

    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    # todo change the costs
    costs = data.get("costs_sell")
    index_s = data['index_sell']
    index_b = data['index_buy']
    index_v = data['index_val']
    img_paths = data['img_paths']
    print("Data type of index_s:", len(img_paths))
    print(f"Seller Data Shape: {x_s.shape}".center(40, "="))
    print(f"Buyer Data Shape: {x_b.shape}".center(40, "="))
    if costs is not None:
        print(f"Costs Shape: {costs.shape}".center(40, "="))

    # adversary preparation, sample partial data for the adversary
    num_adversary_samples = int(len(x_s) * adversary_ratio)
    adversary_indices = np.random.choice(len(x_s), size=num_adversary_samples, replace=False)
    adv = Adv(x_s, y_s, costs, adversary_indices, emb_model_name, device, img_path)

    # Evaluate the peformance
    eval_range = list(range(1, 30, 1)) + list(
        range(30, args.max_eval_range, args.eval_step)
    )

    benign_node_sampling_result_dict = {}
    benign_training_results, benign_selection_info = sampling_run_one_buyer(x_b, y_b, x_s, y_s, eval_range, costs=costs,
                                                                            args=args, figure_path=figure_path)

    # transform the initial result into dictionary
    for cur_info in benign_selection_info:
        query_number = cur_info["query_number"]
        benign_node_sampling_result_dict[query_number] = cur_info

    print(f"Done initial run, number of queries: {len(benign_selection_info)}")
    # Step 3: Identify Selected and Unselected Data Points
    # For each batch (buyer query), perform the reconstruction
    attack_result_dict = {}

    # For different query, perform the attacks.
    for query_n, info_dic in enumerate(benign_selection_info):
        cur_query_num = info_dic["query_number"]
        m_cur_weight = info_dic["multi_step_weights"]
        s_cur_weight = info_dic["single_step_weights"]
        x_test = info_dic["test_x"]
        y_test = info_dic["test_y"]
        attack_result_path = f"./{figure_path}/poisoned_sampling_query_number_{query_n}"
        # for current data batch, find which points are selected
        # img = Image.open(img_path)
        # embedding = inference_func(preprocess(img)[None].to(device))
        # embeddings.append(embedding.cpu())
        # Get the clean result.
        selected_indices_initial, unsampled_indices_initial = identify_selected_unsampled(
            weights=m_cur_weight,
            num_select=num_select,
        )
        selected_adversary_indices = np.intersect1d(selected_indices_initial, adversary_indices)
        unsampled_adversary_indices = np.intersect1d(unsampled_indices_initial, adversary_indices)

        print(f"Initial Selected Indices: {len(selected_indices_initial)}")
        print(f"Number of Unselected Data Points: {len(unsampled_indices_initial)}")
        print(f"Initial Selected Indices from Adversary: {len(selected_adversary_indices)}")
        print(f"Number of Unselected Data Points from Adversary: {len(unsampled_adversary_indices)}")

        # Step 5: Perform Attack on Unselected Data Points
        modified_images_path = os.path.join(
            result_dir,
            f'step_{args.attack_steps}_lr_{args.attack_lr}_reg_{args.attack_reg}_advr_{adversary_ratio}',
            f'target_query_no_{cur_query_num}'
        )

        os.makedirs(modified_images_path, exist_ok=True)

        # start attack
        attack_param = {
            "target_query": cur_query_num,
            "cost_manipulation_method": "undercut_target",
            "selected_indices": selected_adversary_indices,
            "unselected_indices": unsampled_adversary_indices,
            "use_cost": False,
            "emb_model": emb_model,
            "img_preprocess": img_preprocess,
            "device": device,
            "output_dir": modified_images_path,
            "global_selected_indices": selected_indices_initial,
            "poison_rate": args.poison_rate
        }

        # manipulate the images
        manipulated_img_dict = adv.attack("data_manipulation", attack_param, x_s, costs, img_paths)

        # clone the original x_s, insert perturbed embeddings into the x_s
        x_s_clone = copy.deepcopy(x_s)

        for img_idx, info in manipulated_img_dict.items():
            modified_embedding = info["m_embedding"]
            x_s_clone[img_idx] = modified_embedding

        # use the sample query to perform the attack.
        model_training_result, data_sampling_result = sampling_run_one_buyer(
            x_test, y_test, x_s_clone, y_s, eval_range, costs=costs, args=args, figure_path=attack_result_path
        )

        attack_result_dict[query_n] = {
            "model_training_result": model_training_result,
            "data_sampling_result": data_sampling_result[0]
        }
