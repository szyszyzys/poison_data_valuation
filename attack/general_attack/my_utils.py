from pathlib import Path

import numpy as np

from daved.src.utils import get_gaussian_data, get_mimic_data, get_fitzpatrick_data, get_bone_data, get_drug_data, \
    split_data, get_cost_function


def manipulate_cost_function(costs, method="scale", factor=1.0, offset=0.0, power=1.0):
    """
    Manipulate the cost function to apply different transformations.

    Parameters:
    - costs (np.ndarray): Array of original costs for each data point.
    - method (str): Method to manipulate the cost function. Options:
        - "scale": Scale costs by a factor.
        - "offset": Add a constant offset to costs.
        - "power": Raise costs to a specified power (non-linear transformation).
        - "log": Apply a logarithmic transformation to costs (assumes positive values).
        - "combine": Combine scale, offset, and power transformations.
    - factor (float): Scaling factor for "scale" or "combine" methods.
    - offset (float): Offset to add for "offset" or "combine" methods.
    - power (float): Power to raise costs to for "power" or "combine" methods.

    Returns:
    - np.ndarray: Manipulated costs after applying the specified transformation.
    """

    if method == "scale":
        # Scale costs by the specified factor
        manipulated_costs = costs * factor

    elif method == "offset":
        # Add a constant offset to costs
        manipulated_costs = costs + offset

    elif method == "power":
        # Raise costs to the specified power
        manipulated_costs = np.power(costs, power)

    elif method == "log":
        # Apply a logarithmic transformation to costs (assumes costs are positive)
        manipulated_costs = np.log(costs + 1)  # Adding 1 to avoid log(0)

    elif method == "combine":
        # Apply a combination of scaling, offsetting, and raising to a power
        manipulated_costs = (costs * factor + offset) ** power

    else:
        raise ValueError("Invalid method for manipulating the cost function.")

    return manipulated_costs


def generate_costs(X=None, method="random_uniform", cost_range=(1.0, 5.0), assigned_cost=None):
    """
    Generate different types of costs for data points.

    Parameters:
    - X (np.ndarray): Feature matrix (n_samples, n_features).
    - method (str): Method for generating costs. Options:
        - "random_uniform": Random values within cost_range.
        - "random_choice": Random choice from cost_range.
        - "feature_based": Scale based on a feature (first feature by default).
        - "constant": Assign the same cost to all data points.
        - "assigned": Use pre-assigned costs.
    - cost_range (tuple): Range of possible costs (min, max).
    - assigned_cost (np.ndarray, optional): Pre-assigned costs for each data point.

    Returns:
    - np.ndarray: Array of costs for each data point.
    """

    if method == "random_uniform":
        # Random values uniformly distributed within cost_range
        costs = np.random.uniform(cost_range[0], cost_range[1], size=X.shape[0]).astype(np.single)

    elif method == "random_choice":
        # Randomly choose cost values from cost_range (assumes discrete set of choices)
        costs = np.random.choice(np.linspace(cost_range[0], cost_range[1], 10), size=X.shape[0]).astype(np.single)

    elif method == "feature_based":
        # Scale costs based on the first feature of X, normalized to cost_range
        feature = X[:, 0]
        costs = (feature - feature.min()) / (feature.max() - feature.min())
        costs = costs * (cost_range[1] - cost_range[0]) + cost_range[0]
        costs = costs.astype(np.single)

    elif method == "constant":
        # Assign the same cost to all data points, set at the midpoint of cost_range
        costs = np.full(X.shape[0], (cost_range[0] + cost_range[1]) // 2, dtype=np.single)

    elif method == "assigned" and assigned_cost is not None:
        # Use pre-assigned costs if provided
        costs = assigned_cost.astype(np.single)

    else:
        raise ValueError("Invalid method or assigned_cost not provided for 'assigned' method.")

    return costs


def get_data(
        dataset="gaussian",
        data_dir="./data",
        random_state=0,
        num_seller=10000,
        num_buyer=100,
        num_val=100,
        dim=100,
        noise_level=1,
        use_cost=True,
        cost_gen_mode="None",
        cost_func="linear",
        recompute_embeddings=False,
        assigned_cost=None
):
    total_samples = num_seller + num_buyer + num_val
    data_dir = Path(data_dir)
    match dataset:
        case "gaussian":
            data = get_gaussian_data(total_samples, dim=dim, noise=noise_level)
        case "mimic":
            data = get_mimic_data(total_samples, data_dir=data_dir)
        case "fitzpatrick":
            data = get_fitzpatrick_data(
                total_samples,
                recompute_embeddings=recompute_embeddings,
                data_dir=data_dir,
                model_name='clip',
            )
        case "bone":
            data = get_bone_data(
                total_samples,
                recompute_embeddings=recompute_embeddings,
                data_dir=data_dir,
                model_name='clip',
            )
        case "drug":
            data = get_drug_data(
                total_samples,
                recompute_embeddings=recompute_embeddings,
                data_dir=data_dir,
                model_name='gpt2',
            )
        case _:
            raise Exception("Dataset not found")

    X = data["X"]
    y = data["y"]
    coef = data.get("coef")
    index = data.get("index")

    if use_cost:
        costs = generate_costs(X, cost_gen_mode)
        ret = split_data(
            num_buyer, num_val, random_state=random_state, X=X, y=y, costs=costs, index=index,
        )
    else:
        ret = split_data(num_buyer, num_val, random_state=random_state, X=X, y=y, index=index)

    # dict(
    #     X_sell=X_sell,
    #     y_sell=y_sell,
    #     costs_sell=costs_sell,
    #     X_buy=X_buy,
    #     y_buy=y_buy,
    #     costs_buy=costs_buy,
    #     X_val=X_val,
    #     y_val=y_val,
    #     costs_val=costs_val,
    #     index_buy=index_buy,
    #     index_val=index_val,
    #     index_sell=index_sell,
    # )

    # ret contain the information of the data from different party

    ret['img_paths'] = data['img_paths']
    ret["coef"] = coef
    ret["use_cost"] = use_cost
    ret["cost_func"] = cost_func

    match dataset, use_cost:
        case "gaussian", True:  # gaussian, no costs
            ret["y_buy"] = ret["X_buy"] @ coef
        case _, False:  # not gaussian, no costs
            pass
        case "gaussian", True:  # gaussian with costs
            h = get_cost_function(cost_func)
            e = noise_level * np.random.randn(ret["X_sell"].shape[0])
            print(type(ret["costs_sell"]))
            ret["y_sell"] = (
                    np.einsum("i,ij->ij", h(ret["costs_sell"]), ret["X_sell"]) @ coef + e
            )
            ret["y_buy"] = ret["X_buy"] @ coef
        case _, False:  # not gaussian with costs
            h = get_cost_function(cost_func)
            e = noise_level * np.random.randn(ret["X_sell"].shape[0])
            print(f'{e[:10].round(2)=}', e.mean())
            print(f'{ret["y_sell"][:10]}   {ret["y_sell"].mean()=}')
            print(f'{h(ret["costs_sell"][:10])=}')
            e *= ret["y_sell"].mean() / h(ret["costs_sell"])
            print(f'{e[:10].round(2)=}', e.mean())
            ret["y_sell"] = ret["y_sell"] + e
            print(f'{ret["y_sell"].mean()=}')

    return ret
