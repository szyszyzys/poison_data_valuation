import os
import pickle
from collections import defaultdict
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
from tqdm import tqdm

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
        use_cost=False,
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


def load_model_and_preprocessor(model_name, device):
    """
    Load the model and preprocessing pipeline based on the model name.

    Args:
        model_name (str): Model to load ("clip" or "resnet").
        device (str): Device to load the model onto ("cpu" or "cuda").

    Returns:
        tuple: (model, preprocess, inference_func)
    """
    if model_name == "clip":
        model, preprocess = clip.load("ViT-B/32", device=device)
        inference_func = model.encode_image
    elif model_name == "resnet":
        model = resnet18(pretrained=True).to(device)
        preprocess = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
                Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Handle grayscale
            ]
        )
        inference_func = lambda x: model(x).flatten(start_dim=1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval()  # Ensure the model is in evaluation mode
    return model, preprocess, inference_func


def embed_image(img, model, preprocess, inference_func, device="cpu", normalize_embeddings=True):
    """
    Embed a single image object using the given model and preprocessing pipeline.

    Args:
        img (PIL.Image.Image): Image object.
        model: The loaded model for embedding.
        preprocess: The preprocessing pipeline.
        inference_func: The inference function of the model.
        device (str): Device to run the model on ("cpu" or "cuda").
        normalize_embeddings (bool): Whether to normalize embeddings to unit vectors.

    Returns:
        torch.Tensor: The embedding of the image.
    """
    try:
        img_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
        embedding = inference_func(img_tensor)
        if normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize to unit vector
        return embedding  # Remain on the device
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")


def embed_images(img_paths, model_name="clip", device="cpu", normalize_embeddings=True):
    """
    Embed multiple images using a specified model (CLIP or ResNet).

    Args:
        img_paths (list[str]): List of paths to images.
        model_name (str): Model to use for embedding. Options: "clip", "resnet".
        device (str): Device to run the model on ("cpu" or "cuda").
        normalize_embeddings (bool): Whether to normalize embeddings to unit vectors.

    Returns:
        torch.Tensor: Concatenated embeddings of all images.
    """
    # Load model and preprocessing pipeline
    model, preprocess, inference_func = load_model_and_preprocessor(model_name, device)

    embeddings = []

    # Process images and extract embeddings
    with torch.inference_mode():
        for img_path in tqdm(img_paths, desc=f"Embedding images with {model_name}"):
            try:
                img = Image.open(img_path).convert("RGB")  # Ensure all images are RGB
                embedding = embed_image(img, model, preprocess, inference_func, device, normalize_embeddings)
                embeddings.append(embedding)
            except Exception as e:
                print(e)

    # Concatenate embeddings
    embeddings = torch.cat(embeddings, dim=0)

    # Clean up to release resources
    del model
    torch.cuda.empty_cache()

    return embeddings


def plot_errors_fixed(results, save_path):
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 8))
    errors = results["errors"]
    eval_range = results["eval_range"]

    quantiles = []
    for i, (k, v) in enumerate(errors.items()):
        err = np.array(v)
        quantiles.append(np.quantile(err, 0.9))
        ms = 5
        match k:
            case "LavaEvaluator":
                k = "LAVA"
            case "InfluenceSubsample":
                k = "Influence"
            case "LeaveOneOut":
                k = "Leave One Out"
            case "KNNShapley":
                k = "KNN Shapley"
            case "DataOob":
                k = "Data-OOB"
            case "attack":
                k = "Attack-DEVAD"
            case _:
                k = k

        match k:
            case k if "DAVED (multi-step)" in k:
                lw = 2
                ls = "-"
                marker = "*"
                ms = ms + 5
            case k if "random" in k.lower():
                lw = 5
                ls = "-"
                marker = ""
            case _:
                lw = 2
                ls = "-"
                marker = "s"
        plt.plot(
            eval_range,
            err.mean(0).squeeze(),
            label=k,
            marker=marker,
            ls=ls,
            lw=lw,
            ms=ms,
        )

    plt.xticks(np.arange(0, max(eval_range), 10), fontsize="x-large")
    # plt.yticks(np.arange(0, 10, 0.5), fontsize='x-large')
    plt.ylim(0, np.median(quantiles))
    plt.xlabel("Number of Datapoints selected", fontsize="xx-large", labelpad=8)
    plt.ylabel("Test\nError", fontsize="xx-large", rotation=0, labelpad=30)
    plt.legend(
        fontsize="xx-large", bbox_to_anchor=(0.5, 1.4), loc="upper center", ncols=2
    )
    plt.tight_layout(pad=0, w_pad=0)
    plt.savefig(save_path, bbox_inches="tight")


def plot_errors_under_budget(results, save_path):
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 8))
    error_under_budgets = results["errors"]
    eval_range = results["eval_range"]

    quantiles = []
    for i, (k, v) in enumerate(error_under_budgets.items()):
        error_per_budget = defaultdict(list)
        for v_i in v:
            for b, e in v_i.items():
                error_per_budget[b].append(e)

        budgets = []
        errors = []
        for b, e in dict(sorted(error_per_budget.items())).items():
            budgets.append(b)
            errors.append(np.mean(e))

        quantiles.append(np.quantile(errors, 0.9))
        ms = 5
        match k:
            case "LavaEvaluator":
                k = "LAVA"
            case "InfluenceSubsample":
                k = "Influence"
            case "LeaveOneOut":
                k = "Leave One Out"
            case "KNNShapley":
                k = "KNN Shapley"
            case "DataOob":
                k = "Data-OOB"
            case _:
                k = k

        match k:
            case k if "Ours" in k:
                lw = 2
                ls = "-"
                marker = "*"
                ms = ms + 5
            case k if "random" in k.lower():
                lw = 5
                ls = "-"
                marker = ""
            case _:
                lw = 2
                ls = "-"
                marker = "s"

        plt.plot(budgets, errors, label=k, marker=marker, ls=ls, lw=lw, ms=ms)

    plt.xticks(np.arange(0, max(eval_range), 10), fontsize="x-large")
    # plt.yticks(np.arange(0, 10, 0.5), fontsize='x-large')
    plt.ylim(0, np.median(quantiles))
    plt.xlabel("Budget", fontsize="xx-large", labelpad=8)
    plt.ylabel("Test\nError", fontsize="xx-large", rotation=0, labelpad=30)
    plt.legend(
        fontsize="xx-large", bbox_to_anchor=(0.5, 1.4), loc="upper center", ncols=2
    )
    plt.tight_layout(pad=0, w_pad=0)
    plt.savefig(save_path, bbox_inches="tight")


def plot_image_selection_rate(figure_path, results, eval_range):
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 8))

    quantiles = []

    for label, data in results.items():
        # Convert data to a numpy array for easier manipulation
        data_array = np.array(data)  # Shape: (num_runs, len(eval_range))

        ms = 5

        match label:
            case label if "benign" in label:
                lw = 2
                ls = "-"
                marker = "*"
                ms = ms + 5
            case label if "malicious" in label.lower():
                lw = 5
                ls = "-"
                marker = ""
            case _:
                lw = 2
                ls = "-"
                marker = "s"

        # Compute the mean selection rate across runs
        mean_rate = data_array.mean(axis=0)
        std_rate = data_array.std(axis=0)  # Optional: Compute standard deviation for error bars

        # Plot the mean selection rate with error bars
        plt.plot(eval_range, mean_rate, label=label, marker=marker,
                 ls=ls,
                 lw=lw,
                 ms=ms, )
        plt.fill_between(
            eval_range,
            mean_rate - std_rate,
            mean_rate + std_rate,
            alpha=0.2
        )  # Add shaded error bars (optional)

    # Add plot labels and legend
    plt.xlabel("Number of Datapoints selected", fontsize="xx-large", labelpad=8)
    plt.ylabel("Selection Rate", fontsize="xx-large", rotation=0, labelpad=30)

    plt.title("Average Selection Rate Across Runs")

    plt.legend(
        fontsize="xx-large", bbox_to_anchor=(0.5, 1.4), loc="upper center", ncols=2
    )

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(pad=0, w_pad=0)
    plt.xticks(np.arange(0, max(eval_range), 10), fontsize="x-large")

    # Save or show the plot
    if figure_path:
        plt.savefig(figure_path, bbox_inches="tight")
        print(f"Figure saved to {figure_path}")
    else:
        plt.show()


def plot_results_data_selection(plot_type, figure_path, results, eval_range):
    if plot_type == "image_selection_rate":
        plot_image_selection_rate(figure_path, results, eval_range)
    print(f"Plot saved to {figure_path}".center(80, "="))


def image_modification(
        modify_info,
        model,
        processor,
        device,
        output_dir='modified_images',
        num_steps=100,
        learning_rate=0.1,
        lambda_reg=0.1,
        epsilon=0.05,
        target_vector=None
):
    """
    Perform the poisoning attack on unsampled images.

    Parameters:
    - unsampled_indices (list): Indices of unsampled data points.
    - image_paths (list): List of image file paths.
    - X_sell (np.array): Feature matrix of seller data.
    - target_vector (np.array): Desired target embedding vector.
    - model (CLIPModel): Pre-trained CLIP model.
    - processor (CLIPProcessor): CLIP processor.
    - device (str): Device to run computations on.
    - output_dir (str): Directory to save modified images.
    - num_steps, learning_rate, lambda_reg, epsilon: Optimization parameters.

    Returns:
    - modified_indices (list): Indices of images that were modified.
    - similarities (dict): Cosine similarities after modification.
    """

    os.makedirs(output_dir, exist_ok=True)

    # save the modification result for all the images
    modify_result = {}

    # perform modification for each images
    for idx, cur_info in tqdm(modify_info.items(), desc="Performing Attack on Unsampled Images"):
        target_vector = cur_info["target_vector"]
        modify_image_path = cur_info["original_img_path"]
        target_img_path = cur_info["target_img_path"]
        target_img_idx = cur_info["target_index"]
        modified_image, modified_embedding, similarity = modify_image(
            image_path=modify_image_path,
            target_vector=target_vector,
            model=model,
            processor=processor,
            device=device,
            num_steps=num_steps,
            learning_rate=learning_rate,
            lambda_reg=lambda_reg,
            epsilon=epsilon
        )

        # Save the modified image
        modified_image_path = os.path.join(output_dir, f'modified_{Path(modify_image_path).name}')
        modified_image.save(modified_image_path)
        o_image_path = os.path.join(output_dir, f'o_{Path(modify_image_path).name}')
        o_image = load_image(modify_image_path)
        o_image = processor(o_image)
        o_image = denormalize_image(o_image)
        # Convert the tensor back to a PIL image
        o_image = transforms.ToPILImage()(o_image)
        # Save the image
        o_image.save(o_image_path)

        modify_result[idx] = {"target_image": target_img_idx,
                              "similarity": similarity,
                              "m_embedding": modified_embedding,
                              "modified_image": modified_image,
                              "modified_img_original": modify_image_path
                              }
    return modify_result


def denormalize_image(tensor):
    """
    Reverses the normalization applied to the image tensor.

    Parameters:
    - tensor (torch.Tensor): The normalized image tensor to be denormalized.

    Returns:
    - torch.Tensor: The denormalized image tensor.
    """

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    return tensor * std + mean


def extract_features(image, model, processor, device):
    """
    Extract the image features using the CLIP model.
    """
    inputs = processor(image).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    # Normalize the features
    image_features = F.normalize(image_features, p=2, dim=-1)
    return image_features.squeeze(0)  # Remove batch dimension


class IMG:
    def __init__(self, path, idx, emb, is_modified, modify_target_idx):
        self.path = path
        self.idx = idx
        self.emb = emb
        self.is_modified = is_modified
        self.modify_target_idx = modify_target_idx


def plot_selection_weights(initial_weights, updated_weights, save_path=None):
    """
    Plot the weights before and after the attack.

    Parameters:
    - initial_weights (np.ndarray): Weights before the attack.
    - updated_weights (np.ndarray): Weights after the attack.
    - save_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    plt.figure(figsize=(12, 6))
    plt.hist(initial_weights, bins=50, alpha=0.5, label='Initial Weights')
    plt.hist(updated_weights, bins=50, alpha=0.5, label='Updated Weights')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Weights Before and After Attack')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def save_results(results, save_dir, save_name):
    """
    Save the results dictionary to a pickle file.

    Parameters:
    - results (dict): Results to save.
    - save_dir (str): Directory to save the results.
    - save_name (str): Filename for the saved results.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / f"{save_name}.pkl", "wb") as f:
        pickle.dump(results, f)


def load_results(save_dir, save_name):
    """
    Load the results dictionary from a pickle file.

    Parameters:
    - save_dir (str): Directory where the results are saved.
    - save_name (str): Filename of the saved results.

    Returns:
    - results (dict): Loaded results.
    """
    with open(Path(save_dir) / f"{save_name}.pkl", "rb") as f:
        results = pickle.load(f)
    return results


def load_image(image_path):
    """
    Load an image and return a PIL Image.
    """
    return Image.open(image_path)


def preprocess_image(image):
    """
    Preprocess the image for CLIP.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return preprocess(image)


def modify_image(
        image_path,
        target_vector,
        model,
        processor,
        device,
        num_steps=100,
        learning_rate=0.01,
        lambda_reg=0.1,
        epsilon=0.05,
        verbose=False
):
    """
    Modify an image to align its CLIP embedding with the target vector.

    Parameters:
    - image_path (str): Path to the image to be modified.
    - target_vector (np.array): Target embedding vector (1D array).
    - model (CLIPModel): Pre-trained CLIP model.
    - processor (CLIPProcessor): CLIP processor.
    - device (str): Device to run computations on ('cuda' or 'cpu').
    - num_steps (int): Number of optimization steps.
    - learning_rate (float): Learning rate for optimizer.
    - lambda_reg (float): Regularization strength.
    - epsilon (float): Maximum allowed perturbation per pixel.
    - verbose (bool): Whether to print progress messages.

    Returns:
    - modified_image (PIL.Image): The optimized image.
    - final_similarity (float): Cosine similarity with the target vector after modification.
    """
    # Load and preprocess the original image
    image = Image.open(image_path)
    image_tensor = processor(image).unsqueeze(0).to(device)  # No need for ['pixel_values']

    # If you want gradients with respect to the image tensor, set requires_grad
    image_tensor = image_tensor.clone().detach().requires_grad_(True)

    original_image_tensor = image_tensor.clone().detach()

    if verbose:
        print(f"Starting optimization for image: {image_path}")
        print("Initial image tensor shape:", image_tensor.shape)

    # Define optimizer
    optimizer = torch.optim.AdamW([image_tensor], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0)

    # Convert target_vector to torch tensor and normalize
    target_tensor = torch.tensor(target_vector, dtype=torch.float32).to(device)
    target_tensor = F.normalize(target_tensor, p=2, dim=0)

    # Set model to evaluation mode
    model.eval()

    # Retrieve normalization parameters from the processor
    # mean = torch.tensor(processor.feature_extractor.image_mean).view(3, 1, 1).to(device)
    # std = torch.tensor(processor.feature_extractor.image_std).view(3, 1, 1).to(device)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)
    # Initialize variables for early stopping
    previous_loss = float('inf')
    patience = 10  # Number of steps to wait for improvement
    patience_counter = 0

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass: get embedding
        embedding = model.encode_image(image_tensor)
        embedding = F.normalize(embedding, p=2, dim=-1)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(embedding, target_tensor, dim=-1).mean()

        # Compute perturbation norm (L-infinity)
        perturbation = image_tensor - original_image_tensor
        reg_loss = lambda_reg * torch.norm(perturbation, p=float('inf'))

        # Compute loss: maximize cosine similarity and minimize perturbation
        loss = -cosine_sim + reg_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([image_tensor], max_norm=1.0)

        # Check gradients
        if image_tensor.grad is not None:
            grad_norm = image_tensor.grad.norm().item()
            if verbose:
                print(
                    f"Step {step + 1}/{num_steps}, Grad Norm: {grad_norm:.4f}, Loss: {loss.item():.4f}, "
                    f"Cosine Similarity: {cosine_sim.item():.4f}")
        else:
            if verbose:
                print(f"Step {step + 1}/{num_steps}, No gradients computed.")
            grad_norm = 0.0

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Clamp the image tensor to maintain valid pixel range and limit perturbation
        with torch.no_grad():
            # Compute perturbation in normalized space
            perturbation = torch.clamp(image_tensor - original_image_tensor, -epsilon, epsilon)
            # Apply perturbation and clamp to ensure values are within normalized bounds
            new_image = torch.clamp(original_image_tensor + perturbation, (0 - mean) / std, (1 - mean) / std)
            image_tensor.copy_(new_image)

        # Early Stopping Check
        current_loss = loss.item()
        if verbose:
            print(f"Step {step + 1}/{num_steps}, Current Loss: {current_loss:.4f}")

        if abs(previous_loss - current_loss) < 1e-4:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break
        else:
            patience_counter = 0
        previous_loss = current_loss

        # Optional: Save intermediate images for visualization
        if (step + 1) % 50 == 0 and verbose:
            intermediate_image = image_tensor.detach().cpu().squeeze(0)
            intermediate_pil = transforms.ToPILImage()(intermediate_image)
            intermediate_pil.save(f"modified_step_{step + 1}.jpg")
            if verbose:
                print(f"Saved intermediate image at step {step + 1}")

    # Denormalize the image tensor
    def denormalize_image(tensor):
        return tensor * std.cpu() + mean.cpu()

    modified_image = denormalize_image(image_tensor.detach().cpu().squeeze(0))
    modified_image_pil = transforms.ToPILImage()(modified_image.clamp(0, 1))

    # Compute final similarity
    with torch.no_grad():
        modified_embedding = model.encode_image(image_tensor)
        modified_embedding = F.normalize(modified_embedding, p=2, dim=-1)
        final_similarity = F.cosine_similarity(modified_embedding, target_tensor, dim=-1).item()

    return modified_image_pil, modified_embedding, final_similarity
