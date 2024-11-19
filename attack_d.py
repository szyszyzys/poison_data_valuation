import argparse
import copy
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# CLIP model and processor
import daved.src.frank_wolfe as frank_wolfe  # Ensure this module contains the design_selection function
from attack.adv import Adv
# Import your custom modules or utilities
from attack.general_attack.my_utils import get_data, plot_results_data_selection
from daved.src import utils
from daved.src.main import plot_results


# def load_and_preprocess_data(data_dir, csv_path, max_char_length=2048, exclude_long_reviews=False):
#     """
#     Load and preprocess the dataset.
#
#     Parameters:
#     - data_dir (str): Directory where the CSV file is located.
#     - csv_path (str): Path to the CSV file containing the data.
#     - max_char_length (int): Maximum character length for reviews.
#     - exclude_long_reviews (bool): Whether to exclude reviews exceeding max_char_length.
#
#     Returns:
#     - data (dict): Dictionary containing split datasets.
#     - reviews (list): List of review texts.
#     - labels (list): List of corresponding labels.
#     """
#     df = pd.read_csv(Path(data_dir) / csv_path)
#     reviews = []
#     labels = []
#     for i, r in tqdm(df.iterrows(), total=df.shape[0], desc="Loading Data"):
#         x = f'Benefits: {r.benefitsReview}\nSide effects: {r.sideEffectsReview}\nComments: {r.commentsReview}'
#         if exclude_long_reviews and len(x) > max_char_length:
#             continue
#         reviews.append(x)
#         labels.append(r.rating)
#     print(f'Total Reviews Loaded: {len(reviews)}')
#
#     # Assuming `get_drug_data` is a utility function that handles data splitting and embedding extraction
#     data = utils.get_drug_data(
#         num_samples=len(reviews),
#         data_dir=data_dir,
#         csv_path=csv_path,
#         embedding_path=f"druglib/druglib_embeddings_clip.pt",  # Adjust as needed
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         model_name='clip',
#         max_char_length=max_char_length,
#     )
#     return data, reviews, labels


def identify_selected_unsampled(weights, num_select=10):
    """
    Identify which data points are selected and which are unselected based on weights.

    Parameters:
    - weights (np.ndarray): Weights assigned to each data point.
    - num_select (int): Number of data points to select.

    Returns:
    - selected_indices (set): Indices of selected data points.
    - unsampled_indices (list): Indices of unselected data points.
    """
    selected_indices = set(weights.argsort()[::-1][:num_select])
    unsampled_indices = list(set(range(len(weights))) - selected_indices)
    return list(selected_indices), unsampled_indices


def perform_attack(x_s, unsampled_indices, selected_indices, attack_strength=0.1):
    """
    Modify unselected data points to make them more likely to be selected.

    Parameters:
    - x_s (np.ndarray): Feature matrix for seller data.
    - unsampled_indices (list): Indices of unselected data points.
    - selected_indices (set): Indices of initially selected data points.
    - attack_strength (float): Strength of the attack (how much to modify features).

    Returns:
    - x_s_modified (np.ndarray): Modified feature matrix.
    - modified_indices (list): Indices of data points that were modified.
    """
    x_s_modified = x_s.copy()
    selected_features = x_s[list(selected_indices)]
    target_vector = selected_features.mean(axis=0)

    for idx in unsampled_indices:
        # Simple attack: move the feature vector closer to the target vector
        original_vector = x_s_modified[idx]
        perturbation = attack_strength * (target_vector - original_vector)
        x_s_modified[idx] = original_vector + perturbation

    return x_s_modified, unsampled_indices  # Assuming all unsampled are modified


def evaluate_attack_success_selection(initial_selected, updated_selected, modified_indices, total_selected=10):
    """
    Evaluate how many of the modified (attacked) data points were selected after the attack.

    Parameters:
    - initial_selected (set): Indices of initially selected data points.
    - updated_selected (set): Indices of selected data points after the attack.
    - modified_indices (list): Indices of data points that were modified.
    - total_selected (int): Total number of samples selected.
    Returns:
    - success_rate (float): Proportion of modified data points that were selected after the attack.
    - num_success (int): Number of modified data points selected after the attack.
    """
    updated_selected = set(updated_selected)
    modified_selected = updated_selected.intersection(modified_indices)
    num_success = len(modified_selected)
    success_rate = num_success / total_selected
    return success_rate, num_success


# def modify_image(
#         image_path,
#         target_vector,
#         model,
#         processor,
#         device,
#         num_steps=100,
#         learning_rate=0.01,
#         lambda_reg=0.1,
#         epsilon=0.05,
#         verbose=True
# ):
#     """
#     Modify an image to align its CLIP embedding with the target vector.
#
#     Parameters:
#     - image_path (str): Path to the image to be modified.
#     - target_vector (np.array): Target embedding vector (1D array).
#     - model (CLIPModel): Pre-trained CLIP model.
#     - processor (CLIPProcessor): CLIP processor.
#     - device (str): Device to run computations on ('cuda' or 'cpu').
#     - num_steps (int): Number of optimization steps.
#     - learning_rate (float): Learning rate for optimizer.
#     - lambda_reg (float): Regularization strength.
#     - epsilon (float): Maximum allowed perturbation per pixel.
#     - verbose (bool): Whether to print progress messages.
#
#     Returns:
#     - modified_image (PIL.Image): The optimized image.
#     - final_similarity (float): Cosine similarity with the target vector after modification.
#     """
#     # Load and preprocess the original image
#     # image = load_image(image_path)
#     # original_image_tensor = preprocess_image(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
#     # original_image_tensor = original_image_tensor.detach()  # Detach to prevent gradients flowing into original image
#     image = Image.open(image_path)
#     image_tensor = processor(image).unsqueeze(0).to(device).clone().detach().requires_grad_(
#         True)  # Shape: (1, 3, 224, 224)
#     original_image_tensor = image_tensor.clone().detach()
#     # Initialize the image tensor to be optimized
#     # image_tensor = original_image_tensor.clone().requires_grad_(True)
#
#     if verbose:
#         print(f"Starting optimization for image: {image_path}")
#         print("Initial image tensor shape:", image_tensor.shape)
#
#     # Define optimizer
#     optimizer = torch.optim.AdamW([image_tensor], lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0)
#
#     # Convert target_vector to torch tensor and normalize
#     target_tensor = torch.tensor(target_vector, dtype=torch.float32).to(device)
#     target_tensor = F.normalize(target_tensor, p=2, dim=0)
#
#     # Set model to evaluation mode
#     model.eval()
#
#     # Initialize variables for early stopping
#     previous_loss = float('inf')
#     patience = 10  # Number of steps to wait for improvement
#     patience_counter = 0
#
#     for step in range(num_steps):
#         optimizer.zero_grad()
#
#         # Forward pass: get embedding
#         embedding = model.encode_image(image_tensor)
#         # embedding = F.normalize(embedding, p=2, dim=-1)
#
#         # Compute cosine similarity and aggregate to scalar
#         cosine_sim = F.cosine_similarity(embedding, target_tensor, dim=-1).mean()
#
#         # Compute perturbation norm
#         perturbation = image_tensor - original_image_tensor
#         reg_loss = lambda_reg * torch.norm(perturbation)
#
#         # Compute loss: maximize cosine similarity and minimize perturbation
#         loss = -cosine_sim + reg_loss
#
#         # Backward pass
#         loss.backward()
#
#         # Check gradients
#         if image_tensor.grad is not None:
#             grad_norm = image_tensor.grad.norm().item()
#             if verbose:
#                 print(
#                     f"Step {step + 1}/{num_steps}, Grad Norm: {grad_norm:.4f}, Loss: {loss.item():.4f}, Cosine Similarity: {cosine_sim.item():.4f}")
#         else:
#             if verbose:
#                 print(f"Step {step + 1}/{num_steps}, No gradients computed.")
#             grad_norm = 0.0
#
#         # Optimizer step
#         optimizer.step()
#         scheduler.step()
#
#         # Clamp the image tensor to maintain valid pixel range and limit perturbation
#         with torch.no_grad():
#             # Calculate perturbation and clamp
#             perturbation = torch.clamp(image_tensor - original_image_tensor, -epsilon, epsilon)
#             # Apply perturbation
#             image_tensor.copy_(torch.clamp(original_image_tensor + perturbation, 0, 1))
#
#         # Early Stopping Check
#         current_loss = loss.item()
#         if verbose:
#             print(f"Step {step + 1}/{num_steps}, Current Loss: {current_loss:.4f}")
#
#         if abs(previous_loss - current_loss) < 1e-4:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 if verbose:
#                     print("Early stopping triggered.")
#                 break
#         else:
#             patience_counter = 0
#         previous_loss = current_loss
#
#         # Optional: Save intermediate images for visualization
#         if (step + 1) % 50 == 0 and verbose:
#             intermediate_image = image_tensor.detach().cpu().squeeze(0)
#             intermediate_pil = transforms.ToPILImage()(intermediate_image)
#             intermediate_pil.save(f"modified_step_{step + 1}.jpg")
#             if verbose:
#                 print(f"Saved intermediate image at step {step + 1}")
#
#     mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
#     std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
#
#     # Denormalize the image tensor
#     def denormalize_image(tensor):
#         return tensor * std + mean
#
#     # In your save step
#     modified_image = image_tensor.detach().cpu().squeeze(0)  # Remove batch dimension
#     modified_image = denormalize_image(modified_image)  # Apply denormalization
#     modified_image_pil = transforms.ToPILImage()(modified_image.clamp(0, 1))
#
#     # Compute final similarity
#     with torch.no_grad():
#         # normalized_modified_image = clip_normalize(preprocess_image(modified_image_pil).unsqueeze(0).to(device))
#         # modified_embedding = model.get_image_features(pixel_values=normalized_modified_image)
#         # modified_embedding = F.normalize(modified_embedding, p=2, dim=-1)
#
#         # normalized_modified_image = processor(modified_image_pil)
#         modified_embedding = model.encode_image(image_tensor)
#         # modified_embedding = F.normalize(modified_embedding, p=2, dim=-1)
#         final_similarity = F.cosine_similarity(modified_embedding, target_tensor, dim=-1).item()
#
#     return modified_image_pil, final_similarity


def convert_arrays_to_lists(obj):
    """Recursively convert NumPy arrays to lists in a nested structure."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_arrays_to_lists(element) for element in obj]
    else:
        return obj


def save_results_trained_model(args, results):
    for k, v in vars(args).items():
        if k not in results:
            if isinstance(v, Path):
                v = str(v)
            results[k] = v
        else:
            print(f"Found {k} in results. Skipping.")

    result_path = f"{args.result_dir}/{args.save_name}-results.json"
    results = convert_arrays_to_lists(results)
    with open(result_path, "w") as f:
        json.dump(results, f, default=float)
    print(f"Results saved to {result_path}".center(80, "="))


def save_json(path, results):
    for k, v in vars(args).items():
        if k not in results:
            if isinstance(v, Path):
                v = str(v)
            results[k] = v
        else:
            print(f"Found {k} in results. Skipping.")

    results = convert_arrays_to_lists(results)
    with open(path, "w") as f:
        json.dump(results, f, default=float)
    print(f"Results saved to {path}".center(80, "="))


def evaluate_model_raw_data(
        x_test,
        y_test,
        x_train,
        y_train,
        w,
        eval_range=range(1, 10),
        task='classification',
        use_sklearn=False,
        return_list=False,
        normalize=True,
        reduce_dim=False,
        dim_components=100,
        img_paths=None,
        test_img_indices=None,
        sell_img_indices=None,

):
    """
    Evaluate model performance using raw images instead of embeddings.

    Parameters:
    - x_test: NumPy array of raw test images, shape (n_test_samples, height, width, channels)
    - y_test: NumPy array of test labels, shape (n_test_samples,)
    - x_train: NumPy array of raw training images, shape (n_train_samples, height, width, channels)
    - y_train: NumPy array of training labels, shape (n_train_samples,)
    - w: NumPy array of weights for selecting training samples, shape (n_train_samples,)
    - eval_range: Iterable of integers representing different k values for top-k selection
    - task: String, either 'regression' or 'classification'
    - use_sklearn: Boolean flag to use scikit-learn's models
    - return_list: Boolean flag to return errors as a list instead of a dictionary
    - normalize: Boolean flag to apply normalization
    - reduce_dim: Boolean flag to apply dimensionality reduction
    - dim_components: Number of components for PCA if reduce_dim is True

    Returns:
    - metrics: Dictionary or list of evaluation metrics for each k in eval_range
    """
    x_test = []
    x_train = []

    # Load and preprocess test images
    for img_idx in tqdm(test_img_indices, desc="Load Test"):
        img = Image.open(img_paths[img_idx])
        img_preprocessed = preprocess(img).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
        x_test.append(img_preprocessed)

    # Load and preprocess training images
    for img_idx in tqdm(sell_img_indices, desc="Load Train"):
        img = Image.open(img_paths[img_idx])
        img_preprocessed = preprocess(img).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
        x_train.append(img_preprocessed)

    # Stack tensors and move to CPU for further processing
    x_train = torch.cat(x_train).cpu().numpy()  # Shape: (n_train, 3, 224, 224)
    x_test = torch.cat(x_test).cpu().numpy()  # Shape: (n_test, 3, 224, 224)

    # Flatten the raw images for model training
    x_train_flat = x_train.reshape(x_train.shape[0], -1)  # Shape: (n_train, height*width*channels)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)  # Shape: (n_test, height*width*channels)

    # Normalize the data if required
    if normalize:
        scaler = StandardScaler()
        x_train_flat = scaler.fit_transform(x_train_flat)
        x_test_flat = scaler.transform(x_test_flat)

    # Dimensionality Reduction (Optional)
    if reduce_dim:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim_components)
        x_train_flat = pca.fit_transform(x_train_flat)
        x_test_flat = pca.transform(x_test_flat)

    # Sort weights in descending order and get sorted indices
    sorted_w = np.argsort(w)[::-1]

    # Initialize metrics dictionary
    metrics = {}
    errors = {}
    for k in eval_range:
        # Select top-k training samples based on weights
        selected_indices = sorted_w[:k]
        x_k = x_train_flat[selected_indices]
        y_k = y_train[selected_indices]

        # Initialize and train the model
        if task == 'regression':
            if use_sklearn:
                model = LinearRegression(fit_intercept=True)
                model.fit(x_k, y_k)
                y_pred = model.predict(x_test_flat)
            else:
                # Using pseudo-inverse for linear regression
                beta_k = np.linalg.pinv(x_k) @ y_k
                y_pred = x_test_flat @ beta_k
        elif task == 'classification':
            if use_sklearn:
                model = LogisticRegression(max_iter=1000)
                model.fit(x_k, y_k)
                y_pred = model.predict(x_test_flat)
                y_prob = model.predict_proba(x_test_flat)[:, 1] if len(np.unique(y_k)) == 2 else None
            else:
                raise NotImplementedError("Custom classification model not implemented.")
        else:
            raise ValueError("Invalid task type. Choose 'regression' or 'classification'.")

        # Compute Evaluation Metrics
        if task == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics[k] = {'MSE': mse, 'MAE': mae, 'R2': r2}
        elif task == 'classification':
            # accuracy = accuracy_score(y_test, y_pred)
            # precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            # recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            # f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            #
            # if y_prob is not None:
            #     roc_auc = roc_auc_score(y_test, y_prob)
            #     metrics[k] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1,
            #                   'ROC-AUC': roc_auc}
            # else:
            #     metrics[k] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
            errors[k] = mean_squared_error(y_test, y_pred)

    return list(errors.values()) if return_list else errors

    # if return_list:
    #     # Return metrics as a list ordered by eval_range
    #     return [metrics[k] for k in eval_range]
    # else:
    #     # Return metrics as a dictionary
    #     return metrics


def sampling_run_one_buyer(x_b, y_b, x_s, y_s, eval_range, costs=None, args=None, figure_path="./figure",
                           img_paths=None, test_img_indices=None, sell_img_indices=None):
    # Dictionaries to store errors, runtimes, and weights for each method and test point
    errors = defaultdict(list)
    runtimes = defaultdict(list)
    weights = defaultdict(list)
    test_point_info = []  # To store details for each test point evaluation

    # Loop over each test point in buyer's data, in batches
    for i, j in tqdm(enumerate(range(0, x_b.shape[0], args.batch_size))):
        # Get batch of test points
        x_test = x_b[j: j + args.batch_size]
        y_test = y_b[j: j + args.batch_size]

        # Prepare keyword arguments for the error function
        err_kwargs = dict(
            x_test=x_test,
            y_test=y_test,
            x_train=x_s,
            y_train=y_s,
            eval_range=eval_range,
            img_paths=img_paths,
            test_img_indices=test_img_indices,
            sell_img_indices=sell_img_indices,
            task='classification',
        )

        if True:
            error_func = evaluate_model_raw_data
            err_kwargs["return_list"] = True
        elif costs is not None:
            error_func = utils.get_error_under_budget
            err_kwargs["costs"] = costs
        else:
            error_func = utils.get_error_fixed
            err_kwargs["return_list"] = True

        # Perform single-step optimization (DAVED single step)
        os_start = time.perf_counter()
        w_os = frank_wolfe.one_step(x_s, x_test)
        os_end = time.perf_counter()

        # Store runtime and weights for single-step
        runtimes["DAVED (single step)"].append(os_end - os_start)
        weights["DAVED (single step)"].append(w_os)

        # Record error for single-step
        errors["DAVED (single step)"].append(error_func(w=w_os, **err_kwargs))

        # Perform multi-step optimization (DAVED multi-step)
        fw_start = time.perf_counter()
        res_fw = frank_wolfe.design_selection(
            x_s,
            y_s,
            x_test,
            y_test,
            num_select=10,
            num_iters=args.num_iters,
            alpha=None,
            recompute_interval=0,
            line_search=True,
            costs=costs,
            reg_lambda=args.reg_lambda,
        )
        fw_end = time.perf_counter()

        # Store runtime, weights, and errors for multi-step
        w_fw = res_fw["weights"]
        runtimes["DAVED (multi-step)"].append(fw_end - fw_start)
        weights["DAVED (multi-step)"].append(w_fw)
        errors["DAVED (multi-step)"].append(error_func(w=w_fw, **err_kwargs))

        # Store information about the test point, the indices used, and their results
        test_point_info.append({
            "query_number": i,
            "test_point_start_index": j,
            "test_x": x_test,
            "test_y": y_test,
            "single_step_weights": w_os,
            "single_step_error": errors["DAVED (single step)"][-1],
            "multi_step_weights": w_fw,
            "multi_step_error": errors["DAVED (multi-step)"][-1],
            "runtime_single_step": runtimes["DAVED (single step)"][-1],
            "runtime_multi_step": runtimes["DAVED (multi-step)"][-1],
            "eval_range": eval_range
        })

        # Save intermediate results periodically
        # if i % 25 == 0:
        #     attack_model_result = dict(errors=errors, eval_range=eval_range, runtimes=runtimes)
        #     save_results_trained_model(args=args, results=attack_model_result)
        #     plot_results(f"{figure_path}_inter_r_{i}_res.png", results=attack_model_result, args=args)
        #     print(f"Checkpoint: Saved results at round {i}".center(40, "="))

    # Final save of all results if not skipped
    if not args.skip_save:
        attack_model_result = dict(errors=errors, eval_range=eval_range, runtimes=runtimes)
        with open(f"{args.result_dir}/{args.save_name}-weights.pkl", "wb") as f:
            pickle.dump(weights, f)
        save_results_trained_model(args=args, results=attack_model_result)
        plot_results(f"{figure_path}_error_plotting.png", results=attack_model_result, args=args)

    return attack_model_result, test_point_info


def print_evaluation_results(evaluation_results):
    """
    Print the evaluation results of the adversarial poisoning effectiveness in a readable format.

    Parameters:
    - evaluation_results (dict): Dictionary containing metrics on poisoning effectiveness.
    """

    print("==== Adversarial Poisoning Effectiveness Evaluation ====")
    print(f"Number of adversarial samples selected before poisoning: {evaluation_results['num_adv_selected_before']}")
    print(f"Number of adversarial samples selected after poisoning: {evaluation_results['num_adv_selected_after']}")
    print(f"Increase in selected adversarial samples: {evaluation_results['increase_in_selected_adv']}")

    print("\n-- Weight Analysis for Adversarial Samples --")
    print(f"Mean weight of adversarial samples before poisoning: {evaluation_results['mean_adv_weight_before']:.4f}")
    print(f"Mean weight of adversarial samples after poisoning: {evaluation_results['mean_adv_weight_after']:.4f}")
    print(f"Weight increase for adversarial samples: {evaluation_results['weight_increase_adv']:.4f}")

    print("\n-- Weight Analysis for Non-Adversarial Samples --")
    print(
        f"Mean weight of non-adversarial samples before poisoning: {evaluation_results['mean_non_adv_weight_before']:.4f}")
    print(
        f"Mean weight of non-adversarial samples after poisoning: {evaluation_results['mean_non_adv_weight_after']:.4f}")
    print(f"Weight increase for non-adversarial samples: {evaluation_results['weight_increase_non_adv']:.4f}")

    print("========================================================")


def calculate_average_metrics(results):
    """
    Calculate average metrics from a list of evaluation results.

    Parameters:
    - results (list of dict): List where each element is a dictionary containing the metrics
                              for a single evaluation, structured as shown in your example.

    Returns:
    - avg_metrics (dict): Dictionary with the average of each metric.
    """
    # Initialize accumulators
    total_selected_num = 0
    total_num_adv_selected_before = 0
    total_num_adv_selected_after = 0
    total_selection_rate_before = 0
    total_selection_rate_after = 0

    total_mean_adv_weight_before = 0
    total_mean_adv_weight_after = 0
    total_weight_increase_adv = 0
    total_mean_non_adv_weight_before = 0
    total_mean_non_adv_weight_after = 0
    total_weight_increase_non_adv = 0

    num_results = len(results)
    selected_num = []
    # Accumulate values
    for result in results:
        total_mean_adv_weight_before += result["mean_adv_weight_before"]
        total_mean_adv_weight_after += result["mean_adv_weight_after"]
        total_weight_increase_adv += result["weight_increase_adv"]
        total_mean_non_adv_weight_before += result["mean_non_adv_weight_before"]
        total_mean_non_adv_weight_after += result["mean_non_adv_weight_after"]
        total_weight_increase_non_adv += result["weight_increase_non_adv"]

    # Calculate averages
    avg_metrics = {
        "avg_selected_num": total_selected_num / num_results,
        "avg_num_adv_selected_before": total_num_adv_selected_before / num_results,
        "avg_num_adv_selected_after": total_num_adv_selected_after / num_results,
        "avg_selection_rate_before": total_selection_rate_before / num_results,
        "avg_selection_rate_after": total_selection_rate_after / num_results,

        "avg_mean_adv_weight_before": total_mean_adv_weight_before / num_results,
        "avg_mean_adv_weight_after": total_mean_adv_weight_after / num_results,
        "avg_weight_increase_adv": total_weight_increase_adv / num_results,
        "avg_mean_non_adv_weight_before": total_mean_non_adv_weight_before / num_results,
        "avg_mean_non_adv_weight_after": total_mean_non_adv_weight_after / num_results,
        "avg_weight_increase_non_adv": total_weight_increase_non_adv / num_results,
    }

    return avg_metrics


def evaluate_poisoning_effectiveness_ranged(
        initial_weights,
        updated_weights,
        adversary_indices,
        eval_range):
    """
    Evaluate the effectiveness of adversarial poisoning by comparing initial and updated weights and selections.

    Parameters:
    - initial_weights (np.ndarray): Array of weights before poisoning.
    - updated_weights (np.ndarray): Array of weights after poisoning.
    - adversary_indices (np.ndarray): Indices of adversarial samples.

    Returns:
    - dict: Evaluation metrics including counts and weight statistics.
    """
    # the selection rate of adversary under different selection number.
    selection_info = {}
    for selected_num in eval_range:
        o_selected, _ = identify_selected_unsampled(initial_weights, selected_num)
        u_selected, _ = identify_selected_unsampled(updated_weights, selected_num)

        # Step 1: Find adversarial samples in selected set before and after poisoning
        initial_selected_adv = np.intersect1d(o_selected, adversary_indices)
        updated_selected_adv = np.intersect1d(u_selected, adversary_indices)

        # Step 2: Calculate number of adversarial samples selected before and after poisoning
        n_before = len(initial_selected_adv)
        n_after = len(updated_selected_adv)
        selection_info[selected_num] = {
            "selected_num": selected_num,
            "num_adv_selected_before": n_before,
            "num_adv_selected_after": n_after,
            "selection_rate_before": n_before / selected_num,
            "selection_rate_after": n_after / selected_num,
        }

    # Step 3: Calculate mean weight for adversarial samples before and after poisoning
    adv_weights_before = initial_weights[adversary_indices]
    adv_weights_after = updated_weights[adversary_indices]
    mean_adv_weight_before = np.mean(adv_weights_before)
    mean_adv_weight_after = np.mean(adv_weights_after)
    weight_increase_adv = mean_adv_weight_after - mean_adv_weight_before

    # For comparison, calculate mean weight increase for non-adversarial samples
    non_adversary_indices = np.setdiff1d(np.arange(len(initial_weights)), adversary_indices)
    non_adv_weights_before = initial_weights[non_adversary_indices]
    non_adv_weights_after = updated_weights[non_adversary_indices]
    mean_non_adv_weight_before = np.mean(non_adv_weights_before)
    mean_non_adv_weight_after = np.mean(non_adv_weights_after)
    weight_increase_non_adv = mean_non_adv_weight_after - mean_non_adv_weight_before

    evaluation_results = {
        "selection_eval_range": eval_range,
        "selection_info": selection_info,
        "mean_adv_weight_before": mean_adv_weight_before,
        "mean_adv_weight_after": mean_adv_weight_after,
        "weight_increase_adv": weight_increase_adv,
        "mean_non_adv_weight_before": mean_non_adv_weight_before,
        "mean_non_adv_weight_after": mean_non_adv_weight_after,
        "weight_increase_non_adv": weight_increase_non_adv,
    }

    return evaluation_results


def evaluate_poisoning_effectiveness(
        initial_weights,
        updated_weights,
        selected_indices_initial,
        selected_indices_updated,
        adversary_indices
):
    """
    Evaluate the effectiveness of adversarial poisoning by comparing initial and updated weights and selections.

    Parameters:
    - initial_weights (np.ndarray): Array of weights before poisoning.
    - updated_weights (np.ndarray): Array of weights after poisoning.
    - selected_indices_initial (np.ndarray): Indices of samples selected before poisoning.
    - selected_indices_updated (np.ndarray): Indices of samples selected after poisoning.
    - adversary_indices (np.ndarray): Indices of adversarial samples.

    Returns:
    - dict: Evaluation metrics including counts and weight statistics.
    """

    # Step 1: Find adversarial samples in selected set before and after poisoning
    initial_selected_adv = np.intersect1d(selected_indices_initial, adversary_indices)
    updated_selected_adv = np.intersect1d(selected_indices_updated, adversary_indices)

    # Step 2: Calculate number of adversarial samples selected before and after poisoning
    num_adv_selected_before = len(initial_selected_adv)
    num_adv_selected_after = len(updated_selected_adv)

    # Step 3: Calculate mean weight for adversarial samples before and after poisoning
    adv_weights_before = initial_weights[adversary_indices]
    adv_weights_after = updated_weights[adversary_indices]

    mean_adv_weight_before = np.mean(adv_weights_before)
    mean_adv_weight_after = np.mean(adv_weights_after)
    weight_increase_adv = mean_adv_weight_after - mean_adv_weight_before

    # For comparison, calculate mean weight increase for non-adversarial samples
    non_adversary_indices = np.setdiff1d(np.arange(len(initial_weights)), adversary_indices)
    non_adv_weights_before = initial_weights[non_adversary_indices]
    non_adv_weights_after = updated_weights[non_adversary_indices]

    mean_non_adv_weight_before = np.mean(non_adv_weights_before)
    mean_non_adv_weight_after = np.mean(non_adv_weights_after)
    weight_increase_non_adv = mean_non_adv_weight_after - mean_non_adv_weight_before

    # Compile the evaluation results into a dictionary
    evaluation_results = {
        "num_adv_selected_before": num_adv_selected_before,
        "num_adv_selected_after": num_adv_selected_after,
        "increase_in_selected_adv": num_adv_selected_after - num_adv_selected_before,
        "mean_adv_weight_before": mean_adv_weight_before,
        "mean_adv_weight_after": mean_adv_weight_after,
        "weight_increase_adv": weight_increase_adv,
        "mean_non_adv_weight_before": mean_non_adv_weight_before,
        "mean_non_adv_weight_after": mean_non_adv_weight_after,
        "weight_increase_non_adv": weight_increase_non_adv,
    }
    print_evaluation_results(evaluation_results)
    return evaluation_results


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

    result_dir = f'{result_dir}/attack_{attack_type}_num_buyer_{num_buyer}_num_seller_{num_seller}_advr_{adversary_ratio}_prate_{args.poison_rate}_querys_{args.batch_size}/'
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
    adv = Adv(x_s, y_s, costs, adversary_indices, emb_model=emb_model_name, device=device, img_paths=img_paths)

    # Evaluate the peformance
    eval_range = list(range(1, 30, 1)) + list(
        range(30, args.max_eval_range, args.eval_step)
    )

    benign_node_sampling_result_dict = {}
    benign_training_results, benign_selection_info = sampling_run_one_buyer(x_b, y_b, x_s, y_s, eval_range, costs=costs,
                                                                            args=args,
                                                                            figure_path=f"{figure_path}benign"
                                                                            , img_paths=img_paths,
                                                                            test_img_indices=index_b,
                                                                            sell_img_indices=index_s
                                                                            )

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
        img_paths_clone = copy.deepcopy(img_paths)
        for img_idx, info in manipulated_img_dict.items():
            modified_embedding = info["m_embedding"]
            x_s_clone[img_idx] = modified_embedding
            img_paths_clone[img_idx] = info["modified_img_path"]

        # use the sample query to perform the attack.
        model_training_result, data_sampling_result = sampling_run_one_buyer(
            x_test, y_test, x_s_clone, y_s, eval_range, costs=costs, args=args, figure_path=attack_result_path,
            img_paths=img_paths_clone, test_img_indices=index_b, sell_img_indices=index_s

        )

        attack_result_dict[query_n] = {
            "model_training_result": model_training_result,
            "data_sampling_result": data_sampling_result[0]
        }
    evaluation_results_list = []
    torch.save(attack_result_dict, f"{result_dir}/selection_result.pt")

    # benign_training_results = dict(errors=errors, eval_range=eval_range, runtimes=runtimes)
    attack_training_error = []
    selection_info = defaultdict(list)
    # Step 6: attack evaluation, identify Updated Selected Data Points
    for query_n in attack_result_dict.keys():
        malicious_info = attack_result_dict[query_n]
        malicious_data_sampling_result = malicious_info["data_sampling_result"]
        malicious_model_training_result = malicious_info["model_training_result"]

        # find the benign info
        benign_data_sampling_result = benign_node_sampling_result_dict[query_n]

        m_s_weight = malicious_data_sampling_result["single_step_weights"]
        m_m_weight = malicious_data_sampling_result["multi_step_weights"]

        b_s_weight = benign_data_sampling_result["single_step_weights"]
        b_m_weight = benign_data_sampling_result["multi_step_weights"]

        evaluation_results_single = evaluate_poisoning_effectiveness_ranged(
            initial_weights=b_s_weight,
            updated_weights=m_s_weight,
            adversary_indices=adversary_indices,
            eval_range=eval_range)
        selection_info_single = evaluation_results_single["selection_info"]

        evaluation_results_multi = evaluate_poisoning_effectiveness_ranged(
            initial_weights=b_m_weight,
            updated_weights=m_m_weight,
            adversary_indices=adversary_indices,
            eval_range=eval_range)
        selection_info_multi = evaluation_results_multi["selection_info"]

        selection_info_bs = []
        selection_info_ms = []
        selection_info_bm = []
        selection_info_mm = []

        for idx in eval_range:
            selection_info_bs.append(selection_info_single[idx]["num_adv_selected_before"])
            selection_info_ms.append(selection_info_single[idx]["num_adv_selected_after"])
            selection_info_bm.append(selection_info_multi[idx]["num_adv_selected_before"])
            selection_info_mm.append(selection_info_multi[idx]["num_adv_selected_after"])

        selection_info["benign(single)"].append(selection_info_bs)
        selection_info["malicious(single)"].append(selection_info_ms)
        selection_info["benign(multi)"].append(selection_info_bm)
        selection_info["malicious(multi)"].append(selection_info_mm)

    plot_results_data_selection("image_selection_rate", f"{figure_path}/selection_eval.png", selection_info,
                                eval_range)

    # start combine the result for trained model performance

    m_errors = defaultdict(list)
    m_runtimes = defaultdict(list)
    for query_n in attack_result_dict.keys():
        malicious_info = attack_result_dict[query_n]
        malicious_model_training_result = malicious_info["model_training_result"]
        m_runtimes["ADV DAVED (single step)"].append(
            malicious_model_training_result["runtimes"]["DAVED (single step)"][0])
        m_errors["ADV DAVED (single step)"].append(malicious_model_training_result["errors"]["DAVED (single step)"][0])
        m_runtimes["ADV DAVED (multi-step)"].append(
            malicious_model_training_result["runtimes"]["DAVED (single step)"][0])
        m_errors["ADV DAVED (multi-step)"].append(malicious_model_training_result["errors"]["DAVED (single step)"][0])

    for k, v in m_errors.items():
        benign_training_results["errors"][k] = v
    for k, v in m_runtimes.items():
        benign_training_results["runtimes"][k] = v

    # selection_attack_result = os.path.join(
    #     result_dir,
    #     f'modified_images_{attack_type}_multi_step',
    #     f'step_{args.attack_steps}_lr_{args.attack_lr}_reg_{args.attack_reg}_advr_{adversary_ratio}',
    #     f'target_query_batch_{test_point_index}'
    # )

    # save_json(f"{selection_attack_result}/all_result_selection.json", evaluation_results_list)
    # save_json(f"{selection_attack_result}/avg_result_selection.json", averaged_data_selection_results)
    # Step 5: Perform Attack on Unselected Data Points

    # Step 8: Evaluate Attack Success on trained model
    args.save_name = "attack_result"
    plot_results(f"{figure_path}/model_training_result.png", results=benign_training_results, args=args)
    save_results_trained_model(args, benign_training_results)

    # comprehensive_evaluation(
    #     X_sell_original, y_sell_original,
    #     X_sell_modified, y_sell_modified,
    #     X_buy, y_buy,
    #     data_indices_original, data_indices_modified,
    #     cv_folds=5
    # )

    # mse_before = evaluate_attack_trained_model(
    #     x_s, y_s, x_b, y_b, selected_indices_initial, inverse_covariance=None)
    #
    # mse_after = evaluate_attack_trained_model(
    #     x_s, y_s, x_b, y_b, selected_indices_updated, inverse_covariance=None)

    # Step 8: Plot Weight Distributions
    # plot_selection_weights(
    #     initial_weights=initial_results['weights'],
    #     updated_weights=updated_results['weights'],
    #     save_path=f'{result_dir}/',  # Set a path to save the plot if desired
    # )

    # Step 9: Save Results (Optional)
    # if save_results_flag:
    #     results = {
    #         'initial_selected_indices': selected_indices_initial,
    #         'unsampled_indices_initial': unsampled_indices_initial,
    #         'x_s_modified': x_s,
    #         'updated_selected_indices': selected_indices_updated,
    #         'modified_indices': modified_indices,
    #         'attack_success_rate': success_rate,
    #         'num_successfully_selected': num_success,
    #         'trained_model_mse_before': mse_before,
    #         'trained_model_mse_after': mse_after,
    #     }
    #     save_results(
    #         results=results,
    #         save_dir=result_dir,
    #         save_name=save_name,
    #     )
    #     print(f"Results saved to {result_dir}/{save_name}.pkl")
    #
    # return {
    #     'initial_selected_indices': selected_indices_initial,
    #     'unsampled_indices_initial': unsampled_indices_initial,
    #     'updated_selected_indices': selected_indices_updated,
    #     'modified_indices': modified_indices,
    #     'attack_success_rate': success_rate,
    #     'num_successfully_selected': num_success,
    # }


# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def attack_target_sampling(data, labels, strategy="random", num_samples=100, **kwargs):
    """
    Samples data points for fine-tuning using different selection strategies.

    Parameters:
    - data (numpy.ndarray): Array of data points, shape (n_samples, n_features).
    - labels (numpy.ndarray): Array of labels corresponding to data points, shape (n_samples,).
    - strategy (str): Sampling strategy to use; options are 'random', 'uncertainty', 'stratified', 'similarity'.
    - num_samples (int): Number of samples to select.
    - **kwargs: Additional parameters for specific strategies.
        - uncertainty_scores (numpy.ndarray): Array of uncertainty scores, required for 'uncertainty' strategy.
        - similarity_target (numpy.ndarray): Target vector for similarity-based sampling, required for 'similarity' strategy.
        - similarity_metric (str): Similarity metric to use for 'similarity' strategy; default is 'cosine'.

    Returns:
    - selected_indices (numpy.ndarray): Indices of selected data points.
    """
    n_samples = data.shape[0]

    # Validate num_samples
    if num_samples > n_samples:
        raise ValueError("num_samples cannot exceed the total number of data points.")

    # Random Sampling
    if strategy == "random":
        selected_indices = np.random.choice(n_samples, num_samples, replace=False)

    # Uncertainty Sampling
    elif strategy == "uncertainty":
        uncertainty_scores = kwargs.get("uncertainty_scores")
        if uncertainty_scores is None or len(uncertainty_scores) != n_samples:
            raise ValueError(
                "For 'uncertainty' strategy, 'uncertainty_scores' must be provided with length equal to data points.")

        # Sort by highest uncertainty and select top indices
        selected_indices = np.argsort(uncertainty_scores)[-num_samples:]

    # Stratified Sampling
    elif strategy == "stratified":
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=num_samples, random_state=42)
        train_idx, selected_indices = next(stratified_split.split(data, labels))

    # Similarity-Based Sampling
    elif strategy == "similarity":
        similarity_target = kwargs.get("similarity_target")
        similarity_metric = kwargs.get("similarity_metric", "cosine")
        if similarity_target is None:
            raise ValueError("For 'similarity' strategy, 'similarity_target' must be provided.")

        # Calculate similarity scores
        distances = pairwise_distances(data, similarity_target.reshape(1, -1), metric=similarity_metric).flatten()
        selected_indices = np.argsort(distances)[:num_samples]  # Select closest samples to the target vector

    else:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Choose from 'random', 'uncertainty', 'stratified', or 'similarity'.")

    return selected_indices


def embed_images(img_paths, model, preprocess, device):
    """
    Embed images using CLIP's encode_image function.

    Parameters:
    - img_paths (list of str): Paths to the images to be embedded.
    - model (CLIPModel): Pre-trained CLIP model.
    - preprocess (callable): CLIP's preprocess function.
    - device (str): Device to run computations on ('cpu' or 'cuda').

    Returns:
    - embeddings (torch.Tensor): Concatenated embeddings of all images.
    """
    embeddings = []
    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Embedding Images"):
            img = Image.open(img_path)
            img_preprocessed = preprocess(img).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
            embedding = model.encode_image(img_preprocessed)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize embedding
            embeddings.append(embedding.cpu())
    return torch.cat(embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Data subset selection with experimental design informed loss using Frank-Wolfe optimization",
    )
    parser.add_argument("--random_seed", default=12345, help="random seed")
    parser.add_argument("--figure_dir", default="./figures")
    parser.add_argument("--result_dir", default="./results")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument(
        "-d",
        "--dataset",
        default="fitzpatrick",
        choices=[
            "gaussian",
            "mimic",
            "bone",
            "fitzpatrick",
            "drug",
        ],
        type=str,
        help="dataset to run experiment on",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="number of test points to optimize at once",
    )
    parser.add_argument(
        "--num_buyers",
        default=30,
        type=int,
        help="number of test buyer points used in experimental design",
    )
    parser.add_argument(
        "--num_seller",
        default=1000,
        type=int,
        help="number of seller points used in experimental design",
    )
    parser.add_argument(
        "--num_val",
        default=100,
        type=int,
        help="number of validation points for baselines",
    )
    parser.add_argument(
        "--num_dim",
        default=100,
        type=int,
        help="dimensionality of the data samples",
    )
    parser.add_argument(
        "--num_select",
        default=500,
        type=int,
        help="dimensionality of the data samples",
    )

    parser.add_argument(
        "--num_iters",
        default=500,
        type=int,
        help="number of iterations to optimize experimental design",
    )
    parser.add_argument(
        "--max_eval_range",
        default=150,
        type=int,
        help="max number training points to select for evaluation",
    )
    parser.add_argument(
        "--eval_step",
        default=5,
        type=int,
        help="evaluation interval",
    )

    parser.add_argument(
        "--attack_reg",
        default=0,
        type=float,
        help="attack reg",
    )
    parser.add_argument(
        "--attack_lr",
        default=0.1,
        type=float,
        help="attack lr",
    )

    parser.add_argument(
        "--baselines",
        nargs="*",
        default=[
            # "AME",
            # "BetaShapley",
            # "DataBanzhaf",
            "DataOob",
            # "DataShapley",
            "DVRL",
            # "InfluenceSubsample",
            "KNNShapley",
            "LavaEvaluator",
            # "LeaveOneOut",
            "RandomEvaluator",
            # "RobustVolumeShapley",
        ],
        type=str,
        help="Compare to other data valution baselines in opendataval",
    )
    parser.add_argument("--debug", action="store_true", help="Turn on debugging mode")
    parser.add_argument("--use_cost", action="store_true", help="If use cost")

    parser.add_argument(
        "--skip_save", action="store_true", help="Don't save weights or data"
    )

    parser.add_argument(
        "--cost_range",
        nargs="*",
        default=None,
        help="""
    Choose range of costs to sample uniformly.
    E.g. costs=[1, 2, 3, 9] will randomly set each seller data point
    to one of these costs and apply the cost_func during optimization.
    If set to None, no cost is applied during optimization.
    Default is None.
    """,
    )
    parser.add_argument(
        "--cost_func",
        default="linear",
        choices=["linear", "squared", "square_root"],
        type=str,
        help="Choose cost function to apply.",
    )

    parser.add_argument(
        "--save_name",
        default="robustness_test",
        type=str,
        help="save_name.",
    )
    parser.add_argument(
        "--cost_gen_mode",
        default="constant",
        choices=["random_uniform", "random_choice", "constant"],
        type=str,
        help="Choose cost function to apply.",
    )

    parser.add_argument(
        "--noise_level",
        default=0,
        type=float,
        help="level of noise to add for cost",
    )
    parser.add_argument(
        "--reg_lambda",
        default=0.0,
        type=float,
        help="Regularization initial inverse information matrix to be identity. Should be between 0 and 1.",
    )

    parser.add_argument(
        "--poison_rate",
        default=0.2,
        type=float,
        help="rate of images to do poison for a buyer",
    )

    parser.add_argument(
        "--attack_steps",
        default=200,
        type=int,
        help="attack step",
    )
    args = parser.parse_args()
    save_name = f'attack_evaluation_{args.dataset}_b_{args.num_buyers}_s_{args.num_seller}'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_model, preprocess = clip.load("ViT-B/32", device=device)
    emb_model.eval()  # Set model to evaluation mode

    # Define experiment parameters
    experiment_params = {
        'dataset': args.dataset,
        'data_dir': './data',
        'csv_path': "./fitzpatrick17k/fitzpatrick-mod.csv",
        'img_path' './fitzpatrick17k/images'
        'num_buyer': args.num_buyers,
        'num_seller': args.num_seller,
        'num_val': args.num_val,
        'max_eval_range': args.max_eval_range,
        'eval_step': args.eval_step,
        'num_iters': args.num_iters,
        'reg_lambda': 0.1,
        'attack_strength': 0.1,
        'save_results_flag': True,
        'result_dir': 'results',
        'save_name': save_name,
        "num_select": args.num_select,
        "adversary_ratio": 0.25,
        "emb_model": emb_model,
        "img_preprocess": preprocess,
        "emb_model_name": "clip"
    }

    # Run the attack evaluation experiment
    results = evaluate_poisoning_attack(args, **experiment_params)

    # Print final results
    print("\nFinal Attack Evaluation Metrics:")
    # print(f"Attack Success Rate: {results['attack_success_rate'] * 100:.2f}%")
    # print(f"Number of Successfully Selected Modified Data Points: {results['num_successfully_selected']}")
