import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# CLIP model and processor
import daved.src.frank_wolfe as frank_wolfe  # Ensure this module contains the design_selection function
# Import your custom modules or utilities
import daved.src.utils as utils  # Ensure this module contains necessary utility functions


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


def initial_selection(x_s, y_s, x_b, y_b, num_iters, reg_lambda, costs):
    """
    Perform the initial data selection using the design_selection function.

    Parameters:
    - x_s (np.ndarray): Feature matrix for seller data.
    - y_s (np.ndarray): Labels for seller data.
    - x_b (np.ndarray): Feature matrix for buyer data.
    - y_b (np.ndarray): Labels for buyer data.
    - num_iters (int): Number of iterations for the selection algorithm.
    - reg_lambda (float): Regularization parameter.

    Returns:
    - initial_results (dict): Results from the initial selection.
    """
    initial_results = frank_wolfe.design_selection(
        x_s,
        y_s,
        x_b,
        y_b,
        num_select=10,
        num_iters=num_iters,
        alpha=None,
        recompute_interval=0,
        line_search=True,
        costs=costs,
        reg_lambda=reg_lambda,
    )
    #     num_select=10,
    #     num_iters=num_iters,
    #     alpha=None,
    #     recompute_interval=0,
    #     line_search=True,
    #     costs=None,
    #     reg_lambda=reg_lambda,
    # )
    return initial_results


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


def re_run_selection(x_s_modified, y_s, x_b, y_b, num_iters, reg_lambda):
    """
    Perform data selection again on the modified dataset.

    Parameters:
    - x_s_modified (np.ndarray): Modified feature matrix for seller data.
    - y_s (np.ndarray): Labels for seller data.
    - x_b (np.ndarray): Feature matrix for buyer data.
    - y_b (np.ndarray): Labels for buyer data.
    - num_iters (int): Number of iterations for the selection algorithm.
    - reg_lambda (float): Regularization parameter.

    Returns:
    - updated_results (dict): Results from the updated selection.
    """
    updated_results = frank_wolfe.design_selection(
        x_s_modified,
        y_s,
        x_b,
        y_b,
        num_select=10,
        num_iters=num_iters,
        alpha=None,
        recompute_interval=0,
        line_search=True,
        costs=None,
        reg_lambda=reg_lambda,
    )
    return updated_results


def evaluate_attack_success(initial_selected, updated_selected, modified_indices):
    """
    Evaluate how many of the modified (attacked) data points were selected after the attack.

    Parameters:
    - initial_selected (set): Indices of initially selected data points.
    - updated_selected (set): Indices of selected data points after the attack.
    - modified_indices (list): Indices of data points that were modified.

    Returns:
    - success_rate (float): Proportion of modified data points that were selected after the attack.
    - num_success (int): Number of modified data points selected after the attack.
    """
    modified_selected = updated_selected.intersection(modified_indices)
    num_success = len(modified_selected)
    success_rate = num_success / len(modified_indices) if len(modified_indices) > 0 else 0.0
    return success_rate, num_success


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
    return Image.open(image_path).convert("RGB")


def preprocess_image(image):
    """
    Preprocess the image for CLIP.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return preprocess(image)


# def modify_image(
#         image_path,
#         target_vector,
#         model,
#         processor,
#         device,
#         num_steps=100,
#         learning_rate=0.01,
#         lambda_reg=0.1,
#         epsilon=0.05
# ):
#     """
#     Modify an image to align its CLIP embedding with the target vector.
#
#     Parameters:
#     - image_path (str): Path to the image to be modified.
#     - target_vector (np.array): Target embedding vector.
#     - model (CLIPModel): Pre-trained CLIP model.
#     - processor (CLIPProcessor): CLIP processor.
#     - device (str): Device to run computations on.
#     - num_steps (int): Number of optimization steps.
#     - learning_rate (float): Learning rate for optimizer.
#     - lambda_reg (float): Regularization strength.
#     - epsilon (float): Maximum allowed perturbation per pixel.
#
#     Returns:
#     - modified_image (PIL.Image): The optimized image.
#     - final_similarity (float): Cosine similarity with the target vector after modification.
#     """
#     # Load and preprocess the image
#     image = load_image(image_path)
#     image_tensor = preprocess_image(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
#     image_tensor.requires_grad = True
#     print("Current image tensor shape:", image_tensor.shape)
#
#     # Define optimizer
#     optimizer = torch.optim.Adam([image_tensor], lr=learning_rate)
#
#     # Convert target_vector to torch tensor and normalize
#     target_tensor = torch.tensor(target_vector, dtype=torch.float32).to(device)
#     target_tensor = F.normalize(target_tensor, p=2, dim=0)
#
#     for step in range(num_steps):
#         optimizer.zero_grad()
#
#         # Forward pass: get embedding
#         # Ensure image_tensor is detached and moved to CPU before processing
#         inputs = processor(images=image_tensor.squeeze(0).detach().cpu(), return_tensors="pt").to(device)
#         embedding = model.get_image_features(**inputs)
#         embedding = F.normalize(embedding, p=2, dim=-1)
#
#         # Compute cosine similarity across the embedding dimension and take the mean to get a scalar
#         cosine_sim = F.cosine_similarity(embedding, target_tensor, dim=-1).mean()
#
#         # Compute loss: maximize cosine similarity and minimize perturbation
#         # Negative cosine similarity is used because we want to maximize similarity
#         perturbation = image_tensor - preprocess_image(image).unsqueeze(0).to(device)
#         loss = -cosine_sim + lambda_reg * torch.norm(perturbation)
#
#         # Backward pass
#         print(f"Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}, Cosine Similarity: {cosine_sim.item():.4f}")
#         loss.backward()
#         optimizer.step()
#
#         # Clamp the image tensor to maintain valid pixel range and limit perturbation
#         with torch.no_grad():
#             perturbation = torch.clamp(perturbation, -epsilon, epsilon)
#             image_tensor.copy_(torch.clamp(preprocess_image(image).unsqueeze(0).to(device) + perturbation, 0, 1))
#
#         # Optional: Print progress every 20 steps
#         if (step + 1) % 20 == 0 or step == 0:
#             print(f"Step {step + 1}/{num_steps}, Cosine Similarity: {cosine_sim.item():.4f}, Loss: {loss.item():.4f}")
#
#     # Detach and convert to PIL Image
#     modified_image = image_tensor.detach().cpu().squeeze(0)
#     modified_image_pil = transforms.ToPILImage()(modified_image)
#
#     # Compute final similarity
#     with torch.no_grad():
#         inputs = processor(images=modified_image_pil, return_tensors="pt").to(device)
#         modified_embedding = model.get_image_features(**inputs)
#         modified_embedding = F.normalize(modified_embedding, p=2, dim=-1)
#         final_similarity = F.cosine_similarity(modified_embedding, target_tensor, dim=-1).item()
#
#     return modified_image_pil, final_similarity

# def modify_image(
#         image_path,
#         target_vector,
#         model,
#         processor,
#         device,
#         num_steps=100,
#         learning_rate=0.01,
#         lambda_reg=0.1,
#         epsilon=0.05
# ):
#     """
#     Modify an image to align its CLIP embedding with the target vector.
#
#     Parameters:
#     - image_path (str): Path to the image to be modified.
#     - target_vector (np.array): Target embedding vector (1D array).
#     - model (CLIPModel): Pre-trained CLIP model.
#     - device (str): Device to run computations on ('cuda' or 'cpu').
#     - num_steps (int): Number of optimization steps.
#     - learning_rate (float): Learning rate for optimizer.
#     - lambda_reg (float): Regularization strength.
#     - epsilon (float): Maximum allowed perturbation per pixel.
#
#     Returns:
#     - modified_image (PIL.Image): The optimized image.
#     - final_similarity (float): Cosine similarity with the target vector after modification.
#     """
#     # Load and preprocess the original image
#     image = load_image(image_path)
#     original_image_tensor = preprocess_image(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
#     original_image_tensor = original_image_tensor.detach()  # Detach to prevent gradients flowing into original image
#
#     # Initialize the image tensor to be optimized
#     image_tensor = original_image_tensor.clone().requires_grad_(True)
#
#     print("Current image tensor shape:", image_tensor.shape)
#
#     # Define optimizer
#     optimizer = torch.optim.Adam([image_tensor], lr=learning_rate)
#
#     # Define CLIP normalization (mean and std)
#     clip_normalize = transforms.Normalize(
#         mean=[0.48145466, 0.4578275, 0.40821073],
#         std=[0.26862954, 0.26130258, 0.27577711]
#     )
#
#     # Convert target_vector to torch tensor and normalize
#     target_tensor = torch.tensor(target_vector, dtype=torch.float32).to(device)
#     target_tensor = F.normalize(target_tensor, p=2, dim=0)
#
#     # Set model to evaluation mode
#     model.eval()
#
#     for step in range(num_steps):
#         optimizer.zero_grad()
#
#         # Normalize the image tensor as per CLIP's requirements
#         normalized_image = clip_normalize(image_tensor)
#
#         # Get image features
#         embedding = model.get_image_features(pixel_values=normalized_image)
#         embedding = F.normalize(embedding, p=2, dim=-1)
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
#             print(
#                 f"Step {step + 1}/{num_steps}, Grad Norm: {grad_norm:.4f}, Loss: {loss.item():.4f}, Cosine Similarity: {cosine_sim.item():.4f}")
#         else:
#             print(f"Step {step + 1}/{num_steps}, No gradients computed.")
#
#         # Optimizer step
#         optimizer.step()
#
#         # Clamp the image tensor to maintain valid pixel range and limit perturbation
#         with torch.no_grad():
#             # Calculate perturbation and clamp
#             perturbation = torch.clamp(image_tensor - original_image_tensor, -epsilon, epsilon)
#             # Apply perturbation
#             image_tensor.copy_(torch.clamp(original_image_tensor + perturbation, 0, 1))
#
#         # Optional: Print progress every 20 steps
#         if (step + 1) % 20 == 0 or step == 0:
#             print(f"Step {step + 1}/{num_steps}, Cosine Similarity: {cosine_sim.item():.4f}, Loss: {loss.item():.4f}")
#
#     # Detach and convert to PIL Image
#     modified_image = image_tensor.detach().cpu().squeeze(0)
#     modified_image_pil = transforms.ToPILImage()(modified_image)
#
#     # Compute final similarity
#     with torch.no_grad():
#         normalized_modified_image = clip_normalize(preprocess_image(modified_image_pil).unsqueeze(0).to(device))
#         modified_embedding = model.get_image_features(pixel_values=normalized_modified_image)
#         modified_embedding = F.normalize(modified_embedding, p=2, dim=-1)
#         final_similarity = F.cosine_similarity(modified_embedding, target_tensor, dim=-1).item()
#
#     return modified_image_pil, final_similarity


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
        verbose=True
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
    image = load_image(image_path)
    original_image_tensor = preprocess_image(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
    original_image_tensor = original_image_tensor.detach()  # Detach to prevent gradients flowing into original image

    # Initialize the image tensor to be optimized
    image_tensor = original_image_tensor.clone().requires_grad_(True)

    if verbose:
        print("Initial image tensor shape:", image_tensor.shape)

    # Define optimizer
    optimizer = torch.optim.Adam([image_tensor], lr=learning_rate)

    # Define CLIP normalization (mean and std)
    clip_normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    # Convert target_vector to torch tensor and normalize
    target_tensor = torch.tensor(target_vector, dtype=torch.float32).to(device)
    target_tensor = F.normalize(target_tensor, p=2, dim=0)

    # Set model to evaluation mode
    model.eval()

    # Initialize variables for early stopping
    previous_loss = float('inf')
    patience = 10  # Number of steps to wait for improvement
    patience_counter = 0

    for step in range(num_steps):
        optimizer.zero_grad()

        # Normalize the image tensor as per CLIP's requirements
        normalized_image = clip_normalize(image_tensor)

        # Get image features
        embedding = model.get_image_features(pixel_values=normalized_image)
        embedding = F.normalize(embedding, p=2, dim=-1)

        # Compute cosine similarity and aggregate to scalar
        cosine_sim = F.cosine_similarity(embedding, target_tensor, dim=-1).mean()

        # Compute perturbation norm
        perturbation = image_tensor - original_image_tensor
        reg_loss = lambda_reg * torch.norm(perturbation)

        # Compute loss: maximize cosine similarity and minimize perturbation
        loss = -cosine_sim + reg_loss

        # Backward pass
        loss.backward()

        # Check gradients
        if image_tensor.grad is not None:
            grad_norm = image_tensor.grad.norm().item()
            if verbose:
                print(
                    f"Step {step + 1}/{num_steps}, Grad Norm: {grad_norm:.4f}, Loss: {loss.item():.4f}, Cosine Similarity: {cosine_sim.item():.4f}")
        else:
            if verbose:
                print(f"Step {step + 1}/{num_steps}, No gradients computed.")
            grad_norm = 0.0

        # Optimizer step
        optimizer.step()

        # Clamp the image tensor to maintain valid pixel range and limit perturbation
        with torch.no_grad():
            # Calculate perturbation and clamp
            perturbation = torch.clamp(image_tensor - original_image_tensor, -epsilon, epsilon)
            # Apply perturbation
            image_tensor.copy_(torch.clamp(original_image_tensor + perturbation, 0, 1))

        # Early Stopping Check
        current_loss = loss.item()
        if verbose:
            print(f"Step {step + 1}/{num_steps}, Current Loss: {current_loss:.4f}")

        if abs(previous_loss - current_loss) < 1e-6:
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

    # Detach and convert to PIL Image
    modified_image = image_tensor.detach().cpu().squeeze(0)
    modified_image_pil = transforms.ToPILImage()(modified_image)

    # Compute final similarity
    with torch.no_grad():
        normalized_modified_image = clip_normalize(preprocess_image(modified_image_pil).unsqueeze(0).to(device))
        modified_embedding = model.get_image_features(pixel_values=normalized_modified_image)
        modified_embedding = F.normalize(modified_embedding, p=2, dim=-1)
        final_similarity = F.cosine_similarity(modified_embedding, target_tensor, dim=-1).item()

    return modified_image_pil, final_similarity


# def modify_image(
#         image_path,
#         target_vector,
#         model,
#         processor,
#         device,
#         num_steps=300,
#         learning_rate=0.01,
#         lambda_reg=0.1,
#         epsilon=0.05
# ):
#     """
#     Modify an image to align its CLIP embedding with the target vector.
#
#     Parameters:
#     - image_path (str): Path to the image to be modified.
#     - target_vector (np.array): Target embedding vector.
#     - model (CLIPModel): Pre-trained CLIP model.
#     - processor (CLIPProcessor): CLIP processor.
#     - device (str): Device to run computations on.
#     - num_steps (int): Number of optimization steps.
#     - learning_rate (float): Learning rate for optimizer.
#     - lambda_reg (float): Regularization strength.
#     - epsilon (float): Maximum allowed perturbation per pixel.
#
#     Returns:
#     - modified_image (PIL.Image): The optimized image.
#     - similarity (float): Cosine similarity with the target vector.
#     """
#     # Load and preprocess image
#     image = load_image(image_path)
#     image_tensor = preprocess_image(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
#     image_tensor.requires_grad = True
#     print("cur img shape =====================")
#     print(image_tensor.shape)
#     # Define optimizer
#     optimizer = torch.optim.Adam([image_tensor], lr=learning_rate)
#
#     # Convert target_vector to torch tensor
#     target_tensor = torch.tensor(target_vector).to(device)
#     target_tensor = F.normalize(target_tensor, p=2, dim=0)
#
#     for step in range(num_steps):
#         optimizer.zero_grad()
#
#         # Forward pass: get embedding
#         # inputs = processor(images=image_tensor.squeeze(0).cpu(), return_tensors="pt").to(device)
#         inputs = preprocess_image()
#         embedding = model.get_image_features(**inputs)
#         embedding = F.normalize(embedding, p=2, dim=-1)
#
#         # # Compute cosine similarity
#         # cosine_sim = F.cosine_similarity(embedding, target_tensor, dim=0)
#         #
#         # # Compute loss: maximize cosine similarity and minimize perturbation
#         # # loss = -cosine_sim + lambda_reg * torch.norm(image_tensor - preprocess_image(image).unsqueeze(0).to(device))
#         # loss = -cosine_sim + lambda_reg * torch.norm(
#         #     image_tensor - preprocess_image(image).unsqueeze(0).to(device)).mean()
#         # Backward pass
#         # Compute cosine similarity across the embedding dimension and take the mean to get a scalar
#         cosine_sim = F.cosine_similarity(embedding, target_tensor, dim=-1).mean()
#
#         # Compute loss: maximize cosine similarity and minimize perturbation
#         # Negative cosine similarity is used because we want to maximize similarity
#         perturbation = image_tensor - preprocess_image(image).unsqueeze(0).to(device)
#         loss = -cosine_sim + lambda_reg * torch.norm(perturbation)
#
#         loss.backward()
#         optimizer.step()
#
#         # Clamp the image tensor to maintain valid pixel range and limit perturbation
#         with torch.no_grad():
#             perturbation = torch.clamp(image_tensor - preprocess_image(image).unsqueeze(0).to(device), -epsilon,
#                                        epsilon)
#             image_tensor.copy_(torch.clamp(preprocess_image(image).unsqueeze(0).to(device) + perturbation, 0, 1))
#
#         # Optional: Print progress
#         if (step + 1) % 20 == 0 or step == 0:
#             print(f"Step {step + 1}/{num_steps}, Cosine Similarity: {cosine_sim.item():.4f}, Loss: {loss.item():.4f}")
#
#     # Detach and convert to PIL Image
#     modified_image = image_tensor.detach().cpu().squeeze(0)
#     modified_image_pil = transforms.ToPILImage()(modified_image)
#
#     # Compute final similarity
#     inputs = processor(images=modified_image_pil, return_tensors="pt").to(device)
#     with torch.no_grad():
#         modified_embedding = model.get_image_features(**inputs)
#     modified_embedding = F.normalize(modified_embedding, p=2, dim=-1)
#     final_similarity = F.cosine_similarity(modified_embedding, target_tensor, dim=-1).item()
#
#     return modified_image_pil, final_similarity


def extract_features(image, model, processor):
    """
    Extract the image features using the CLIP model.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    # Normalize the features
    image_features = F.normalize(image_features, p=2, dim=-1)
    return image_features.squeeze(0)  # Remove batch dimension


def perform_attack_on_unsampled(
        unsampled_indices,
        image_paths,
        X_sell,
        target_vector,
        model,
        processor,
        device,
        output_dir='modified_images',
        num_steps=100,
        learning_rate=0.1,
        lambda_reg=0.1,
        epsilon=0.05
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
    modified_indices = []
    similarities = {}
    img_mapping = {}

    for idx in tqdm(unsampled_indices, desc="Performing Attack on Unsampled Images"):
        image_path = image_paths[idx]
        modified_image, similarity = modify_image(
            image_path=image_path,
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
        modified_image_path = os.path.join(output_dir, f'modified_{Path(image_path).name}')
        modified_image.save(modified_image_path)
        o_image_path = os.path.join(output_dir, f'o_{Path(image_path).name}')
        o_image = load_image(image_path)
        o_image = preprocess_image(o_image)

        # Convert the tensor back to a PIL image
        o_image = transforms.ToPILImage()(o_image)

        # Save the image
        o_image.save(o_image_path)

        img_mapping[modified_image_path] = image_path
        # Update the feature matrix
        X_sell[idx] = extract_features(modified_image, model, processor).cpu().numpy()
        # Record the modification
        modified_indices.append(idx)
        similarities[idx] = similarity

    return modified_indices, similarities


def evaluate_attack(
        args,
        dataset='./data',
        data_dir='./data',
        batch_size=1,
        csv_path="druglib/druglib.csv",
        img_path="/images",
        num_buyer=2,
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

    data = utils.get_data(
        dataset=dataset,
        num_buyer=num_buyer * batch_size,
        num_seller=num_seller,
        num_val=num_val,
        dim=num_dim,
        noise_level=args.noise_level,
        random_state=args.random_seed,
        cost_range=args.cost_range,
        cost_func=args.cost_func,
    )
    # Extract relevant data
    x_s = data["X_sell"].astype(np.float32)
    y_s = data["y_sell"].astype(np.float32)
    x_b = data["X_buy"].astype(np.float32)
    y_b = data["y_buy"].astype(np.float32)
    # todo change the costs
    costs = data.get("costs_sell")
    index_s = data['index_sell']
    index_b = data['index_buy']
    index_v = data['index_val']
    img_paths = data['img_paths']
    print("Data type of index_s:", len(img_paths))
    sell_img_path = [img_paths[i] for i in index_s]
    print(f"Seller Data Shape: {x_s.shape}".center(40, "="))
    print(f"Buyer Data Shape: {x_b.shape}".center(40, "="))
    if costs is not None:
        print(f"Costs Shape: {costs.shape}".center(40, "="))

    # Step 2: Initial Data Selection
    initial_results = initial_selection(
        x_s=x_s,
        y_s=y_s,
        x_b=x_b,
        y_b=y_b,
        num_iters=num_iters,
        reg_lambda=reg_lambda,
        costs=costs
    )

    # Step 3: Identify Selected and Unselected Data Points
    selected_indices_initial, unsampled_indices_initial = identify_selected_unsampled(
        weights=initial_results['weights'],
        num_select=num_select,
    )

    print(f"Initial Selected Indices: {selected_indices_initial}")
    print(f"Number of Unselected Data Points: {len(unsampled_indices_initial)}")

    # learn the vector can be selected
    target_vector = x_s[selected_indices_initial].mean(axis=0)
    target_vector = target_vector / np.linalg.norm(target_vector)

    modified_images_path = f'./result/{dataset}/modified_images'
    # Step 4: Perform Attack on Unselected Data Points
    modified_indices, similarities = perform_attack_on_unsampled(
        unsampled_indices=unsampled_indices_initial,
        image_paths=sell_img_path,
        X_sell=x_s,
        target_vector=target_vector,
        model=model,
        processor=processor,
        device=device,
        num_steps=args.attack_steps,  # 200
        learning_rate=args.attack_learning_rate,  # learning rate 0.1
        lambda_reg=args.attack_lambda_reg,  # 0.1
        epsilon=0.05,
        output_dir=modified_images_path,
    )

    print(f"Number of Data Points Modified: {len(modified_indices)}")

    # Step 5: Re-run Data Selection on Modified Data
    updated_results = re_run_selection(
        x_s_modified=x_s,
        y_s=y_s,
        x_b=x_b,
        y_b=y_b,
        num_iters=num_iters,
        reg_lambda=reg_lambda,
    )

    # Step 6: Identify Updated Selected Data Points
    selected_indices_updated, _ = identify_selected_unsampled(
        weights=updated_results['weights'],
        num_select=num_select,
    )

    print(f"Updated Selected Indices: {selected_indices_updated}")

    # Step 7: Evaluate Attack Success
    success_rate, num_success = evaluate_attack_success(
        initial_selected=selected_indices_initial,
        updated_selected=selected_indices_updated,
        modified_indices=modified_indices,
    )

    print(f"Attack Success Rate: {success_rate * 100:.2f}% ({num_success}/{len(modified_indices)})")

    # Step 8: Plot Weight Distributions
    plot_selection_weights(
        initial_weights=initial_results['weights'],
        updated_weights=updated_results['weights'],
        save_path=f'{result_dir}/res.jpg',  # Set a path to save the plot if desired
    )

    # Step 9: Save Results (Optional)
    if save_results_flag:
        results = {
            'initial_selected_indices': selected_indices_initial,
            'unsampled_indices_initial': unsampled_indices_initial,
            'x_s_modified': x_s,
            'updated_selected_indices': selected_indices_updated,
            'modified_indices': modified_indices,
            'attack_success_rate': success_rate,
            'num_successfully_selected': num_success,
        }
        save_results(
            results=results,
            save_dir=result_dir,
            save_name=save_name,
        )
        print(f"Results saved to {result_dir}/{save_name}.pkl")

    return {
        'initial_selected_indices': selected_indices_initial,
        'unsampled_indices_initial': unsampled_indices_initial,
        'updated_selected_indices': selected_indices_updated,
        'modified_indices': modified_indices,
        'attack_success_rate': success_rate,
        'num_successfully_selected': num_success,
    }


device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Data subset selection with experimental design informed loss using Frank-Wolfe optimization",
    )
    parser.add_argument("--random_seed", default=12345, help="random seed")
    parser.add_argument("--figure_dir", default="../figures")
    parser.add_argument("--result_dir", default="../results")
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
        "--attack_step",
        default=200,
        type=int,
        help="attack steps",
    )

    parser.add_argument(
        "--attack_lambda_reg",
        default=0,
        type=float,
        help="attack reg",
    )
    parser.add_argument(
        "--attack_learning_rate",
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
        "--attack_steps",
        default=200,
        type=int,
        help="attack step",
    )
    args = parser.parse_args()

    # Define experiment parameters
    experiment_params = {
        'dataset': args.dataset,
        'data_dir': './data',
        'csv_path': "./fitzpatrick17k/fitzpatrick-mod.csv",
        'img_path' './fitzpatrick17k/images'
        'num_buyer': 2,
        'num_seller': 1000,
        'num_val': 1,
        'max_eval_range': 50,
        'eval_step': 5,
        'num_iters': 500,
        'reg_lambda': 0.1,
        'attack_strength': 0.1,
        'save_results_flag': True,
        'result_dir': 'results',
        'save_name': 'attack_evaluation',
    }

    # Run the attack evaluation experiment
    results = evaluate_attack(args, **experiment_params)

    # Print final results
    print("\nFinal Attack Evaluation Metrics:")
    print(f"Attack Success Rate: {results['attack_success_rate'] * 100:.2f}%")
    print(f"Number of Successfully Selected Modified Data Points: {results['num_successfully_selected']}")
