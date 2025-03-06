import numpy as np
import random
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, softmax
from torcheval.metrics.functional import multiclass_f1_score
from typing import Dict

from marketplace.utils.gradient_market_utils.clustering import optimal_k_gap, kmeans
from model.utils import get_model, load_model, apply_gradient_update


# -----------------------------------------------------
# External modules you mentioned (assumed to exist):
#   from train import train_model, evaluate_model, kappa
#   from quant_aggregation import integrated_quant_aggregation
#   from model_saver import load_model, save_model, get_backup_name_from_model_name
#   from cluster import optimal_k, kmeans
#   from dataset import dataset_output_dim
#   from homo_encryption import private_model_evaluation
#
# Make sure these are correctly imported according to your environment.
# -----------------------------------------------------

class Aggregator:
    """
    Aggregation mechanism for a federated learning marketplace.
    Handles:
      - gradient updates retrieval
      - custom cluster-based selection logic (martFL)
      - optional quantization
      - server role rotation / outlier filtering
    """

    def __init__(self,
                 save_path: str,
                 n_seller: int,
                 dataset_name: str,
                 model_structure: nn.Module = None,
                 quantization: bool = False,
                 aggregation_method: str = "martfl",
                 change_base: bool = True,
                 device=None,
                 clip_norm=0.01, loss_fn=None,
                 buyer_data_loader=None):
        """
        :param save_path:         Name/identifier of the current experiment.
        :param n_seller:   Total number of participant clients.
        :param model_structure:  A torch.nn.Module structure (uninitialized) used to get param shapes.
        :param quantization:     Whether to do quantized aggregation.
        :param device:           Torch device (CPU/GPU) to run computations.
        """
        self.save_path = save_path
        self.n_seller = n_seller
        self.model_structure = model_structure
        self.device = device
        self.quantization = quantization
        self.global_model = get_model(dataset_name)
        # An example to track "best candidate" or further logic if you need:
        self.max_indexes = [0]
        self.aggregation_method = aggregation_method
        self.baseline_id = None
        self.change_base = change_base
        self.clip_norm = clip_norm
        self.buyer_data_loader = buyer_data_loader
        self.loss_fn = loss_fn

    # ---------------------------
    # Gradient update utilities
    # ---------------------------
    def get_params(self) -> Dict[str, torch.Tensor]:
        """
        Return the current global model parameters as a dict or a
        list. This is just an example interface. Adjust as needed.
        """
        return {
            k: v.clone().detach()
            for k, v in self.global_model.state_dict().items()
        }

    def set_params(self, params: Dict[str, torch.Tensor]):
        """
        Set global model parameters from a dict or list of Tensors.
        """
        self.global_model.load_state_dict(params)

    def apply_gradient(self, aggregated_gradient, learning_rate: float = 1.0):
        """
        Update the global model parameters by descending along aggregated_gradient.
        Convert the aggregated gradient into a single numpy array (if it's a list) and
        then apply the update to self.global_model.
        """
        # If aggregated_gradient is a list of tensors, flatten and convert to numpy array.
        if isinstance(aggregated_gradient, list):
            aggregated_gradient = np.concatenate(
                [grad.cpu().numpy().ravel() for grad in aggregated_gradient]
            )

        # Check if the aggregated gradient is empty.
        if aggregated_gradient.size == 0:
            return

        # Convert the numpy array back to a torch tensor.
        aggregated_torch = torch.from_numpy(aggregated_gradient).float().to(self.device)

        # Get model state dict and apply updates
        with torch.no_grad():
            current_params = self.global_model.state_dict()
            idx = 0
            updated_params = {}

            # Total number of elements in the model
            total_elements = sum(tensor.numel() for tensor in current_params.values())

            for name, tensor in current_params.items():
                numel = tensor.numel()
                grad_slice = aggregated_torch[idx: idx + numel].reshape(tensor.shape)
                idx += numel
                updated_params[name] = tensor - learning_rate * grad_slice

            if idx != aggregated_torch.numel():
                raise ValueError("Mismatch between aggregated gradient and model parameters count.")

            # Load updated parameters back into the model
            self.global_model.load_state_dict(updated_params)

    def aggregate(self, global_epoch, seller_updates, buyer_updates, clip = False):
        if self.aggregation_method == "martfl":
            return self.martFL(global_epoch=global_epoch, seller_updates=seller_updates, buyer_updates=buyer_updates, clip = clip)
        elif self.aggregation_method == "fedavg":
            return self.fedavg(global_epoch=global_epoch, seller_updates=seller_updates, buyer_updates=buyer_updates)
        else:
            raise NotImplementedError(f"current aggregator not implemented {self.aggregation_method}")

    # ---------------------------
    # Main Federated Aggregation (martFL)

    # ---------------------------
    def martFL(self,
               global_epoch: int,
               seller_updates,
               buyer_updates,
               ground_truth_model=None, clip=False):
        """
        Performs the martFL aggregation:
          1) Compute each client's flattened & clipped gradient update.
          2) Compute the cosine similarity of each client's update to the current baseline update.
          3) Cluster the cosine similarities to identify outliers.
          4) Normalize the resulting "non-outlier" scores to derive weights.
          5) Aggregate the updates using the computed weights.
          6) If change_base is True, select a new baseline seller for the next round.

        Returns:
           aggregated_gradient: the aggregated update (list of tensors, same structure as model parameters)
           selected_ids: list of indices of sellers whose gradients are selected (non-outliers)
           outlier_ids: list of indices of sellers whose gradients are labeled as outliers
           baseline_seller_id: ID of the seller selected as the next baseline (if change_base=True)
        """
        print(f"Global epoch: {global_epoch}, Change baseline: {self.change_base}")

        # Number of sellers
        self.n_seller = len(seller_updates)
        seller_ids = list(seller_updates.keys())

        print("Starting gradient aggregation")
        aggregated_gradient = [
            torch.zeros_like(param, device=self.device)
            for param in self.model_structure.parameters()
        ]

        print("Flattening and clipping gradients")
        # Process each seller update: clip then flatten
        clients_update_flattened = [
            flatten(clip_gradient_update(update, clip_norm=self.clip_norm))
            for sid, update in seller_updates.items()
        ]
        # clients_update_flattened = [
        #     flatten(update)
        #     for sid, update in seller_updates.items()
        # ]
        # Process the current baseline update
        if self.baseline_id is not None and self.baseline_id in seller_updates:
            print(f"Using seller {self.baseline_id} as baseline")
            if clip:
                cg = clip_gradient_update(seller_updates[self.baseline_id], clip_norm=self.clip_norm)
            else:
                cg = seller_updates[self.baseline_id]
            baseline_update_flattened = flatten(cg)
        else:
            print("Using buyer as baseline")
            if clip:
                cg = clip_gradient_update(buyer_updates, clip_norm=self.clip_norm)
            else:
                cg = buyer_updates

            baseline_update_flattened = flatten(cg)
            self.baseline_id = None

        print("Computing cosine similarities")
        # Vectorize cosine similarity:
        clients_stack = torch.stack(clients_update_flattened)  # shape: (n_seller, d)
        baseline = baseline_update_flattened.unsqueeze(0)  # shape: (1, d)
        cosine_similarities = torch.nn.functional.cosine_similarity(baseline, clients_stack, dim=1)
        np_cosine_result = cosine_similarities.cpu().numpy()
        np_cosine_result = np.nan_to_num(np_cosine_result, nan=0.0)
        print(f"Similarity metrics: {np_cosine_result}")

        # Clustering on cosine similarities using Gap Statistics
        diameter = np.max(np_cosine_result) - np.min(np_cosine_result)
        print(f"Similarity diameter: {diameter:.4f}")

        # Get optimal number of clusters using Gap statistics
        n_clusters = optimal_k_gap(np_cosine_result)
        if n_clusters == 1 and diameter > 0.05:
            n_clusters = 2

        # Primary clustering
        clusters, centroids = kmeans(np_cosine_result, n_clusters)
        print(f"Centroids: {centroids}")
        print(f"{n_clusters} Clusters: {clusters}")

        # Identify the highest-score cluster centroid
        best_centroid_idx = np.argmax(centroids)
        center = centroids[best_centroid_idx]

        # Secondary clustering for further outlier detection (with k=2)
        if n_clusters == 1:
            clusters_secondary = [1] * self.n_seller
        else:
            clusters_secondary, _ = kmeans(np_cosine_result, 2)
        print(f"Secondary Clustering: {clusters_secondary}")

        # Determine border for outlier detection (max distance from center)
        border = 0.0
        for i, (cos_sim, sec_cluster) in enumerate(zip(np_cosine_result, clusters_secondary)):
            if n_clusters == 1 or sec_cluster != 0:
                dist = abs(center - cos_sim)
                if dist > border:
                    border = dist
        print(f"Border: {border:.4f}")

        # Track different seller quality types
        high_quality_sellers = []  # P₁ in the paper
        qualified_sellers = []  # P₂ in the paper
        outlier_sellers = []  # Low-quality or malicious sellers

        # Mark outliers: Build a list of non_outlier scores
        non_outliers = [1.0 for _ in range(self.n_seller)]
        for i in range(self.n_seller):
            seller_id = seller_ids[i]

            # Skip the current baseline seller if it exists
            if seller_id == self.baseline_id:
                continue

            if clusters_secondary[i] == 0 or np_cosine_result[i] == 0.0:
                # Low-quality model (mark as outlier)
                non_outliers[i] = 0.0
                outlier_sellers.append(seller_id)
            else:
                if clusters[i] == best_centroid_idx:
                    # High-quality model in the best cluster
                    high_quality_sellers.append(seller_id)
                    non_outliers[i] = 1.0
                else:
                    # Qualified but weighted model
                    dist = abs(center - np_cosine_result[i])
                    non_outliers[i] = 1.0 - dist / (border + 1e-6)
                    qualified_sellers.append(seller_id)

        # Identify selected (inlier) and outlier seller indices
        selected_ids = [i for i in range(self.n_seller) if non_outliers[i] > 0.0]
        outlier_ids = [i for i in range(self.n_seller) if non_outliers[i] == 0.0]

        print(f"High-quality sellers: {high_quality_sellers}")
        print(f"Qualified sellers: {qualified_sellers}")
        print(f"Outlier sellers: {outlier_sellers}")

        # Random sampling from P₂ (qualified sellers) if needed
        # If P₂ is empty and P₁ is small, select some random additional sellers
        if not qualified_sellers and len(high_quality_sellers) < 0.5 * self.n_seller:
            # Get all candidate sellers excluding high quality and outliers
            all_remaining_sellers = list(set(seller_ids) -
                                         set(high_quality_sellers) -
                                         set(outlier_sellers))

            # Calculate random sample size (β in the paper)
            beta = 0.1  # Default in the paper is 10%
            random_sample_count = max(1, int(beta * self.n_seller))

            # Sample randomly from remaining sellers
            if all_remaining_sellers:
                sampled_qualified = random.sample(all_remaining_sellers,
                                                  min(random_sample_count, len(all_remaining_sellers)))
                print(f"Randomly sampled additional qualified sellers: {sampled_qualified}")

                # Add to qualified sellers and update weights
                qualified_sellers.extend(sampled_qualified)
                for seller_id in sampled_qualified:
                    i = seller_ids.index(seller_id)
                    non_outliers[i] = 0.5  # Assign moderate weight
                    if i not in selected_ids:
                        selected_ids.append(i)

        # Use the non_outlier scores as final weights
        for i in range(self.n_seller):
            np_cosine_result[i] = non_outliers[i]

        cosine_weight = torch.tensor(np_cosine_result, dtype=torch.float, device=self.device)
        denom = torch.sum(torch.abs(cosine_weight)) + 1e-6
        weight = cosine_weight / denom

        print(f"Aggregation weights: {weight}")

        # Final aggregation: sum weighted gradients
        for idx, (gradient, wt) in enumerate(zip(seller_updates.values(), weight)):
            add_gradient_updates(aggregated_gradient, gradient, weight=wt)

        # ========== IMPROVED DYNAMIC BASELINE ADJUSTMENT ==========
        if self.change_base:
            print("Performing baseline adjustment for next round")

            # Candidate sellers include all high-quality sellers and qualified sellers
            candidate_sellers = high_quality_sellers + qualified_sellers

            # If no candidates, consider all non-outlier sellers
            if not candidate_sellers:
                candidate_sellers = [seller_ids[i] for i in selected_ids]

            print(f"Baseline candidate sellers: {candidate_sellers}")

            # Early exit if no candidates
            if not candidate_sellers:
                print("No viable baseline candidates found")
                return aggregated_gradient, selected_ids, outlier_ids, None

            # Evaluate each candidate seller using Kappa coefficient
            max_kappa = float('-inf')
            best_seller = None

            for seller_id in candidate_sellers:
                # Create a model with this seller's update applied
                seller_idx = seller_ids.index(seller_id)

                torch_updates = [torch.from_numpy(delta).float() if isinstance(delta, np.ndarray) else delta
                                 for delta in seller_updates[seller_id]]

                # Apply the seller's update to get their model
                updated_model = apply_gradient_update(self.global_model, torch_updates)

                kappa = martfl_eval(updated_model, self.buyer_data_loader, self.loss_fn, self.device, num_classes=10)[2]

                print(f"Seller {seller_id} Kappa score: {kappa:.4f}")

                # Keep track of the best seller
                if kappa > max_kappa:
                    max_kappa = kappa
                    best_seller = seller_id

            self.baseline_id = best_seller
            print(f"Selected new baseline seller for next round: {self.baseline_id} with score {max_kappa:.4f}")

        # Return aggregated gradient, selected seller IDs, outlier seller IDs, and new baseline seller ID
        return aggregated_gradient, selected_ids, outlier_ids

    def fedavg(self,
               global_epoch: int,
               seller_updates,
               buyer_updates,
               change_base: bool = False,
               ground_truth_model=None):
        print("global_epoch :", global_epoch)

        # Number of sellers
        self.n_seller = len(seller_updates)

        # Initialize aggregated gradients to zeros
        aggregated_gradient = [
            torch.zeros_like(param, device=self.device)
            for param in self.model_structure.parameters()
        ]

        # If weights for sellers are provided (e.g., as self.seller_weights), use them.
        # Otherwise, assign equal weights.
        seller_weights = [1.0 for _ in seller_updates]

        # Normalize weights so they sum to 1
        total_weight = sum(seller_weights)
        normalized_weights = [w / total_weight for w in seller_weights]

        # Final aggregation: sum weighted gradients.
        for idx, (gradient, wt) in enumerate(zip(seller_updates.values(), normalized_weights)):
            add_gradient_updates(aggregated_gradient, gradient, weight=wt)
            # Optionally, compute and print the norm of each aggregated parameter:
            # norms = [torch.norm(param).item() for param in aggregated_gradient]
            # print(f"After update {idx}, norms: {norms}")

        return aggregated_gradient, [i for i in range(self.n_seller)], []


# ---------------------------------------------------------
#  HELPER FUNCTIONS
# ---------------------------------------------------------

def martfl_eval(model, dataloader, loss_fn, device, num_classes):
    """
    Evaluates a model on the given dataloader and calculates various metrics including Kappa.

    Parameters:
        model: The neural network model to evaluate
        dataloader: DataLoader containing the evaluation data
        loss_fn: Loss function to calculate the loss
        device: Device to run the evaluation on (cuda/cpu)
        num_classes: Number of classes in the classification task

    Returns:
        tuple: (average_loss, accuracy, kappa_score, f1_score)
    """
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)

    # Initialize metrics
    correct_samples = 0.0
    total_samples = 0.0
    total_loss = 0.0

    # Lists to store predictions and true labels
    y_true = []
    y_score = []  # Probability scores
    y_pred = []  # Predicted class labels

    # No gradient calculation during evaluation
    with torch.no_grad():
        # Iterate through batches
        for batch_idx, batch in enumerate(dataloader):
            # Handle different dataset formats
            # if isinstance(batch, Batch):  # For text datasets like TREC
            #     data, target = batch.text, batch.label
            #     data = data.permute(1, 0)  # Transpose for proper shape
            # else:  # For image datasets like MNIST/CIFAR
            data, target = batch[0], batch[1]
            # Move data to the specified device
            data, target = data.to(device), target.to(device)
            print(data.device, target.device)
            # Forward pass
            outputs = model(data)

            # Calculate loss
            loss = loss_fn(outputs, target)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Update metrics
            correct_samples += (predicted == target).sum().item()
            total_samples += target.size(0)
            total_loss += loss.item()

            # Store predictions and true labels for metrics calculation
            y_true.extend(target.cpu().numpy())
            y_score.extend(softmax(outputs, dim=1).cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate final metrics
    accuracy = correct_samples / total_samples
    average_loss = total_loss / (batch_idx + 1)

    # Calculate confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate F1 score
    f1 = multiclass_f1_score(
        torch.tensor(y_pred),
        torch.tensor(y_true),
        num_classes=num_classes,
        average='macro'
    ).item()

    # Calculate Cohen's Kappa
    kappa_score = calculate_kappa(cf_matrix)

    return round(average_loss, 6), round(accuracy, 6), kappa_score, f1


def calculate_kappa(confusion_matrix):
    """
    Calculates Cohen's Kappa coefficient from a confusion matrix.

    The Kappa coefficient measures the agreement between predicted and actual classifications,
    accounting for agreement that would happen by chance.

    Parameters:
        confusion_matrix: A numpy array representing the confusion matrix

    Returns:
        float: The Kappa coefficient (between -1 and 1)
    """
    # Total number of samples
    n_samples = np.sum(confusion_matrix)

    # Return 0 if confusion matrix is empty
    if n_samples == 0:
        return 0.0

    # Calculate observed agreement (accuracy)
    observed_agreement = np.trace(confusion_matrix) / n_samples

    # Calculate expected agreement (by chance)
    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    expected_agreement = np.sum(row_sums * col_sums) / (n_samples * n_samples)

    # Calculate Cohen's Kappa
    if expected_agreement == 1:
        return 1.0  # Perfect agreement by chance

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return kappa


def clip_gradient_update(grad_update, clip_norm: float):
    """
    Clamp each parameter update to [-clip_norm, clip_norm].
    If an update is a numpy array/scalar, it will be converted to a torch tensor.
    """
    clipped_updates = []
    for param in grad_update:
        # Convert numpy arrays or scalars to torch tensors
        if isinstance(param, np.ndarray):
            param = torch.tensor(param)
        elif isinstance(param, (np.float32, np.float64)):
            param = torch.tensor(param)
        # Clamp the value and add it to the list.
        update = torch.clamp(param, min=-clip_norm, max=clip_norm)
        clipped_updates.append(update)
    return clipped_updates


def compute_ground_truth_updates(exp_name, old_model_name, new_model, device=None):
    """
    For reference or debugging, compare old_model (on disk)
    to a known 'ground_truth' model and return the parameter deltas.
    """
    old_model = load_model(exp_name, old_model_name, device)
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)

    return [
        (new_param.data - old_param.data)
        for old_param, new_param in zip(old_model.parameters(), new_model.parameters())
    ]


def compute_update_gradients(exp_name, old_model_name, new_model_name, device=None):
    """
    Return the difference between (new_model - old_model).
    Both are loaded from disk by name.
    """
    old_model = load_model(exp_name, old_model_name, device)
    new_model = load_model(exp_name, new_model_name, device)
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [
        (new_param.data - old_param.data)
        for old_param, new_param in zip(old_model.parameters(), new_model.parameters())
    ]


def flatten(parameters):
    """
    Flatten a list of tensors into a single 1D tensor.

    Args:
        parameters: List of parameter tensors or numpy arrays

    Returns:
        Flattened 1D tensor
    """
    # Handle case where input is already a tensor
    if isinstance(parameters, torch.Tensor):
        return parameters.flatten()

    # Handle case where input is a numpy array
    if isinstance(parameters, np.ndarray):
        return torch.tensor(parameters.flatten())

    # For list of parameters (tensors or numpy arrays)
    flattened_tensors = []
    for param in parameters:
        if isinstance(param, torch.Tensor):
            flattened_tensors.append(param.cpu().flatten())
        elif isinstance(param, np.ndarray):
            flattened_tensors.append(torch.tensor(param.flatten()))
        elif param is not None:  # Skip None values
            # Try to convert to tensor safely
            try:
                tensor = torch.tensor(param)
                flattened_tensors.append(tensor.flatten())
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not flatten parameter of type {type(param)}: {e}")
                # Skip this parameter

    # Concatenate all flattened tensors
    if flattened_tensors:
        return torch.cat(flattened_tensors)
    else:
        # Return empty tensor if no valid parameters
        return torch.tensor([])


def unflatten(flattened: torch.Tensor, normal_shape):
    """
    Reshape a flat vector back into the original parameter shapes in 'normal_shape'.
    'normal_shape' should be a list of tensors (or nn.Parameters) from which we
    know the dimension sizes.
    """
    grad_update = []
    current_pos = 0
    for param in normal_shape:
        num_params = param.numel()
        # Get the slice of the flattened tensor corresponding to the current parameter.
        slice_ = flattened[current_pos:current_pos + num_params]
        # Use tuple(param.shape) to ensure the shape is a tuple.
        grad_update.append(slice_.reshape(tuple(param.shape)))
        current_pos += num_params
    return grad_update


def add_update_to_model(model: nn.Module, update, weight=1.0, device=None):
    """
    Add 'update' (list of tensors) to 'model' parameters (in-place).
    """
    if not update:
        return model
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]

    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model


def add_gradient_updates(grad_accumulator, grad_update, weight=1.0):
    """
    In-place: grad_accumulator[i] += weight * grad_update[i]
    This version avoids using .data and instead uses in-place operations.
    """
    assert len(grad_accumulator) == len(grad_update), "Mismatch in grad lists."
    for acc, g in zip(grad_accumulator, grad_update):
        # Ensure g is a tensor (if it's not, convert it)
        if not isinstance(g, torch.Tensor):
            g = torch.tensor(g, device=acc.device)
        else:
            g = g.to(acc.device)
        # Perform the update in-place
        acc.add_(g * weight)


def cosine_xy(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Plain PyTorch cosine similarity between two 1D vectors.
    """
    x = x.detach().cpu()
    y = y.detach().cpu()
    cos_val = torch.cosine_similarity(x, y, dim=0)
    return cos_val.item()
