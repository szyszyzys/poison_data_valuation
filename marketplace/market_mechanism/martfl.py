import random
import threading
from typing import Dict

import numpy as np
import torch
from torch import nn

from marketplace.utils.gradient_market_utils.clustering import kmeans, gap
from marketplace.utils.model_utils import load_model, save_model


# -----------------------------------------------------
# External modules you mentioned (assumed to exist):
#   from train import train_model, evaluate_model, kappa
#   from quant_aggregation import integrated_quant_aggregation
#   from model_saver import load_model, save_model, get_backup_name_from_model_name
#   from cluster import gap, kmeans
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
                 exp_name: str,
                 n_seller: int,
                 n_adversaries: int,
                 backup_models: list,
                 client_models: list,
                 model_structure: nn.Module = None,
                 quantization: bool = False,
                 device=None):
        """
        :param exp_name:         Name/identifier of the current experiment.
        :param n_seller:   Total number of participant clients.
        :param n_adversaries:    Number of adversarial clients (not always used).
        :param backup_models:    List of model checkpoints (names) used as "backup."
        :param client_models:    List of current client model checkpoints.
        :param model_structure:  A torch.nn.Module structure (uninitialized) used to get param shapes.
        :param quantization:     Whether to do quantized aggregation.
        :param device:           Torch device (CPU/GPU) to run computations.
        """
        self.exp_name = exp_name
        self.n_seller = n_seller
        self.n_adversaries = n_adversaries
        self.backup_models = backup_models
        self.client_models = client_models
        self.model_structure = model_structure
        self.device = device
        self.server = 0
        self.quantization = quantization

        # An example to track "best candidate" or further logic if you need:
        self.max_indexes = [0]

    # ---------------------------
    # Gradient update utilities
    # ---------------------------
    def get_update_gradients(self):
        """
        Compute each local model's update relative to its *own* old model.
        i.e. delta = (client_model_i - backup_model_i).
        """
        return [
            compute_update_gradients(self.exp_name, old_m, new_m, self.device)
            for old_m, new_m in zip(self.backup_models, self.client_models)
        ]

    def get_update_gradients_toserver(self):
        """
        Compute each local model's update relative to the *server* old model (index 0).
        i.e. delta_i = (client_model_i - backup_model_0).
        """
        server_backup = self.backup_models[0]
        return [
            compute_update_gradients(self.exp_name, server_backup, new_m, self.device)
            for new_m in self.client_models
        ]

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

    def apply_gradient(self, aggregated_gradient: np.ndarray, learning_rate: float = 1.0):
        """
        Update the global model parameters by descending along
        aggregated_gradient. You must convert the numpy array
        to torch, then apply it to self.global_model.
        """
        if aggregated_gradient.size == 0:
            return

        # Convert to torch
        aggregated_torch = torch.from_numpy(aggregated_gradient).float().to(self.device)
        # Flatten model params, apply update, unflatten
        # (for demonstration, assume you have a flatten/unflatten routine)
        # Or do something simpler if your model is small.
        with torch.no_grad():
            current_params = self.global_model.state_dict()
            idx = 0
            for name, tensor in current_params.items():
                numel = tensor.numel()
                grad_slice = aggregated_torch[idx: idx + numel].reshape(tensor.shape)
                idx += numel
                # Update rule (SGD):
                tensor[...] = tensor - learning_rate * grad_slice

    # ---------------------------
    # Main Federated Aggregation (martFL)
    # ---------------------------
    def martFL(self,
               global_epoch: int,
               server_dataloader,
               loss_fn,
               buyer_updates,
               seller_updates,
               change_base: bool,
               ground_truth_model=None):
        """
        Performs the martFL algorithm:
          1) Calculate each client's gradient update (vs. its own old model).
          2) Compute a 'cosine' score vs. the server's gradient update.
          3) Cluster clients, identify outliers or malicious updates,
             optionally rotate 'server' to the best candidate in that cluster.
          4) Aggregate the selected (non-outlier) updates, with weighting.
          5) Update all client models, optionally with quantized integration.
        :param global_epoch:       Current global epoch index.
        :param server_dataloader:  Dataloader for evaluating potential server models.
        :param loss_fn:            Loss function used in evaluation.
        :param change_base:        If True, attempt to change the "server" client based on cluster logic.
        :param ground_truth_model: A reference model to compare with (for baseline scoring).
        :return: (weight, self.server, baseline_score)
        """

        print("change_base :", change_base)
        print("global_epoch :", global_epoch)

        # 1. Initialize an empty aggregated gradient container.
        # Using torch.zeros_like ensures we create tensors with the same shape and device.
        aggregated_gradient = [
            torch.zeros_like(param, device=self.device)
            for param in self.model_structure.parameters()
        ]

        # 2. Compute each client's clipped gradient update.
        #    The update is flattened and clipped to a norm bound (e.g., 0.01).
        clients_update_flattened = [
            flatten(clip_gradient_update(update, clip_norm=0.01))
            for update in seller_updates
        ]

        # 3. Get the server's update.
        #    Here we clone and detach to ensure the update is not part of any gradient computation.
        server_update_flattened = seller_updates.clone().detach()

        # 4. Compute similarity (using an encrypted cosine function) between the server update
        #    and each client's flattened update. We assume that seller_updates includes the server,
        #    so if needed, skip the server during further processing.
        cosine_result = [
            encrypted_cosine_xy(server_update_flattened, clients_update_flattened[i])
            for i in range(self.n_seller)
        ]
        np_cosine_result = np.array(cosine_result)  # For numerical operations in clustering

        # 5. Run clustering on the cosine similarities.
        #    Compute the diameter (max difference) of the similarity values.
        diameter = np.max(np_cosine_result) - np.min(np_cosine_result)
        print("Diameter:", diameter)

        # Determine the optimal number of clusters using a gap statistic function.
        n_clusters = gap(np_cosine_result)
        # Heuristic: if gap suggests 1 cluster but the diameter is large, force 2 clusters.
        if n_clusters == 1 and diameter > 0.05:
            n_clusters = 2

        # Perform k-means clustering on the cosine similarities.
        clusters, centroids = kmeans(np_cosine_result, n_clusters)
        print("Centroids:", centroids)
        print(f"{n_clusters} Clusters:", clusters)

        # Choose a reference center from the centroids.
        # Here, we use the last centroid if available (consider adapting this logic as needed).
        center = centroids[-1] if centroids else 0.0

        # 6. Secondary clustering check with k=2 for further outlier detection.
        if n_clusters == 1:
            clusters_secondary = [1] * self.n_seller
        else:
            clusters_secondary, _ = kmeans(np_cosine_result, 2)
        print("Secondary Clustering:", clusters_secondary)

        # 7. Determine the border for outlier detection.
        #    We compute the maximum absolute distance from the reference center,
        #    skipping the server (assumed to be at index 0 or a specified index self.server).
        border = 0.0
        for i, (cos_sim, sec_cluster) in enumerate(zip(cosine_result, clusters_secondary)):
            if i == 0 or i == self.server:
                continue

            # For the main cluster (or if only one cluster was identified), measure the distance.
            if n_clusters == 1 or sec_cluster != 0:
                dist = abs(center - cos_sim)
                if dist > border:
                    border = dist
        print("Border:", border)

        # 8. Mark outliers:
        #    Here we initialize a marker list where 1.0 indicates a non-outlier (kept)
        #    and 0.0 will later indicate an outlier.
        non_outliers = [1.0] * self.n_seller
        candidate_server = []

        for i in range(self.n_seller):
            if clusters_secondary[i] == 0 or cosine_result[i] == 0.0:
                # Mark potential “attackers”
                non_outliers[i] = 0.0
            else:
                # We measure how far from the center
                dist = abs(center - cosine_result[i])
                non_outliers[i] = 1.0 - dist / (border + 1e-6)
                # If it is in the last cluster (n_clusters-1), consider it a candidate
                if clusters[i] == n_clusters - 1:
                    candidate_server.append(i)
                    non_outliers[i] = 1.0

        # 8. Combine outlier factor with the original cosines
        for i in range(self.n_seller):
            cosine_result[i] = non_outliers[i]

        # 9. Normalize to get final weights
        cosine_weight = torch.tensor(cosine_result, dtype=torch.float)
        denom = torch.sum(torch.abs(cosine_weight)) + 1e-6
        weight = cosine_weight / denom

        # -------------------------------------------
        # Compare to “ground_truth_model” for baseline_score
        # -------------------------------------------
        ground_truth_updates = compute_ground_truth_updates(
            self.exp_name, self.backup_models[0], ground_truth_model, self.device
        )
        ground_truth_updates_flat = flatten(ground_truth_updates)
        baseline_updates_flat = clients_update_flattened[self.server]
        baseline_score = cosine_xy(ground_truth_updates_flat, baseline_updates_flat)
        print("baseline_score", baseline_score)

        # -------------------------------------------
        # Final aggregation: sum up all updates with 'weight'
        # -------------------------------------------
        update_gradients = self.get_update_gradients()
        for gradient, wt in zip(update_gradients, weight):
            add_gradient_updates(aggregated_gradient, gradient, weight=wt)

        # -------------------------------------------
        # Update each client model
        # -------------------------------------------
        if self.quantization:
            raise NotImplementedError("quantization not implemented.")
        else:
            # Classic approach: update each client by applying aggregated_gradient
            for i, model_name in enumerate(self.client_models):
                model = load_model(
                    self.exp_name,
                    get_backup_name_from_model_name(model_name),
                    self.device
                )
                updated_model = add_update_to_model(model, aggregated_gradient, weight=1.0, device=self.device)
                save_model(self.exp_name, model_name, updated_model)

        return weight, self.server, baseline_score


# ---------------------------------------------------------
#  HELPER FUNCTIONS
# ---------------------------------------------------------

def clip_gradient_update(grad_update, clip_norm: float):
    """
    Clamp each parameter update to [-grad_clip, grad_clip].
    """
    clipped_updates = []
    for param in grad_update:
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


def flatten(grad_update):
    """
    Concatenate all parameter tensors into a single 1D vector.
    """
    return torch.cat([param.view(-1) for param in grad_update], dim=0)


def unflatten(flattened: torch.Tensor, normal_shape):
    """
    Reshape a flat vector back into the original param shapes in 'normal_shape'.
    'normal_shape' must be a list of Tensors (or nn.Parameter) from which we
    know the dimension sizes.
    """
    grad_update = []
    current_pos = 0
    for param in normal_shape:
        num_params = param.numel()
        slice_ = flattened[current_pos:current_pos + num_params]
        grad_update.append(slice_.reshape(param.shape))
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
    """
    assert len(grad_accumulator) == len(grad_update), "Mismatch in grad lists."
    for acc, g in zip(grad_accumulator, grad_update):
        acc.data += g.data * weight


def cosine_xy(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Plain PyTorch cosine similarity between two 1D vectors.
    """
    x = x.detach().cpu()
    y = y.detach().cpu()
    cos_val = torch.cosine_similarity(x, y, dim=0)
    return cos_val.item()


def encrypted_cosine_xy(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Example placeholder for homomorphic-encrypted cosine similarity.
    Currently calls `private_model_evaluation(x, y)`.
    """
    return private_model_evaluation(x, y)
