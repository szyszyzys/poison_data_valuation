from typing import Dict

import numpy as np
import torch
from torch import nn

from model.utils import get_model, load_model


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
                 n_adversaries: int,
                 dataset_name: str,
                 model_structure: nn.Module = None,
                 quantization: bool = False,
                 device=None):
        """
        :param save_path:         Name/identifier of the current experiment.
        :param n_seller:   Total number of participant clients.
        :param n_adversaries:    Number of adversarial clients (not always used).
        :param model_structure:  A torch.nn.Module structure (uninitialized) used to get param shapes.
        :param quantization:     Whether to do quantized aggregation.
        :param device:           Torch device (CPU/GPU) to run computations.
        """
        self.save_path = save_path
        self.n_seller = n_seller
        self.n_adversaries = n_adversaries
        self.model_structure = model_structure
        self.device = device
        self.quantization = quantization
        self.global_model = get_model(dataset_name)
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
            compute_update_gradients(self.save_path, old_m, new_m, self.device)
            for old_m, new_m in zip(self.backup_models, self.client_models)
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

    def aggregate(self, global_epoch, seller_updates, buyer_updates, method="martfl"):
        return self.martFL(global_epoch=global_epoch, seller_updates=seller_updates, buyer_updates=buyer_updates)

    # ---------------------------
    # Main Federated Aggregation (martFL)
    # ---------------------------
    def martFL(self,
               global_epoch: int,
               seller_updates,
               buyer_updates,
               change_base: bool = False,
               ground_truth_model=None):
        """
        Performs the martFL aggregation:
          1) Compute each client's flattened & clipped gradient update.
          2) Compute the cosine similarity of each client’s update to the server’s update.
          3) Cluster the cosine similarities to identify outliers.
          4) Normalize the resulting “non-outlier” scores to derive weights.
          5) For each seller, compute its cosine similarity to a baseline update (from a ground-truth model).
          6) Aggregate the updates using the computed weights.
          7) Update client models accordingly.

        Returns:
           aggregated_gradient: the aggregated update (list of tensors, same structure as model parameters)
           selected_ids: list of indices of sellers whose gradients are selected (non-outliers)
           outlier_ids: list of indices of sellers whose gradients are labeled as outliers
           baseline_similarities: list of cosine similarities of each seller's update to the baseline update.
        """
        print("change_base :", change_base)
        print("global_epoch :", global_epoch)

        # Number of sellers
        self.n_seller = len(seller_updates)

        print("start gradient aggregation")
        aggregated_gradient = [
            torch.zeros_like(param, device=self.device)
            for param in self.model_structure.parameters()
        ]

        print("start flatten and clipping")
        # Process each seller update: clip then flatten
        clients_update_flattened = [
            flatten(clip_gradient_update(update, clip_norm=0.01))
            for update in seller_updates
        ]

        # Process buyer update (baseline)
        baseline_update_flattened = flatten(buyer_updates)

        print("start cosine baseline")
        # Vectorize cosine similarity:
        clients_stack = torch.stack(clients_update_flattened)  # shape: (n_seller, d)
        baseline = baseline_update_flattened.unsqueeze(0)  # shape: (1, d)
        cosine_similarities = torch.nn.functional.cosine_similarity(baseline, clients_stack, dim=1)
        np_cosine_result = cosine_similarities.cpu().numpy()
        np_cosine_result = np.nan_to_num(np_cosine_result, nan=0.0)
        # # 5. Clustering on cosine similarities:
        # diameter = np.max(np_cosine_result) - np.min(np_cosine_result)
        #
        # print("Diameter:", diameter)
        # n_clusters = optimal_k_gap(np_cosine_result)
        # if n_clusters == 1 and diameter > 0.05:
        #     n_clusters = 2
        # clusters, centroids = kmeans(np_cosine_result, n_clusters)
        # print("Centroids:", centroids)
        # print(f"{n_clusters} Clusters:", clusters)
        # center = centroids[-1] if len(centroids) > 0 else 0.0
        #
        # # 6. Secondary clustering for further outlier detection (with k=2)
        # if n_clusters == 1:
        #     clusters_secondary = [1] * self.n_seller
        # else:
        #     clusters_secondary, _ = kmeans(np_cosine_result, 2)
        # print("Secondary Clustering:", clusters_secondary)
        #
        # # 7. Determine border for outlier detection (max distance from center, skipping server)
        # border = 0.0
        # for i, (cos_sim, sec_cluster) in enumerate(zip(np_cosine_result, clusters_secondary)):
        #     if n_clusters == 1 or sec_cluster != 0:
        #         dist = abs(center - cos_sim)
        #         if dist > border:
        #             border = dist
        # print("Border:", border)

        # 8. Mark outliers: Build a list of non_outlier scores.
        non_outliers = [1.0 for _ in range(self.n_seller)]
        # candidate_server = []
        # for i in range(self.n_seller):
        #     if clusters_secondary[i] == 0 or np_cosine_result[i] == 0.0:
        #         non_outliers[i] = 0.0  # mark as outlier
        #     else:
        #         dist = abs(center - np_cosine_result[i])
        #         non_outliers[i] = 1.0 - dist / (border + 1e-6)
        #         if clusters[i] == n_clusters - 1:
        #             candidate_server.append(i)
        #             non_outliers[i] = 1.0  # force inlier for candidate server

        # Identify selected (inlier) and outlier seller indices.
        selected_ids = [i for i in range(self.n_seller) if non_outliers[i] > 0.0]
        outlier_ids = [i for i in range(self.n_seller) if non_outliers[i] == 0.0]

        # 9. Use the non_outlier scores as final weights.
        # (Overwrite cosine_result with non_outliers)
        for i in range(self.n_seller):
            np_cosine_result[i] = non_outliers[i]
        cosine_weight = torch.tensor(np_cosine_result, dtype=torch.float, device=self.device)
        denom = torch.sum(torch.abs(cosine_weight)) + 1e-6
        weight = cosine_weight / denom
        print(f"current_weight: {weight}")
        print(f"similarity: {np_cosine_result}")

        # 10. Compute baseline cosine similarity for each seller.
        # Compute the baseline update from a ground-truth model.
        # ground_truth_updates = compute_ground_truth_updates(
        #     self.save_path, self.backup_models[0], ground_truth_model, self.device
        # )
        # ground_truth_updates_flat = flatten(ground_truth_updates)
        # baseline_similarities = []
        # for i in range(self.n_seller):
        #     bs = cosine_xy(ground_truth_updates_flat, clients_update_flattened[i])
        #     baseline_similarities.append(bs)
        # print("Baseline similarities:", baseline_similarities)

        # 11. Final aggregation: sum weighted gradients.
        # update_gradients = self.get_update_gradients()  # Should return a list of updates (each update is list of tensors)
        # norms = [torch.norm(acc).item() for acc in aggregated_gradient]
        # print(f"Initial, norms: {norms}")
        for idx, (gradient, wt) in enumerate(zip(seller_updates, weight)):
            add_gradient_updates(aggregated_gradient, gradient, weight=wt)
            # Compute and print norm of each aggregated parameter
            # norms = [torch.norm(acc).item() for acc in aggregated_gradient]
            # print(f"After update {idx}, norms: {norms}")

        # def tensors_allclose(list1, list2, atol=1e-6):
        #     return all(torch.allclose(a.cpu(), b.cpu(), atol=atol) for a, b in zip(list1, list2))
        #
        # if tensors_allclose(aggregated_gradient, seller_updates[0]):
        #     print("They are close enough")
        # else:
        #     print("They differ")
        # 12. Update each client model (here using classic update; quantization not implemented)
        # if self.quantization:
        #     raise NotImplementedError("Quantization not implemented.")
        # else:
        #     for i, model_name in enumerate(self.client_models):
        #         model = load_model(, get_backup_name_from_model_name(model_name), self.device)
        #         updated_model = add_update_to_model(model, aggregated_gradient, weight=1.0, device=self.device)
        #         save_model(self.exp_name, model_name, updated_model)

        # Return aggregated gradient, selected seller IDs, outlier seller IDs, and baseline similarities.
        return aggregated_gradient, selected_ids, outlier_ids


# todo


# ---------------------------------------------------------
#  HELPER FUNCTIONS
# ---------------------------------------------------------

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


def flatten(grad_update):
    """
    Flatten a list of gradient updates (tensors) into a single 1D tensor.
    If an element is not a tensor, convert it to one.
    """
    flattened = []
    for param in grad_update:
        # Convert non-tensor elements to a tensor.
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param)
        flattened.append(param.view(-1))
    return torch.cat(flattened, dim=0)


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
