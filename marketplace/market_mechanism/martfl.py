import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors  # Keep for potential fallback
from torch import nn, softmax, optim
from torcheval.metrics.functional import multiclass_f1_score

from entry.gradient_market.skymask import classify
from entry.gradient_market.skymask.models import create_masknet
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
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for i, group in enumerate(optimizer.param_groups):
        for j, param in enumerate(group["params"]):
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train_masknet(masknet: torch.nn.Module,
                  server_data_loader: torch.utils.data.DataLoader,
                  epochs: int,
                  lr: float,
                  grad_clip: float,
                  device: torch.device,
                  optimizer_class: torch.optim.Optimizer = optim.SGD,
                  loss_fn: callable = F.nll_loss,
                  early_stopping_delta: float = 1e-4,  # Min change to consider improvement
                  early_stopping_patience: int = 3,  # How many epochs to wait after min delta not met
                  verbose: bool = True
                  ) -> torch.nn.Module:
    """
    Trains the provided MaskNet using server data.

    Args:
        masknet: The MaskNet nn.Module to train.
        server_data_loader: DataLoader providing batches of (X, y) server data.
        epochs: Maximum number of training epochs.
        lr: Learning rate for the optimizer.
        grad_clip: Gradient clipping threshold.
        device: The torch device ('cuda' or 'cpu').
        optimizer_class: The optimizer class (default: optim.SGD).
        loss_fn: The loss function (default: F.nll_loss).
        early_stopping_delta (float): Minimum change in loss to qualify as improvement for early stopping.
        early_stopping_patience (int): Number of epochs with no improvement to wait before stopping.
        verbose (bool): If True, print training progress per epoch.

    Returns:
        The trained masknet (modified in-place).
    """
    print(f"Starting MaskNet Training: Epochs={epochs}, LR={lr}, Clip={grad_clip}")
    masknet = masknet.to(device)  # Ensure model is on the correct device
    optimizer = optimizer_class(masknet.parameters(), lr=lr)

    best_loss = float('inf')
    epochs_no_improve = 0
    last_epoch_loss = float('inf')  # Track previous epoch's average loss

    for epoch in range(epochs):
        masknet.train()
        total_loss = 0.0
        total_samples = 0
        # Optional: track accuracy
        # total_correct = 0

        for batch_idx, (X, y) in enumerate(server_data_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            output = masknet(X)  # Forward pass

            loss = loss_fn(output, y)  # Calculate loss

            # Backward pass and optimization
            loss.backward()
            clip_gradient(optimizer=optimizer, grad_clip=grad_clip)  # Clip gradients
            optimizer.step()  # Update weights

            total_loss += loss.item() * X.size(0)  # Accumulate loss weighted by batch size
            total_samples += X.size(0)

            # Optional: calculate accuracy
            # _, preds = torch.max(output, dim=1)
            # total_correct += (preds == y).sum().item()

        avg_epoch_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        # avg_epoch_acc = total_correct / total_samples if total_samples > 0 else 0.0

        if verbose:
            print(f"MaskNet Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}")
            # print(f"MaskNet Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_acc:.4f}")

        # Early Stopping Check (based on average epoch loss)
        # Check if loss improved significantly
        if avg_epoch_loss < best_loss - early_stopping_delta:
            best_loss = avg_epoch_loss
            epochs_no_improve = 0  # Reset patience counter
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"MaskNet training stopped early at epoch {epoch + 1} due to lack of improvement.")
            break

        last_epoch_loss = avg_epoch_loss  # Store for next iteration's check (optional)

    print(f"MaskNet Training Finished. Final Avg Loss: {avg_epoch_loss:.4f}")
    return masknet  # Return the trained network


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
                 buyer_data_loader=None,
                 sm_args=None,
                 sm_model_type='cnn'):
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

        self.sm_args = sm_args
        self.sm_model_type = sm_model_type
        if self.sm_args is None:
            # Set defaults for MaskNet params if args is missing
            self.mask_threshold = 0.5
            if sm_model_type == "cnn":
                self.mask_lr_config = 1e7
                self.mask_clip_config = 1e-7
            elif sm_model_type == "resnet20":
                mask_lr = 1e8
                clip_lmt = 1e-7
            elif sm_model_type == "LR":
                mask_lr = 1e7
                clip_lmt = 1e-7
            self.mask_epochs = 20
            self.mask_server_pc_config = 128

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

    def aggregate(self, global_epoch, seller_updates, buyer_updates, remove_baseline, clip=False, server_data=None):
        if self.aggregation_method == "martfl":
            return self.martFL(global_epoch=global_epoch, seller_updates=seller_updates, buyer_updates=buyer_updates,
                               clip=clip, remove_baseline=remove_baseline)
        elif self.aggregation_method == "fedavg":
            return self.fedavg(global_epoch=global_epoch, seller_updates=seller_updates, buyer_updates=buyer_updates)
        elif self.aggregation_method == "skymask":
            return self.skymask(global_epoch=global_epoch, seller_updates=seller_updates, buyer_updates=buyer_updates,
                                clip=clip, server_data_loader=server_data)

        elif self.aggregation_method == "fltrust":
            return self.fltrust(global_epoch=global_epoch, seller_updates=seller_updates, buyer_updates=buyer_updates,
                                clip=clip)
        else:
            raise NotImplementedError(f"current aggregator not implemented {self.aggregation_method}")

    def _calculate_dynamic_eps(self, data: np.ndarray, k: int) -> float:
        """ Calculates dynamic DBSCAN epsilon using the k-NN distance method. (Helper for DBSCAN) """
        # ... (implementation from previous DBSCAN answer) ...
        n_samples = data.shape[0]
        if k >= n_samples: k = n_samples - 1
        if k < 1: k = 1
        if n_samples <= 1 or k == 0: return 0.5
        neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
        neighbors.fit(data.astype(np.float32))
        distances, _ = neighbors.kneighbors(data.astype(np.float32))
        k_distances = np.sort(distances[:, -1])
        chosen_eps = np.median(k_distances)
        # print(f"Dynamically determined eps based on {k}-NN distances (median): {chosen_eps:.4f}")
        return max(chosen_eps, 1e-5)

    def fltrust(self,
                global_epoch: int,
                seller_updates: Dict[str, List[torch.Tensor]],
                buyer_updates: List[torch.Tensor],  # This is the 'baseline' from your code
                clip: bool = True) -> Tuple[List[torch.Tensor], List[int], List[int]]:
        """
        Performs FLTrust aggregation, adapted from the provided logic.
          1) Flattens and optionally clips seller updates.
          2) Flattens the buyer (server/root) update (baseline).
          3) Computes cosine similarity between each seller update and the buyer update.
          4) Computes Trust Scores (ReLU of similarities).
          5) Normalizes Trust Scores to get weights.
          6) Performs weighted aggregation of original (clipped) seller updates.
          7) Scales the final aggregated result by the buyer update norm.

        Returns:
           aggregated_gradient: The aggregated update (list of tensors).
           selected_ids: List of indices of sellers with positive trust scores (weight > 0).
           outlier_ids: List of indices of sellers with zero trust scores (weight == 0).
        """
        print(f"--- Starting FLTrust Aggregation (Epoch {global_epoch}) ---")
        n_seller = len(seller_updates)
        seller_ids = list(seller_updates.keys())  # Store original IDs if needed later

        if n_seller == 0:
            print("Warning: No seller updates received.")
            return ([torch.zeros_like(p, device=self.device) for p in self.model_structure.parameters()], [], [])

        print(f"Processing {n_seller} seller updates.")

        # Initialize aggregated gradient structure (unflattened)
        aggregated_gradient = [
            torch.zeros_like(param, device=self.device)
            for param in self.model_structure.parameters()
        ]

        # 1) Flatten and optionally clip seller updates
        print("Flattening and clipping seller gradients...")
        clients_update_flattened = []
        original_updates_list = list(seller_updates.values())  # Keep original structure for aggregation

        # Store the processed (clipped, device-correct) *unflattened* updates
        processed_updates_unflattened = []

        for i, update in enumerate(original_updates_list):
            # Ensure update tensors are on the correct device first
            update_device = [p.clone().to(self.device) for p in update]
            processed_update = update_device
            if clip:
                processed_update = clip_gradient_update(processed_update, self.clip_norm)

            processed_updates_unflattened.append(processed_update)  # Store unflattened version
            clients_update_flattened.append(flatten(processed_update))  # Flatten for similarity calc

        clients_stack = torch.stack(clients_update_flattened)  # shape: (n_seller, d)

        # 2) Process the buyer (server/root) update (baseline)
        print("Processing buyer (server) update...")
        # Ensure buyer update is on the correct device
        buyer_updates_device = [p.clone().to(self.device) for p in buyer_updates]
        # Optional: Clip the buyer update? Usually not done, but can be added.
        # if clip:
        #      buyer_updates_device = clip_gradient_update(buyer_updates_device, self.clip_norm)
        buyer_update_flattened = flatten(buyer_updates_device)
        buyer_update_norm = torch.norm(buyer_update_flattened, p=2) + 1e-9  # Add epsilon for stability

        print(f"Buyer update norm: {buyer_update_norm.item():.4f}")
        if buyer_update_norm.item() < 1e-8:  # Check against a small threshold
            print("Warning: Buyer update norm is close to zero. FLTrust may yield zero result.")
            # Return zero gradient, all sellers are outliers in this context
            return aggregated_gradient, [], list(range(n_seller))

        # 3) Compute cosine similarity with buyer update (baseline)
        print("Computing cosine similarities with buyer update...")
        # Using the formula: dot(a, b) / (norm(a) * norm(b))
        # Vectorized approach:
        clients_norms = torch.norm(clients_stack, p=2, dim=1) + 1e-9  # Add epsilon for stability
        # Dot product of baseline with each client stack row: baseline @ clients_stack.T
        dot_products = torch.mv(clients_stack, buyer_update_flattened)  # Result shape: (n_seller)

        cosine_similarities = dot_products / (buyer_update_norm * clients_norms)
        cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)  # Clamp for numerical stability

        print(f"Cosine Similarities: {cosine_similarities.cpu().numpy()}")

        # 4) Compute Trust Scores (ReLU of similarities)
        # Equivalent to torch.maximum(cos_sim, torch.tensor(0.0, device=self.device))
        trust_scores = torch.relu(cosine_similarities)
        print(f"Trust Scores (ReLU): {trust_scores.cpu().numpy()}")

        # 5) Normalize Trust Scores to get weights
        total_trust = torch.sum(trust_scores) + 1e-9  # Add epsilon for stability
        print(f"Total Trust Score: {total_trust.item():.4f}")

        if total_trust.item() < 1e-8:  # Check against a small threshold
            print("Warning: Total trust score is close to zero. Aggregation will result in zero.")
            weights = torch.zeros_like(trust_scores)
            selected_ids = []
            outlier_ids = list(range(n_seller))
        else:
            weights = trust_scores / total_trust
            # Consider seller 'i' selected if their weight is significantly > 0
            selected_ids = [i for i, w in enumerate(weights) if w > 1e-9]
            outlier_ids = [i for i, w in enumerate(weights) if w <= 1e-9]

        print(f"Aggregation Weights: {weights.cpu().numpy()}")
        print(f"Selected seller indices ({len(selected_ids)}): {selected_ids}")
        print(f"Outlier seller indices ({len(outlier_ids)}): {outlier_ids}")

        # 6) Perform weighted aggregation using the *original* (clipped) seller updates
        #    This uses the standard FLTrust interpretation: aggregate first.
        temp_aggregated_gradient = [
            torch.zeros_like(param, device=self.device)
            for param in self.model_structure.parameters()
        ]
        if selected_ids:  # Only aggregate if there are selected clients
            print(f"Aggregating updates from {len(selected_ids)} selected sellers...")
            for idx in selected_ids:
                weight = weights[idx]
                # Use the processed (clipped, device-correct) *unflattened* updates
                add_gradient_updates(temp_aggregated_gradient, processed_updates_unflattened[idx], weight=weight)

            # 7) Scale the final aggregated gradient by the norm of the buyer update (baseline)
            scaling_factor = buyer_update_norm.item()  # Use the calculated norm
            print(f"Scaling final aggregated gradient by buyer norm: {scaling_factor:.4f}")
            aggregated_gradient = [param.data * scaling_factor for param in temp_aggregated_gradient]

            # --- Note on Alternative Aggregation from your code ---
            # Your original code normalized each client update *before* summing:
            # new_param_list = []
            # for i in range(n):
            #    norm_i = torch.norm(param_list[i]) + 1e-9
            #    new_param_list.append(param_list[i] * normalized_weights[i] / norm_i * torch.norm(baseline))
            # global_update = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)
            # If you specifically need this behavior, the aggregation loop (step 6 & 7) would change.
            # Let me know if that specific formula is required.
            # --- End Note ---

        else:
            # If no clients selected (all trust scores <= 0), aggregated_gradient remains zeros.
            print("No clients selected based on trust scores. Returning zero gradient.")

        print("--- FLTrust Aggregation Finished ---")
        # Return the UNFLATTENED aggregated gradient, selected indices, outlier indices
        return aggregated_gradient, selected_ids, outlier_ids

    # ---------------------------
    def martFL(self,
               global_epoch: int,
               seller_updates,
               buyer_updates,
               ground_truth_model=None, clip=False, remove_baseline=False):
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
        print(f"Global epoch: {global_epoch}, Change baseline: {self.change_base}, if clip: {clip}")

        # Number of sellers
        self.n_seller = len(seller_updates)
        seller_ids = list(seller_updates.keys())
        seller_id_to_index = {seller_id: index for index, seller_id in enumerate(seller_ids)}

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
        if remove_baseline and self.baseline_id is not None:
            np_cosine_result[seller_id_to_index[self.baseline_id]] = 0.0
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
        if remove_baseline and self.baseline_id is not None:
            clusters[seller_id_to_index[self.baseline_id]] = 0

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
        if remove_baseline and self.baseline_id is not None:
            clusters_secondary[seller_id_to_index[self.baseline_id]] = 0

        # Determine border for outlier detection (max distance from center)
        border = 0.0
        for i, (cos_sim, sec_cluster) in enumerate(zip(np_cosine_result, clusters_secondary)):
            if remove_baseline and self.baseline_id is not None:
                if i == seller_id_to_index[self.baseline_id]:
                    continue
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
            if remove_baseline and self.baseline_id is not None:
                if i == seller_id_to_index[self.baseline_id]:
                    non_outliers[i] = 0.0
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
        print(f"Current base in outlier: {self.baseline_id in outlier_sellers}")

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

    def skymask(self,
                global_epoch: int,  # Corresponds to niter
                seller_updates: Dict[str, List[torch.Tensor]],
                buyer_updates: List[torch.Tensor],  # The server update / baseline
                server_data_loader: torch.utils.data.DataLoader,  # CORRECTED TYPE HINT
                clip: bool = False  # Whether to clip seller_updates first
                ) -> Tuple[List[torch.Tensor], List[int], List[int]]:
        """
        Integrates the original SkyMask function logic (MaskNet+GMM).
        Creates and trains a new MaskNet instance each time it's called.

        Args:
            global_epoch (int): Current global epoch (used as niter).
            seller_updates (Dict): Dict of seller updates (unflattened).
            buyer_updates (List): Server/baseline update (unflattened).
            server_data_loader (DataLoader): DataLoader for server data/labels
                                              to train masknet.
            clip (bool): Apply gradient clipping to seller_updates before flattening.

        Returns:
           Tuple[List[torch.Tensor], List[int], List[int]]:
             - aggregated_gradient: The computed aggregated gradient (unflattened).
             - selected_ids: Indices of sellers selected by GMM.
             - outlier_ids: Indices of sellers filtered out by GMM.
        """
        print(f"--- Starting SkyMask (Original Logic - Recreate MaskNet) Aggregation (Epoch {global_epoch}) ---")
        net = self.model_structure  # Use the stored model structure instance
        args = self.args  # Get args from self if needed for create_masknet/logging

        # 1. Prepare inputs
        print("Preparing inputs...")
        processed_seller_updates = {}
        if clip:
            print(f"Clipping seller updates with norm {self.clip_norm}")
            for sid, update in seller_updates.items():
                # Ensure update tensors are on the correct device before clipping
                update_on_device = [p.to(self.device) for p in update]
                processed_seller_updates[sid] = clip_gradient_update(update_on_device, self.clip_norm)
        else:
            # Still ensure they are on the device for flattening
            processed_seller_updates = {sid: [p.to(self.device) for p in update] for sid, update in
                                        seller_updates.items()}

        flat_seller_updates = [flatten(upd) for upd in processed_seller_updates.values()]
        flat_buyer_update = flatten([p.to(self.device) for p in buyer_updates])  # Ensure baseline is on device

        grad_list = flat_seller_updates + [flat_buyer_update]
        n_grad = len(grad_list)
        n_seller = n_grad - 1
        if n_seller <= 0:
            print("Warning: No seller updates to process after preparation.")
            return ([torch.zeros_like(p, device=self.device) for p in net.parameters()], [], [])

        print(f"Prepared grad_list with {n_seller} seller updates + 1 baseline.")
        datalist = [p.data.clone() for p in net.parameters()]  # Structure for unflattening

        # 2. Create a *new* MaskNet instance for this round
        #    Using configured network type from args if available.
        masknet_type = getattr(args, 'net', 'cnn') if args else 'cnn'
        print(f"Creating new masknet (type: {masknet_type})...")
        # Make sure create_masknet handles the datalist structure correctly
        masknet = create_masknet(datalist, masknet_type, self.device)

        # 3. Train the newly created MaskNet instance
        print("Executing core SkyMask logic (training MaskNet)...")
        if server_data_loader is None:
            raise ValueError("server_data_loader is required for MaskNet training but was not provided.")

        # Get training hyperparams from self (set during __init__)
        mask_lr = self.mask_lr_config
        clip_lmt = self.mask_clip_config
        epochs = self.mask_epochs
        print(f"MaskNet Training Params: LR={mask_lr}, Clip={clip_lmt}, Epochs={epochs}")

        # === Call the training function on the local masknet ===
        # train_masknet modifies the passed masknet in-place and returns it
        masknet = train_masknet(
            masknet=masknet,  # Pass the newly created masknet
            server_data_loader=server_data_loader,
            epochs=epochs,
            lr=mask_lr,
            grad_clip=clip_lmt,
            device=self.device,
        )
        # ========================================================

        # 4. Extract masks from the *trained local* masknet instance
        print("Extracting masks...")
        # Ensure masknet has parameters after training
        masknet_trained_params = list(masknet.parameters())  # Get parameters from the trained instance
        if not masknet_trained_params:
            # Check if any parameter has requires_grad=True, maybe training failed silently
            if not any(p.requires_grad for p in masknet.parameters()):
                print("Warning: MaskNet has no parameters requiring gradients.")
            raise RuntimeError("MaskNet has no parameters after training, cannot extract masks.")

        # Extract data part of parameters
        masknet_trained_param_data = [p.data for p in masknet_trained_params]

        # --- Calculate size_per_seller based on the *actual trained parameters* ---
        total_params_in_masknet = len(masknet_trained_param_data)
        if total_params_in_masknet == 0:  # Should be caught above, but double check
            raise ValueError("Masknet parameter list is empty after training.")

        # Assumption: MaskNet params are structured per seller (interleaved)
        if total_params_in_masknet % n_seller != 0:
            print(
                f"Warning: Trained MaskNet param count ({total_params_in_masknet}) not cleanly divisible by n_seller ({n_seller}). Mask extraction logic might be incorrect.")
            # Fallback guess, might be wrong:
            size_per_seller = total_params_in_masknet // n_seller if total_params_in_masknet >= n_seller else 0
            if size_per_seller == 0: raise ValueError("Cannot determine mask structure (param count < n_seller).")
        else:
            size_per_seller = total_params_in_masknet // n_seller

        print(f"Extracting masks assuming {size_per_seller} param groups per seller.")
        mask_list_np = []
        t = torch.Tensor([self.mask_threshold]).to(self.device)

        for i in range(n_seller):
            mask = []
            for j in range(size_per_seller):
                param_index = i + j * n_seller
                if param_index >= total_params_in_masknet:
                    print(
                        f"Warning: Skipping mask param index {param_index} (out of bounds: {total_params_in_masknet}).")
                    continue
                mask_tensor = masknet_trained_param_data[param_index]
                mask.append(torch.sigmoid(torch.flatten(mask_tensor, start_dim=0, end_dim=-1)))
            if not mask:
                print(f"Warning: No mask parameters extracted for seller {i}.")
                mask_list_np.append(np.array([]))  # Append empty if no params found
                continue
            full_client_mask = torch.cat(mask)
            out = (full_client_mask > t).float() * 1
            mask_list_np.append(out.detach().cpu().numpy())

        # 5. GMM Clustering
        print("Performing GMM clustering...")
        # Ensure masks are not all empty before passing to GMM
        valid_masks = [m for m in mask_list_np if m.size > 0]
        if not valid_masks:
            print("Warning: No valid masks extracted. Cannot perform GMM. Assuming all sellers are outliers.")
            res = [0] * n_seller  # All outliers
        elif len(valid_masks) < n_seller:
            print(f"Warning: Only {len(valid_masks)}/{n_seller} sellers have valid masks for GMM.")
            # Perform GMM on valid ones, mark others as outliers? Or handle differently?
            # Simple approach: Assume missing mask means outlier
            gmm_res_valid = classify.GMM2(valid_masks)
            res = []
            valid_idx = 0
            original_indices_with_valid_masks = [i for i, m in enumerate(mask_list_np) if m.size > 0]
            map_valid_to_gmm = {orig_idx: gmm_idx for gmm_idx, orig_idx in enumerate(original_indices_with_valid_masks)}

            for i in range(n_seller):
                if i in map_valid_to_gmm:
                    if map_valid_to_gmm[i] < len(gmm_res_valid):
                        res.append(gmm_res_valid[map_valid_to_gmm[i]])
                    else:
                        print(f"Warning: GMM result index mismatch for seller {i}. Marking as outlier.")
                        res.append(0)  # GMM result too short? Mark outlier
                else:
                    res.append(0)  # No valid mask, mark as outlier
            if len(res) != n_seller:  # Final sanity check
                print("Error: Final result length mismatch after GMM. Defaulting to all outliers.")
                res = [0] * n_seller
        else:
            # Normal case: all masks were valid
            res = classify.GMM2(valid_masks)  # Get results for N sellers
            if len(res) != n_seller:
                print(f"Warning: GMM returned {len(res)} results, but expected {n_seller}. Padding with outliers.")
                res = list(res) + [0] * (n_seller - len(res))

        # 6. Determine selected/outliers and Aggregate
        selected_ids = [i for i, label in enumerate(res) if label == 1]
        outlier_ids = [i for i, label in enumerate(res) if label == 0]
        num_selected = len(selected_ids)

        print(f"GMM Results (0=outlier, 1=inlier): {res}")
        print(f"Selected seller indices ({num_selected}): {selected_ids}")
        print(f"Outlier seller indices ({len(outlier_ids)}): {outlier_ids}")

        # Perform aggregation
        aggregated_gradient_flat = torch.zeros_like(grad_list[0])  # Shape like first seller grad
        if num_selected > 0:
            weights = 1.0 / num_selected
            print(f"Aggregating {num_selected} updates with weight {weights:.4f}")
            # Stack selected flattened gradients for efficient sum
            selected_grads_stacked = torch.stack([grad_list[i] for i in selected_ids], dim=0)
            aggregated_gradient_flat = torch.sum(selected_grads_stacked, dim=0) * weights
        else:
            print("Warning: No clients selected by GMM. Aggregated gradient is zero.")

        # 7. Unflatten the aggregated gradient
        print("Unflattening aggregated gradient...")
        # Use the correct helper function name
        aggregated_gradient_unflattened = unflatten(aggregated_gradient_flat, datalist)

        print("--- SkyMask (Original Logic - Recreate MaskNet) Aggregation Finished ---")
        return aggregated_gradient_unflattened, selected_ids, outlier_ids


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
