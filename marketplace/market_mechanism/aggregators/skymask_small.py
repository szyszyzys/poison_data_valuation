import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from common.utils import clip_gradient_update
from marketplace.market_mechanism.aggregators.base_aggregator import BaseAggregator
from marketplace.market_mechanism.aggregators.skymask_utils.classify import GMM2
from marketplace.market_mechanism.aggregators.skymask_utils.models import create_masknet
from marketplace.market_mechanism.aggregators.skymask_utils.mytorch import myconv2d, mylinear

logger = logging.getLogger(__name__)


def train_masknet_small(masknet: nn.Module, server_data_loader, epochs: int, lr: float, grad_clip: float,
                        device: torch.device) -> nn.Module:
    """
    Helper function to train the SkyMask MaskNet for the 'Small Dataset' variant.

    FIXED: Convergence check is now performed every 10 epochs to align with official logic
    and prevent premature stopping due to small-batch noise.
    """
    masknet = masknet.to(device)
    optimizer = optim.SGD(masknet.parameters(), lr=lr)
    loss_fn = F.nll_loss

    UPDATES_PER_EPOCH = 4

    # Track loss for official convergence check
    prev_loss = 1e4

    logger.info(f"Starting MaskNet (Small) Training: Epochs={epochs}, LR={lr}, Updates/Epoch={UPDATES_PER_EPOCH}")

    # Create an iterator so we can pull batches manually
    data_iter = iter(server_data_loader)

    for epoch in range(epochs):
        masknet.train()

        current_loss = 0.0

        # --- THE TRICK LOOP ---
        for _ in range(UPDATES_PER_EPOCH):
            try:
                X, y = next(data_iter)
            except StopIteration:
                # If we hit the end of the large dataset, restart
                data_iter = iter(server_data_loader)
                X, y = next(data_iter)

            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = masknet(X)
            loss = loss_fn(output, y)
            loss.backward()

            # Manual gradient clipping
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
            optimizer.step()

            current_loss = loss.item()

        if epoch % 10 == 0:
            loss_diff = prev_loss - current_loss

            # Only stop if we have trained at least 10 epochs AND improvement is small
            if epoch > 0 and loss_diff < 1e-2:
                logger.info(f"MaskNet Converged at epoch {epoch} (Loss diff {loss_diff:.4f}). Stopping.")
                break

            # Update prev_loss only at the check points
            prev_loss = current_loss

    logger.info("MaskNet Training Finished.")
    return masknet

class SkymaskSmallAggregator(BaseAggregator):
    """
    Implements the SkyMask (Small Dataset Variant) aggregation strategy.
    Fixed to handle PCA/GMM convergence errors when masks are identical.
    """

    def __init__(self,
                 clip: bool,
                 sm_model_type: str,
                 mask_epochs: int,
                 mask_lr: float,
                 mask_clip: float,
                 mask_threshold: float,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.clip = clip
        self.sm_model_type = sm_model_type
        self.mask_epochs = mask_epochs
        self.mask_lr = mask_lr
        self.mask_clip = mask_clip
        self.mask_threshold = mask_threshold
        logger.info(f"SkymaskSmallAggregator initialized with mask_epochs={self.mask_epochs}, mask_lr={self.mask_lr}")

    def aggregate(
            self,
            global_epoch: int,
            seller_updates: Dict[str, List[torch.Tensor]],
            root_gradient: List[torch.Tensor],
            **kwargs
    ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"--- SkyMask (Small) Aggregation (Epoch {global_epoch}) ---")

        # 1. Compute full model parameters
        global_params = [p.data.clone() for p in self.global_model.parameters()]
        worker_params = []
        seller_ids = list(seller_updates.keys())
        processed_updates = {}

        for sid in seller_ids:
            update = clip_gradient_update(seller_updates[sid], self.clip_norm) if self.clip else seller_updates[sid]
            processed_updates[sid] = update
            worker_params.append([p_glob + p_upd for p_glob, p_upd in zip(global_params, update)])

        # Buyer parameters (using root_gradient)
        buyer_params = [p_glob + p_upd for p_glob, p_upd in zip(global_params, root_gradient)]
        worker_params.append(buyer_params)

        # 2. Determine the correct model type
        if self.sm_model_type == 'None' or self.sm_model_type is None or self.sm_model_type == '':
            sm_model_type = 'flexiblecnn'
        else:
            sm_model_type = self.sm_model_type

        # 3. Create and train the MaskNet
        masknet = create_masknet(worker_params, sm_model_type, self.device)

        if masknet is None:
            raise RuntimeError(f"Failed to create masknet with type '{sm_model_type}'.")

        masknet = train_masknet_small(
            masknet,
            self.buyer_data_loader,
            self.mask_epochs,
            self.mask_lr,
            self.mask_clip,
            self.device
        )

        # 4. Extract masks
        seller_masks_np = []
        t = torch.tensor([self.mask_threshold], device=self.device)

        for i in range(len(seller_ids)):
            seller_mask_layers = []
            for layer in masknet.modules():
                if isinstance(layer, (myconv2d, mylinear)):
                    if hasattr(layer, 'weight_mask'):
                        seller_mask_layers.append(torch.flatten(torch.sigmoid(layer.weight_mask[i].data)))
                    if hasattr(layer, 'bias_mask') and layer.bias_mask is not None:
                        seller_mask_layers.append(torch.flatten(torch.sigmoid(layer.bias_mask[i].data)))

            if not seller_mask_layers:
                # Fallback for empty layers
                seller_masks_np.append(np.zeros(10))
                continue

            # Apply threshold to make binary
            flat_mask = (torch.cat(seller_mask_layers) > t).float()
            seller_masks_np.append(flat_mask.cpu().numpy())

        # Stack to create matrix (Sellers x Features)
        masks_matrix = np.array(seller_masks_np)

        # --- FIX START: ROBUST CLUSTERING ---

        # 1. Handle NaNs (replace with 0 to prevent crash)
        if np.isnan(masks_matrix).any():
            logger.warning("NaNs detected in seller masks. Replacing with 0.")
            masks_matrix = np.nan_to_num(masks_matrix, nan=0.0)

        # 2. Check for duplicate masks or insufficient data
        # If all sellers have the exact same mask, GMM will fail to find 2 clusters.
        # We calculate unique rows to see if we have diversity.
        unique_masks = np.unique(masks_matrix, axis=0)

        num_sellers = len(seller_ids)
        distinct_clusters_possible = len(unique_masks) >= 2

        if num_sellers < 2 or not distinct_clusters_possible:
            logger.warning(f"Distinct masks ({len(unique_masks)}) < 2. Skipping GMM, assuming all benign.")
            gmm_labels = np.ones(num_sellers)
        else:
            try:
                # Only run GMM if we have diverse data
                gmm_labels = GMM2(masks_matrix)
            except Exception as e:
                logger.error(f"GMM2 clustering failed: {e}. Defaulting to all benign.")
                gmm_labels = np.ones(num_sellers)

        # --- FIX END ---

        # 5. Aggregate using inliers
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        selected_sids, outlier_sids = [], []
        inlier_updates = []

        aggregation_stats = {}

        for i, sid in enumerate(seller_ids):
            is_inlier = i < len(gmm_labels) and gmm_labels[i] == 1
            aggregation_stats[f"skymask_gmm_label_{sid}"] = int(is_inlier)

            if is_inlier:
                selected_sids.append(sid)
                inlier_updates.append(processed_updates[sid])
            else:
                outlier_sids.append(sid)

        if inlier_updates:
            num_inliers = len(inlier_updates)
            # logger.info(f"Aggregating {num_inliers} inlier updates.")
            for update in inlier_updates:
                for agg_grad, upd_grad in zip(aggregated_gradient, update):
                    agg_grad.add_(upd_grad, alpha=1 / num_inliers)

        aggregation_stats["skymask_num_selected"] = len(selected_sids)
        aggregation_stats["skymask_num_rejected"] = len(outlier_sids)

        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats