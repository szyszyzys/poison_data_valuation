import logging
from typing import Dict, List, Tuple

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


def train_masknet(masknet: nn.Module, server_data_loader, epochs: int, lr: float, grad_clip: float,
                  device: torch.device) -> nn.Module:
    """Helper function to train the SkyMask MaskNet."""
    masknet = masknet.to(device)
    optimizer = optim.SGD(masknet.parameters(), lr=lr)
    loss_fn = F.nll_loss

    logger.info(f"Starting MaskNet Training: Epochs={epochs}, LR={lr}")
    for epoch in range(epochs):
        masknet.train()
        for X, y in server_data_loader:
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
    logger.info("MaskNet Training Finished.")
    return masknet


class SkymaskAggregator(BaseAggregator):
    """
    Implements the SkyMask aggregation strategy.

    This method trains a special neural network (MaskNet) on the server's trusted
    data to learn which parameters of a seller's update are beneficial. It then uses
    these learned "masks" to cluster sellers into benign and malicious groups.
    """

    def __init__(self,
                 clip: bool,
                 sm_model_type: str,
                 mask_epochs: int,
                 mask_lr: float,
                 mask_clip: float,
                 mask_threshold: float,
                 *args, **kwargs):

        # Pass all common arguments (global_model, device, etc.) up to the BaseAggregator
        super().__init__(*args, **kwargs)

        # Handle the specific parameters for Skymask
        self.clip = clip
        self.sm_model_type = sm_model_type
        self.mask_epochs = mask_epochs
        self.mask_lr = mask_lr
        self.mask_clip = mask_clip
        self.mask_threshold = mask_threshold
        logger.info(f"SkymaskAggregator initialized with mask_epochs={self.mask_epochs}, mask_lr={self.mask_lr}")

    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]], **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str]]:
        logger.info(f"--- SkyMask Aggregation (Epoch {global_epoch}) ---")

        # 1. Compute full model parameters for each seller and the buyer
        global_params = [p.data.clone() for p in self.global_model.parameters()]
        trust_gradient = self._compute_trust_gradient()

        worker_params = []
        seller_ids = list(seller_updates.keys())

        # Store processed (clipped) updates to avoid doing it twice
        processed_updates = {}
        for sid in seller_ids:
            update = clip_gradient_update(seller_updates[sid], self.clip_norm) if self.clip else seller_updates[sid]
            processed_updates[sid] = update
            worker_params.append([p_glob + p_upd for p_glob, p_upd in zip(global_params, update)])

        buyer_params = [p_glob + p_upd for p_glob, p_upd in zip(global_params, trust_gradient)]
        worker_params.append(buyer_params)  # Buyer is the last model

        # 2. Create and train the MaskNet
        masknet = create_masknet(worker_params, self.sm_model_type, self.device)
        masknet = train_masknet(masknet, self.buyer_data_loader, self.mask_epochs, self.mask_lr, self.mask_clip,
                                self.device)

        # 3. Extract masks and classify with GMM
        seller_masks_np = []
        t = torch.tensor([self.mask_threshold], device=self.device)
        for i in range(len(seller_ids)):  # Iterate over sellers only
            seller_mask_layers = []
            for layer in masknet.children():
                if isinstance(layer, (myconv2d, mylinear)):
                    if hasattr(layer, 'weight_mask'):
                        seller_mask_layers.append(torch.flatten(torch.sigmoid(layer.weight_mask[i].data)))
                    if hasattr(layer, 'bias_mask') and layer.bias_mask is not None:
                        seller_mask_layers.append(torch.flatten(torch.sigmoid(layer.bias_mask[i].data)))

            if not seller_mask_layers:
                logger.warning(f"No mask layers found for seller index {i}. Appending empty mask.")
                seller_masks_np.append(np.array([]))
                continue

            flat_mask = (torch.cat(seller_mask_layers) > t).float()
            seller_masks_np.append(flat_mask.cpu().numpy())

        if len(seller_masks_np) < 2:
            gmm_labels = [1] * len(seller_masks_np)  # Trust all if less than 2
        else:
            gmm_labels = GMM2(seller_masks_np)  # Assumes 1 for inlier, 0 for outlier

        # 4. Aggregate using inliers
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        selected_sids, outlier_sids = [], []
        inlier_updates = []

        for i, sid in enumerate(seller_ids):
            if i < len(gmm_labels) and gmm_labels[i] == 1:
                selected_sids.append(sid)
                # Reuse the already processed update
                inlier_updates.append(processed_updates[sid])
            else:
                outlier_sids.append(sid)

        if inlier_updates:
            num_inliers = len(inlier_updates)
            logger.info(f"Aggregating {num_inliers} inlier updates.")
            for update in inlier_updates:
                for agg_grad, upd_grad in zip(aggregated_gradient, update):
                    agg_grad.add_(upd_grad, alpha=1 / num_inliers)

        return aggregated_gradient, selected_sids, outlier_sids
