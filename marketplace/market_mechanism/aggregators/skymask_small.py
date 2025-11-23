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

    This function explicitly simulates a small root dataset (approx 100 samples)
    by limiting the number of updates per epoch and using early stopping.
    This prevents mask saturation when the actual dataset is large.
    """
    masknet = masknet.to(device)
    optimizer = optim.SGD(masknet.parameters(), lr=lr)
    loss_fn = F.nll_loss

    # --- TRICK SETUP ---
    # The paper relies on a small dataset (100 samples).
    # If your batch_size is 32, roughly 3-4 batches = 100 samples.
    # We restrict the loop to this number to prevent over-training.
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
        # Instead of 'for X, y in server_data_loader:', we loop a fixed number of times.
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

        # --- CONVERGENCE CHECK ---
        # Official logic: Stop if loss improvement is < 0.01
        loss_diff = prev_loss - current_loss

        if epoch > 0 and loss_diff < 1e-2:
            logger.info(f"MaskNet Converged at epoch {epoch} (Loss diff {loss_diff:.4f}). Stopping.")
            break

        prev_loss = current_loss

    logger.info("MaskNet Training Finished.")
    return masknet


class SkymaskSmallAggregator(BaseAggregator):
    """
    Implements the SkyMask (Small Dataset Variant) aggregation strategy.

    This aggregator uses a subsampling trick to simulate a small root dataset (100 samples)
    regardless of the actual size of the buyer_data_loader. This prevents mask saturation
    and aligns with the official paper's constraints.
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

        # Here, we use the passed-in root_gradient to construct the buyer's parameters
        buyer_params = [p_glob + p_upd for p_glob, p_upd in zip(global_params, root_gradient)]
        worker_params.append(buyer_params)

        # 2. Determine the correct model type
        if self.sm_model_type == 'None' or self.sm_model_type is None or self.sm_model_type == '':
            sm_model_type = 'flexiblecnn'
            logger.warning(f"sm_model_type not set. Using 'dynamic' which auto-adapts to model architecture.")
        else:
            sm_model_type = self.sm_model_type
            logger.info(f"Using explicitly set sm_model_type: {sm_model_type}")

        # 3. Create and train the MaskNet
        masknet = create_masknet(worker_params, sm_model_type, self.device)

        # Debugging / Structure Analysis (Optional - can be commented out for production)
        # print("\n" + "=" * 80)
        # print("MASKNET STRUCTURE ANALYSIS:")
        # print("=" * 80)
        # for i, (name, module) in enumerate(masknet.named_modules()):
        #     print(f"{i}: {name} -> {type(module).__name__}")
        # print("=" * 80 + "\n")

        if masknet is None:
            raise RuntimeError(
                f"Failed to create masknet with type '{sm_model_type}'. "
                f"Check your model architecture matches the sm_model_type."
            )

        # CALL THE RENAMED TRAINING FUNCTION
        masknet = train_masknet_small(
            masknet,
            self.buyer_data_loader,
            self.mask_epochs,
            self.mask_lr,
            self.mask_clip,
            self.device
        )

        # 4. Extract masks and classify with GMM
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
                seller_masks_np.append(np.array([]))
                continue

            flat_mask = (torch.cat(seller_mask_layers) > t).float()
            seller_masks_np.append(flat_mask.cpu().numpy())

        if len(seller_masks_np) < 2:
            gmm_labels = np.ones(len(seller_masks_np))
        else:
            gmm_labels = GMM2(seller_masks_np)

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
            logger.info(f"Aggregating {num_inliers} inlier updates.")
            for update in inlier_updates:
                for agg_grad, upd_grad in zip(aggregated_gradient, update):
                    agg_grad.add_(upd_grad, alpha=1 / num_inliers)

        aggregation_stats["skymask_num_selected"] = len(selected_sids)
        aggregation_stats["skymask_num_rejected"] = len(outlier_sids)

        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats