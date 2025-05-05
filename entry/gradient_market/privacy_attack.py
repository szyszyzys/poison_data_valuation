import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
# ... (other necessary imports: numpy, time, copy, logging, warnings, pandas, Path, typing, skimage, matplotlib)
from collections import OrderedDict
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from typing import List, Tuple, Dict, Optional, Type, Union, Any


def gradient_inversion_attack(target_gradient, model, input_shape, num_classes, device, num_images=1, iterations=500,
                              lr=0.1, label_type='optimize', ground_truth_labels=None, **kwargs):
    optimizer_class = kwargs.get('optimizer_class', optim.Adam)
    loss_type = kwargs.get('loss_type', 'l2')
    init_type = kwargs.get('init_type', 'gaussian')
    log_interval = kwargs.get('log_interval')
    regularization_weight = kwargs.get('regularization_weight', 0.001)
    return_best = kwargs.get('return_best', True)
    model.eval()
    model = model.to(device)
    target_gradient = [g.to(device) for g in target_gradient]
    if label_type == 'ground_truth':
        if ground_truth_labels is None or len(ground_truth_labels) != num_images: raise ValueError(
            "GT labels needed for label_type='ground_truth'")
        fixed_labels = ground_truth_labels.to(device)
    elif label_type == 'random':
        fixed_labels = torch.randint(0, num_classes, (num_images,), device=device)
    elif label_type != 'optimize':
        raise ValueError(f"Unknown label_type: {label_type}")
    if init_type == 'gaussian':
        dummy_data = torch.randn(num_images, *input_shape, device=device, requires_grad=True)
    elif init_type == 'random':
        dummy_data = torch.rand(num_images, *input_shape, device=device, requires_grad=True)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    if label_type == 'optimize':
        dummy_labels_param = torch.randn(num_images, num_classes, device=device,
                                         requires_grad=True)
        params_to_optimize = [dummy_data, dummy_labels_param]
    else:
        params_to_optimize = [dummy_data]
    optimizer = optimizer_class(params_to_optimize, lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    best_dummy_data = dummy_data.detach().clone()
    best_dummy_labels = None
    if log_interval is not None: logging.info(f"Starting GIA ({iterations} iter, lr={lr}, labels='{label_type}')...")
    start_time = time.time()
    for it in range(iterations):
        optimizer.zero_grad()
        model.zero_grad()
        if label_type == 'optimize':
            current_labels_prob = F.softmax(dummy_labels_param, dim=-1)
            current_labels_idx = torch.argmax(
                current_labels_prob, dim=-1)
            output = model(dummy_data)
            task_loss = criterion(output,
                                  current_labels_idx)
        else:
            current_labels_idx = fixed_labels
            output = model(dummy_data)
            task_loss = criterion(output,
                                  current_labels_idx)
        dummy_gradient = torch.autograd.grad(task_loss, model.parameters(), create_graph=False)
        grad_loss = 0
        if loss_type == 'l2':
            grad_loss = sum(
                ((dummy_g - target_g) ** 2).sum() for dummy_g, target_g in zip(dummy_gradient, target_gradient))
        elif loss_type == 'cosine':
            flat_dummy = torch.cat([g.reshape(-1) for g in dummy_gradient])
            flat_target = torch.cat(
                [g.reshape(-1) for g in target_gradient])
            grad_loss = 1 - F.cosine_similarity(flat_dummy, flat_target,
                                                dim=0, eps=1e-8)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        tv_loss = 0
        if regularization_weight > 0: diff1 = dummy_data[:, :, :, :-1] - dummy_data[:, :, :, 1:]
        diff2 = dummy_data[:, :, :-1, :] - dummy_data[:, :, 1:, :]
        tv_loss = torch.norm(diff1) + torch.norm(diff2)
        total_loss = grad_loss + regularization_weight * tv_loss
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            dummy_data.clamp_(0, 1)
        current_loss = grad_loss.item()
        if return_best and current_loss < best_loss: best_loss = current_loss
        best_dummy_data = dummy_data.detach().clone()
        if label_type == 'optimize':
            best_dummy_labels = torch.argmax(F.softmax(dummy_labels_param.detach(), dim=-1), dim=-1)
        else:
            best_dummy_labels = current_labels_idx
        if log_interval is not None and (it % log_interval == 0 or it == iterations - 1): logging.info(
            f"  Iter: {it:5d}/{iterations} | Grad Loss: {current_loss:.4f} | TV Loss: {tv_loss.item() if isinstance(tv_loss, torch.Tensor) else tv_loss:.2f} | Best Loss: {best_loss:.4f}")
    end_time = time.time()
    if log_interval is not None: logging.info(
        f"Attack finished in {end_time - start_time:.2f}s. Final Grad Loss: {current_loss:.4f}, Best Grad Loss: {best_loss:.4f}")
    if return_best:
        final_labels = best_dummy_labels if best_dummy_labels is not None else torch.argmax(
            F.softmax(dummy_labels_param.detach(), dim=-1),
            dim=-1) if label_type == 'optimize' else fixed_labels
        final_images = best_dummy_data
    else:
        final_images = dummy_data.detach().clone()
        final_labels = torch.argmax(
            F.softmax(dummy_labels_param.detach(), dim=-1), dim=-1) if label_type == 'optimize' else fixed_labels
    return final_images, final_labels


def calculate_metrics(img1, img2, data_range=1.0):
    if img1.shape != img2.shape or img1.ndim != 4: raise ValueError(
        "Input images must be 4D tensors with the same shape.")
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    mse_val = np.mean((img1_np - img2_np) ** 2)
    psnr_vals = []
    ssim_vals = []
    num_images = img1_np.shape[0]
    num_channels = img1_np.shape[1]
    for i in range(num_images):
        im1_single = img1_np[i]
        im2_single = img2_np[i]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_psnr = compare_psnr(im1_single, im2_single,
                                            data_range=data_range)
            psnr_vals.append(current_psnr)
            if num_channels == 1:
                im1_ssim = np.squeeze(im1_single)
                im2_ssim = np.squeeze(im2_single)
                channel_axis = None
            else:
                im1_ssim = np.transpose(im1_single, (1, 2, 0))
                im2_ssim = np.transpose(im2_single,
                                        (1, 2, 0))
                channel_axis = -1
            win_size = min(7, im1_ssim.shape[0], im1_ssim.shape[1])
            win_size -= (1 - win_size % 2)
            if win_size >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    current_ssim = compare_ssim(im1_ssim, im2_ssim,
                                                data_range=data_range,
                                                channel_axis=channel_axis,
                                                win_size=win_size,
                                                gaussian_weights=True,
                                                use_sample_covariance=False)
                ssim_vals.append(current_ssim)
            else:
                ssim_vals.append(np.nan)
        except Exception as e:
            logging.warning(f"Error calculating PSNR/SSIM for image {i}: {e}")
            psnr_vals.append(
                np.nan)
            ssim_vals.append(np.nan)
    psnr_val = np.nanmean(psnr_vals) if psnr_vals else np.nan
    ssim_val = np.nanmean(ssim_vals) if ssim_vals else np.nan
    return {"mse": float(mse_val), "psnr": float(psnr_val), "ssim": float(ssim_val)}


def evaluate_inversion(reconstructed_images, ground_truth_images, reconstructed_labels=None, ground_truth_labels=None):
    results = {}
    image_metrics = calculate_metrics(reconstructed_images, ground_truth_images)
    results.update(image_metrics)
    label_acc = np.nan
    if reconstructed_labels is not None and ground_truth_labels is not None:
        if reconstructed_labels.shape == ground_truth_labels.shape:
            correct = (
                    reconstructed_labels.cpu() == ground_truth_labels.cpu()).sum().item()
            total = ground_truth_labels.numel()
            label_acc = correct / total if total > 0 else np.nan
        else:
            logging.warning(
                f"Label shape mismatch ({reconstructed_labels.shape} vs {ground_truth_labels.shape}). Skipping label accuracy.")
    results["label_acc"] = float(label_acc)
    return results


def visualize_and_save_attack(reconstructed_images, ground_truth_images, reconstructed_labels, ground_truth_labels,
                              metrics, save_path, max_images=8, figsize_scale=2.0):
    num_images_to_show = min(len(ground_truth_images), max_images)
    if num_images_to_show == 0:
        logging.warning("No images to visualize.")
        return
    num_cols = num_images_to_show
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * figsize_scale, num_rows * figsize_scale))
    if num_cols == 1: axes = axes[:, np.newaxis]
    input_channels = ground_truth_images.shape[1]
    cmap = 'gray' if input_channels == 1 else None
    for i in range(num_images_to_show):
        ax_gt = axes[0, i]
        ax_rec = axes[1, i]
        gt_img = ground_truth_images[i].cpu().numpy().transpose(1, 2, 0).squeeze()
        rec_img = reconstructed_images[i].cpu().numpy().transpose(1, 2, 0).squeeze()
        ax_gt.imshow(gt_img, cmap=cmap)
        ax_gt.set_title(f"GT Lbl: {ground_truth_labels[i].item()}")
        ax_gt.axis('off')
        ax_rec.imshow(rec_img, cmap=cmap)
        ax_rec.set_title(f"Rec Lbl: {reconstructed_labels[i].item()}")
        ax_rec.axis('off')
    title_str = f"Gradient Inversion Results\n(MSE:{metrics.get('mse', np.nan):.3f}, PSNR:{metrics.get('psnr', np.nan):.2f}, SSIM:{metrics.get('ssim', np.nan):.3f}, LblAcc:{metrics.get('label_acc', np.nan):.2f})"
    fig.suptitle(title_str)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(
            f"Visualization saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error saving visualization to {save_path}: {e}")
    finally:
        plt.close(fig)


# ======================================================================
# == Attack Wrapper Function (Encapsulates Attack+Eval+Viz) ==
# ======================================================================
def perform_and_evaluate_inversion_attack(target_gradient, model_template, input_shape, num_classes, device,
                                          attack_config, ground_truth_images=None, ground_truth_labels=None,
                                          save_visuals=True, save_dir=Path("./attack_visualizations"), round_num=None,
                                          victim_id=None) -> Dict[str, Any]:
    logging.info(f"Attempting GIA on victim '{victim_id}' (Round {round_num})...")
    attack_start_time = time.time()
    attack_results_log = {"victim_id": victim_id, "round": round_num}
    try:
        attack_model = copy.deepcopy(model_template).to(device)
        attack_model.eval()
        num_images = attack_config.get('num_images', 1)
        iterations = attack_config.get('iterations', 500)
        lr = attack_config.get('lr', 0.1)
        label_type = attack_config.get('label_type', 'optimize')
        atk_kwargs = {k: v for k, v in attack_config.items() if
                      k not in ['num_images', 'iterations', 'lr', 'label_type']}
        current_gt_labels = None
        if ground_truth_images is None:
            if label_type == 'ground_truth': logging.warning(
                f"GT data unavailable for {victim_id}, switching label_type to 'optimize'.")
            label_type = 'optimize'
        elif label_type == 'ground_truth':
            if ground_truth_labels is None or len(ground_truth_labels) != num_images:
                logging.warning(
                    f"GT labels missing/mismatch for {victim_id}, switching label_type to 'optimize'.")
                label_type = 'optimize'
                ground_truth_labels = None
            else:
                current_gt_labels = ground_truth_labels.to(device)
        reconstructed_images, reconstructed_labels = gradient_inversion_attack(
            target_gradient=[g.clone().to(device) for g in target_gradient], model=attack_model,
            input_shape=input_shape, num_classes=num_classes, device=device, num_images=num_images,
            iterations=iterations, lr=lr, label_type=label_type, ground_truth_labels=current_gt_labels, **atk_kwargs)
        attack_duration_sec = time.time() - attack_start_time
        attack_results_log["duration_sec"] = attack_duration_sec
        logging.info(f"Attack on '{victim_id}' finished in {attack_duration_sec:.2f} seconds.")
        if ground_truth_images is not None:
            logging.info(f"Evaluating reconstruction for victim '{victim_id}'...")
            eval_metrics = evaluate_inversion(reconstructed_images=reconstructed_images,
                                              ground_truth_images=ground_truth_images.to(device),
                                              reconstructed_labels=reconstructed_labels,
                                              ground_truth_labels=ground_truth_labels)
            attack_results_log["metrics"] = eval_metrics
            logging.info(f"  Evaluation Metrics: {eval_metrics}")
            if save_visuals: filename = f"round_{round_num}_victim_{victim_id}_comparison.png" if round_num is not None else f"victim_{victim_id}_comparison.png"
            visualize_and_save_attack(
                reconstructed_images, ground_truth_images, reconstructed_labels, ground_truth_labels, eval_metrics,
                save_dir / filename)
        else:
            attack_results_log["metrics"] = None
            logging.info(f"GT data unavailable for '{victim_id}', skipping quantitative evaluation.")
            if save_visuals:
                try:
                    from torchvision.utils import \
                        save_image
                    filename = f"round_{round_num}_victim_{victim_id}_reconstructed.png" if round_num is not None else f"victim_{victim_id}_reconstructed.png"
                    rec_path = save_dir / filename
                    save_dir.mkdir(
                        parents=True, exist_ok=True)
                    save_image(reconstructed_images.cpu(), rec_path,
                               normalize=True)
                    logging.info(
                        f"Saved reconstructed image (no GT) to {rec_path}")
                except ImportError:
                    logging.warning("torchvision.utils not found, cannot save reconstructed image.")
                except Exception as e_save:
                    logging.error(f"Failed to save reconstructed image: {e_save}")
        del attack_model
    except Exception as e:
        logging.error(f"GIA failed for victim {victim_id}: {e}", exc_info=False)
        attack_results_log["error"] = str(e)
        attack_results_log["duration_sec"] = time.time() - attack_start_time
    return attack_results_log
