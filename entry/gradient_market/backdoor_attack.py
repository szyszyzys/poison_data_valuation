import argparse
import os
import random
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset, DataLoader

from attack.attack_gradient_market.poison_attack.attack_martfl import BackdoorImageGenerator
from attack.evaluation.evaluation_backdoor import evaluate_attack_performance_backdoor_poison
from general_utils.file_utils import save_history_to_json
from marketplace.market.markplace_gradient import DataMarketplaceFederated
from marketplace.market_mechanism.martfl import Aggregator
from marketplace.seller.gradient_seller import GradientSeller, AdvancedBackdoorAdversarySeller, \
    update_local_model_from_global
from marketplace.utils.gradient_market_utils.data_processor import get_data_set
from model.utils import get_model, save_samples


def dataloader_to_tensors(dataloader):
    """
    Convert a DataLoader to tensors X (features) and y (labels).

    :param dataloader: PyTorch DataLoader object.
    :return: Tuple of torch.Tensors (X, y).
    """
    X_list, y_list = [], []

    for batch in dataloader:
        X_batch, y_batch = batch
        X_list.append(X_batch)
        y_list.append(y_batch)

    # Concatenate all batches into single tensors
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    return X, y


def generate_attack_test_set(full_dataset, backdoor_generator, n_samples=1000):
    sample_indices = random.sample(range(len(full_dataset)), n_samples)
    subset_dataset = Subset(full_dataset, sample_indices)

    # ---------------------------
    # 2. Extract Images and Labels
    # ---------------------------
    # FashionMNIST images come in shape (1, H, W). For our backdoor generator,
    # assume we want images as (H, W, C). We can squeeze and then unsqueeze at the end.

    X_list = []
    y_list = []
    for img, label in subset_dataset:
        # img is already (C, H, W) in [0,1] from transforms.ToTensor()
        X_list.append(img)
        y_list.append(label)

    # Stack into (N, C, H, W)
    X = torch.stack(X_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)

    # 3. Generate the poisoned dataset
    X_poisoned, y_poisoned = backdoor_generator.generate_poisoned_dataset(X, y, poison_rate=1)

    # 4. Build DataLoaders
    clean_dataset = TensorDataset(X, y)
    triggered_dataset = TensorDataset(X_poisoned, y_poisoned)

    clean_loader = DataLoader(clean_dataset, batch_size=64, shuffle=True)
    triggered_loader = DataLoader(triggered_dataset, batch_size=64, shuffle=True)

    return clean_loader, triggered_loader

    # X_list, y_list = [], []
    # for img, label in subset_dataset:
    #     # img is a torch.Tensor of shape (1, H, W); convert to (H, W, 1)
    #     img = img.permute(1, 2, 0)  # now shape (H, W, C)
    #     X_list.append(img)
    #     y_list.append(label)
    #
    # X = torch.stack(X_list)  # Shape: (10000, H, W, C)
    # y = torch.tensor(y_list)  # Shape: (10000,)
    #
    # X_poisoned, y_poisoned = backdoor_generator.generate_poisoned_dataset(X, y, poison_rate=0.5)
    #
    # X = X.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    # X_poisoned = X_poisoned.permute(0, 3, 1, 2)
    #
    # clean_dataset = TensorDataset(X, y)
    # triggered_dataset = TensorDataset(X_poisoned, y_poisoned)
    #
    # batch_size = 64
    # clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True)
    # triggered_loader = DataLoader(triggered_dataset, batch_size=batch_size, shuffle=True)
    # return clean_loader, triggered_loader


def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(item) for item in obj]
    else:
        return obj


class FederatedEarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, monitor='loss'):
        """
        Args:
            patience (int): Number of rounds to wait after last improvement.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            monitor (str): Metric to monitor, 'loss' for minimizing metric or 'acc' for maximizing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0

    def update(self, current_value):
        """
        Update the stopper with the latest metric value.

        Returns:
            bool: True if training should be stopped, otherwise False.
        """
        # For loss, lower is better; for accuracy, higher is better.
        if self.best_score is None:
            self.best_score = current_value
            return False

        if self.monitor == 'loss':
            improvement = self.best_score - current_value
        elif self.monitor == 'acc':
            improvement = current_value - self.best_score
        else:
            raise ValueError("Monitor must be 'loss' or 'acc'")

        if improvement > self.min_delta:
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def backdoor_attack(dataset_name, n_sellers, n_adversaries, model_structure, aggregation_method='martfl',
                    global_rounds=100, backdoor_target_label=0, trigger_type: str = "blended_patch", save_path="/",
                    device='cpu', poison_strength=1, poison_test_sample=100, args=None, trigger_rate=0.1):
    # load the dataset
    gradient_manipulation_mode = args.gradient_manipulation_mode
    if dataset_name == "FMNIST":
        channels = 1
    else:
        channels = 3
    loss_fn = nn.CrossEntropyLoss()
    backdoor_generator = BackdoorImageGenerator(trigger_type="blended_patch", target_label=backdoor_target_label,
                                                channels=channels)

    early_stopper = FederatedEarlyStopper(patience=50, min_delta=0.01, monitor='acc')
    local_training_params = {
        "lr": args.local_lr,
        "epochs": args.local_epoch,
        "optimizer": "SGD"
    }
    # setup buyers, only one buyer per query. Set buyer cid as 0 for data split

    # set up the data set for the participants
    client_loaders, full_dataset, test_set_loader = get_data_set(dataset_name, buyer_count=5000, num_sellers=n_sellers,
                                                                 iid=True)

    # config the buyer
    buyer = GradientSeller(seller_id="buyer", local_data=client_loaders["buyer"].dataset, dataset_name=dataset_name,
                           save_path=save_path, local_training_params=local_training_params)

    # config the marketplace
    aggregator = Aggregator(save_path=save_path,
                            n_seller=n_sellers,
                            n_adversaries=n_adversaries,
                            model_structure=model_structure,
                            dataset_name=dataset_name,
                            quantization=False,
                            aggregation_method=aggregation_method
                            )
    marketplace = DataMarketplaceFederated(aggregator,
                                           selection_method=aggregation_method, save_path=save_path)

    # config the seller and register to the marketplace
    for cid, loader in client_loaders.items():
        if cid == "buyer":
            continue
        if n_adversaries > 0:
            cur_id = f"adv_{cid}"
            current_seller = AdvancedBackdoorAdversarySeller(seller_id=cur_id,
                                                             local_data=loader.dataset,
                                                             target_label=backdoor_target_label,
                                                             trigger_type=trigger_type, save_path=save_path,
                                                             backdoor_generator=backdoor_generator,
                                                             device=device,
                                                             poison_strength=poison_strength,
                                                             trigger_rate=trigger_rate,
                                                             dataset_name=dataset_name,
                                                             local_training_params=local_training_params,
                                                             gradient_manipulation_mode=gradient_manipulation_mode
                                                             )
            n_adversaries -= 1
        else:
            cur_id = cid
            current_seller = GradientSeller(seller_id=cur_id, local_data=loader.dataset,
                                            dataset_name=dataset_name, save_path=save_path, device=device,
                                            local_training_params=local_training_params)

        marketplace.register_seller(cur_id, current_seller)

    # config the attack test set.
    clean_loader, triggered_loader = generate_attack_test_set(full_dataset, backdoor_generator, poison_test_sample)
    # save some
    save_samples(clean_loader, filename=f"{save_path}/clean_samples.png", n_samples=16, nrow=4, title="Clean Samples")
    save_samples(triggered_loader, filename=f"{save_path}/triggered_samples.png", n_samples=16, nrow=4,
                 title="Triggered Samples")

    # Start gloal round
    aggregated_gradient = None
    for gr in range(global_rounds):
        # update buyers's local model
        if aggregated_gradient:
            update_local_model_from_global(buyer, dataset_name, aggregated_gradient)
        buyer_gradient = buyer.get_gradient()
        # train the attack model
        round_record, aggregated_gradient = marketplace.train_federated_round(round_number=gr,
                                                                              buyer_gradient=buyer_gradient,
                                                                              test_dataloader_buyer_local=
                                                                              client_loaders["buyer"],
                                                                              test_dataloader_global=test_set_loader,
                                                                              clean_loader=clean_loader,
                                                                              triggered_loader=triggered_loader,
                                                                              loss_fn=loss_fn, device=device,
                                                                              dataset_name=dataset_name,
                                                                              backdoor_target_label=backdoor_target_label)

        if gr % 10 == 0:
            torch.save(marketplace.round_logs, f"{save_path}/market_log_round_{gr}.ckpt")
        if round_record["final_perf_global"] is not None:
            current_val_loss = round_record["final_perf_global"]["loss"]
            if early_stopper.update(current_val_loss):
                print(f"Early stopping triggered at round {gr}.")
                break
    poison_metrics = evaluate_attack_performance_backdoor_poison(marketplace.aggregator.global_model, clean_loader,
                                                                 triggered_loader,
                                                                 marketplace.aggregator.device,
                                                                 target_label=backdoor_target_label, plot=True,
                                                                 save_path=f"{save_path}/final_backdoor_attack_performance.png")

    # post fl process, test the final model.
    torch.save(marketplace.aggregator.global_model.state_dict(), f"{save_path}/final_global_model.pt")
    torch.save(marketplace.round_logs, f"{save_path}/market_log.ckpt")
    converted_logs = convert_np(marketplace.round_logs)
    save_history_to_json(converted_logs, f"{save_path}/market_log.json")
    # record the result for each seller
    all_sellers = marketplace.get_all_sellers
    for seller_id, seller in all_sellers.items():
        converted_logs_user = convert_np(seller.get_federated_history)
        torch.save(converted_logs_user, f"{save_path}/local_log_{seller_id}.ckpt")

    # record the attack result for the final round


def parse_args():
    parser = argparse.ArgumentParser(description="Run Backdoor Attack Experiment")

    # Required arguments
    parser.add_argument('--dataset_name', type=str, default='FMNIST',
                        help='Name of the dataset (e.g., MNIST, CIFAR10)')
    parser.add_argument('--n_sellers', type=int, default=10, help='Number of sellers')
    parser.add_argument('--n_adversaries', type=int, default=1, help='Number of adversaries')

    # Optional arguments with defaults
    parser.add_argument('--global_rounds', type=int, default=1, help='Number of global training rounds')
    parser.add_argument('--backdoor_target_label', type=int, default=0, help='Target label for backdoor attack')
    parser.add_argument('--trigger_type', type=str, default="blended_patch", help='Type of backdoor trigger')
    parser.add_argument('--exp_name', type=str, default="/", help='Experiment name for logging')
    parser.add_argument('--poison_test_sample', type=int, default=1000, help='Number of samples for global test')
    parser.add_argument('--local_epoch', type=int, default=1, help='Number of global training rounds')

    parser.add_argument('--poison_strength', type=float, default=1, help='Strength of poisoning')
    parser.add_argument('--local_lr', type=float, default=1e-3, help='Strength of poisoning')
    parser.add_argument('--trigger_rate', type=float, default=0.5, help='Strength of poisoning')

    parser.add_argument('--gradient_manipulation_mode', type=str, default="cmd",
                        help='Type of gradient manipulation cmd single')
    # Model architecture argument
    parser.add_argument('--model_arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'mlp'],
                        help='Model architecture (resnet18, resnet34, mlp)')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--aggregation_method", type=str, default="martfl", help="agge")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs (e.g., '0,1').")

    args = parser.parse_args()
    return args


def set_seed(seed: int):
    """Set the seed for random, numpy, and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures that CUDA selects deterministic algorithms when available.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to: {seed}")


def get_device(args) -> str:
    """
    Returns a string representing the device based on available GPUs and args.gpu_ids.
    The --gpu_ids argument should be a comma-separated string of GPU indices (e.g., "0,1,2").
    """
    if torch.cuda.is_available():
        # Parse the gpu_ids argument into a list of integers.
        gpu_ids = [int(id_) for id_ in args.gpu_ids.split(',')]
        # Set CUDA_VISIBLE_DEVICES so that only these GPUs are visible.
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
        # Return the first GPU as the default device string.
        device_str = "cuda:0"
        print(f"[INFO] Using GPUs: {gpu_ids}. Default device set to {device_str}.")
    else:
        device_str = "cpu"
        print("[INFO] CUDA not available. Using CPU.")
    return device_str


def clear_work_path(path):
    """
    Delete all files and subdirectories in the specified path.

    Parameters:
        path (str): The directory path to clear.
    """
    if not os.path.exists(path):
        print(f"Path '{path}' does not exist.")
        return

    # List all items in the directory.
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            # Remove a file or symbolic link.
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {file_path}")
            # Remove a directory and its contents.
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    args = parse_args()
    t_model = get_model(args.dataset_name)
    print(
        f"start backdoor attack, current dataset: {args.dataset_name}, n_sellers: {args.n_sellers}, attack method: {args.gradient_manipulation_mode}")
    set_seed(args.seed)
    device = get_device(args)
    if args.gradient_manipulation_mode == "None":
        save_path = f"./results/backdoor/{args.aggregation_method}/{args.dataset_name}/no_attack/n_seller_{args.n_sellers}_local_epoch_{args.local_epoch}_local_lr_{args.local_lr}/"
    elif args.gradient_manipulation_mode == "cmd":
        save_path = f"./results/backdoor/{args.aggregation_method}/{args.dataset_name}/backdoor_mode_{args.gradient_manipulation_mode}_strength_{args.poison_strength}_trigger_type_{args.trigger_type}/n_seller_{args.n_sellers}_n_adv_{args.n_adversaries}_local_epoch_{args.local_epoch}_local_lr_{args.local_lr}/"
    elif args.gradient_manipulation_mode == "single":
        save_path = f"./results/backdoor/{args.aggregation_method}/{args.dataset_name}/backdoor_mode_{args.gradient_manipulation_mode}_trigger_rate_{args.trigger_rate}_trigger_type_{args.trigger_type}/n_seller_{args.n_sellers}_n_adv_{args.n_adversaries}_local_epoch_{args.local_epoch}_local_lr_{args.local_lr}/"
    else:
        raise NotImplementedError(f"No such attack type :{args.gradient_manipulation_mode}")
    clear_work_path(save_path)
    backdoor_attack(
        dataset_name=args.dataset_name,
        n_sellers=args.n_sellers,
        n_adversaries=args.n_adversaries,
        model_structure=t_model,
        global_rounds=args.global_rounds,
        backdoor_target_label=args.backdoor_target_label,
        trigger_type=args.trigger_type,
        save_path=save_path,
        device=device,
        poison_strength=args.poison_strength,
        poison_test_sample=args.poison_test_sample,
        aggregation_method=args.aggregation_method,
        trigger_rate=args.trigger_rate,
        args=args
    )

    # print("Evaluation Results:", eval_results)
