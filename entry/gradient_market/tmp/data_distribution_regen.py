import argparse
import logging
# log_utils.py (or results_logger.py)
import os
from pathlib import Path
from typing import Dict, Optional, Any

from scipy import datasets
from torch import nn

from entry.gradient_market.automate_exp.config_parser import parse_config_for_attack_function
from entry.gradient_market.backdoor_attack import FederatedEarlyStopper, load_config, set_seed
from general_utils.file_utils import save_to_json
from marketplace.utils.gradient_market_utils.data_processor import print_and_save_data_statistics, \
    generate_buyer_bias_distribution, split_dataset_discovery, get_transforms
from model.utils import get_model_name, get_domain

logger = logging.getLogger(__name__)


def poisoning_attack_text_reg_distri(
        dataset_name: str,
        n_sellers: int,
        adv_rate: float,
        model_structure: str,
        attack_type: str,  # 'backdoor' or 'label_flip'
        aggregation_method: str = 'martfl',
        global_rounds: int = 100,
        # --- Backdoor Params ---
        backdoor_target_label: Optional[int] = 0,
        backdoor_trigger_type="",
        backdoor_trigger_content: str = "cf",  # Specific trigger word/phrase
        backdoor_trigger_location: str = "end",
        poison_rate: float = 0.1,
        backdoor_poison_strength: float = 1.0,  # May not be applicable if not manipulating gradients
        # --- Label Flip Params ---
        label_flip_target_label: int = 0,
        label_flip_mode: str = "fixed_target",
        # --- Text Model Hyperparameters ---
        text_model_hyperparams: Optional[Dict] = None,
        # --- Common Params ---
        save_path: str = "/",
        device: str = 'cpu',
        args: Optional[Any] = None,
        buyer_percentage: float = 0.02,
        sybil_params: Optional[Dict] = None,
        local_training_params: Optional[Dict] = None,
        change_base: bool = True,
        data_split_mode: str = "NonIID",
        dm_params: Optional[Dict] = None, local_attack_params=None, privacy_attack={}
):
    """
    Runs a federated learning experiment with either Backdoor or Label Flipping TEXT poisoning.

    Args:
        dataset_name (str): Name of the text dataset ("AG_NEWS" or "TREC").
        # --- Other args similar to image function ---
        backdoor_trigger_content (str): Word/phrase trigger for text backdoor.
        backdoor_trigger_location (str): Where to insert the trigger in text.
        text_model_hyperparams (Optional[Dict]): Hyperparameters for the TextCNN model.
        # --- Other args similar to image function ---
    """
    print(f"--- Starting TEXT Poisoning Attack ---")
    print(f"Dataset: {dataset_name}, Attack Type: {attack_type}")
    print(f"Sellers: {n_sellers}, Adversary Rate: {adv_rate}")

    # --- Basic Setup ---
    if aggregation_method == "skymask":
        print(f"return, not implemented: {aggregation_method} for text data")
    n_adversaries = int(n_sellers * adv_rate)
    if args is None:  # Use default args if none provided
        from types import SimpleNamespace
        args = SimpleNamespace(gradient_manipulation_mode='passive', is_sybil=False, clip=None, remove_baseline=False)
    gradient_manipulation_mode = args.gradient_manipulation_mode
    loss_fn = nn.CrossEntropyLoss()
    es_monitor = 'acc'
    early_stopper = FederatedEarlyStopper(patience=10, min_delta=0.01, monitor=es_monitor)
    if sybil_params is None: sybil_params = {'benign_rounds': 0, 'sybil_mode': 'passive', 'alpha': 1,
                                             'amplify_factor': 1, 'cost_scale': 1, 'trigger_mode': 'data'}
    if local_training_params is None: local_training_params = {'epochs': 1, 'lr': 0.01, 'batch_size': 64}
    if dm_params is None: dm_params = {"discovery_quality": 0.3, "buyer_data_mode": "random"}
    if text_model_hyperparams is None:
        text_model_hyperparams = {"embed_dim": 100, "num_filters": 100, "filter_sizes": [3, 4, 5], "dropout": 0.5}

    # --- Load Data (get vocab and padding_idx) ---
    print("Loading clean text data splits...")
    # This function should return the vocab and padding_idx needed later
    get_text_data_set_distri(
        dataset_name,
        buyer_percentage=buyer_percentage,
        num_sellers=n_sellers,
        batch_size=local_training_params.get('batch_size', 64),
        split_method=data_split_mode,
        n_adversaries=n_adversaries,
        discovery_quality=dm_params[
            "discovery_quality"],
        buyer_data_mode=dm_params[
            "buyer_data_mode"]
    )


def get_text_data_set_distri(
        dataset_name: str,
        buyer_percentage: float = 0.01,
        num_sellers: int = 10,
        split_method: str = "discovery",
        discovery_quality: float = 0.3,
        buyer_data_mode: str = "unbiased",
        buyer_bias_type: str = "dirichlet",
        buyer_dirichlet_alpha: float = 0.3,
        seed: int = 42,
        data_root: str = "./data",
        save_path: str = "./result"
):
    """
    Load only the labels of a text dataset, perform a split, and log distribution stats.

    Returns:
        - buyer_indices: List of indices assigned to the buyer
        - seller_splits: Dict mapping seller_id -> list of indices
        - data_distribution_info: whatever print_and_save_data_statistics returns
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("HuggingFace 'datasets' library required but not installed.")

    # ── LOAD RAW LABELS ────────────────────────────────────────
    if dataset_name == "AG_NEWS":
        logging.info(f"Loading AG_NEWS dataset from HuggingFace cache at {data_root}...")
        ds = hf_load("ag_news", cache_dir=data_root)
        train_ds, test_ds = ds["train"], ds["test"]
        label_field = "label"
        num_classes = 4
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        vocab_source_iter = (ex["text"] for ex in train_ds if isinstance(ex.get("text"), str))
        train_iter = ((ex[label_field], ex["text"]) for ex in train_ds if isinstance(ex.get("text"), str))
        test_iter = ((ex[label_field], ex["text"]) for ex in test_ds if isinstance(ex.get("text"), str))

    elif dataset_name == "TREC":
        logging.info(f"Loading TREC dataset from HuggingFace cache at {data_root}...")
        ds = hf_load("trec", "default", cache_dir=data_root)
        train_ds, test_ds = ds["train"], ds["test"]
        label_field = "coarse_label"
        num_classes = 6
        class_names = ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']
        vocab_source_iter = (ex["text"] for ex in train_ds)
        train_iter = ((ex[label_field], ex["text"]) for ex in train_ds)
        test_iter = ((ex[label_field], ex["text"]) for ex in test_ds)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Extract all labels from the training split
    labels_only = [example[label_field] for example in train_ds]
    total_samples = len(labels_only)
    buyer_count = min(int(total_samples * buyer_percentage), total_samples)

    # ── SPLITTING ───────────────────────────────────────────────
    if split_method == "discovery":
        buyer_bias_dist = generate_buyer_bias_distribution(
            num_classes=num_classes,
            bias_type=buyer_bias_type,
            alpha=buyer_dirichlet_alpha,
            seed=seed
        )
        buyer_indices, seller_splits = split_dataset_discovery(
            dataset=labels_only,
            buyer_count=buyer_count,
            num_clients=num_sellers,
            noise_factor=discovery_quality,
            buyer_data_mode=buyer_data_mode,
            buyer_bias_distribution=buyer_bias_dist,
            seed=seed
        )
    else:
        raise ValueError(f"Unsupported split_method: {split_method!r}")

    # ── LOG & SAVE DISTRIBUTION ─────────────────────────────────
    os.makedirs(save_path, exist_ok=True)
    data_distribution_info = print_and_save_data_statistics(
        labels_only,
        buyer_indices,
        seller_splits,
        save_results=True,
        output_dir=save_path
    )

    return buyer_indices, seller_splits, data_distribution_info


def get_image_data_distribution(
        dataset_name: str,
        n_sellers: int,
        adv_rate: float,
        attack_type: str,  # 'backdoor' or 'label_flip'
        model_structure: str,  # Pass the model class/constructor
        aggregation_method: str = 'martfl',
        global_rounds: int = 100,
        # --- Backdoor Params ---
        backdoor_target_label: Optional[int] = 0,
        backdoor_trigger_type: str = "blended_patch",
        backdoor_trigger_location: str = "bottom_right",  # From args.bkd_loc previously
        poison_rate: float = 0.1,  # trigger_rate previously
        backdoor_poison_strength: float = 1.0,  # poison_strength previously
        # --- Label Flip Params ---
        label_flip_target_label: int = 0,  # Target for fixed_target mode
        label_flip_mode: str = "fixed_target",  # 'fixed_target' or 'random_different'
        # --- Common Params ---
        save_path: str = "/",
        device: str = 'cpu',
        args: Optional[Any] = None,  # Pass general args if needed
        buyer_percentage: float = 0.02,
        sybil_params: Optional[Dict] = None,
        local_training_params: Optional[Dict] = None,
        change_base: bool = True,
        data_split_mode: str = "NonIID",
        dm_params: Optional[Dict] = None, local_attack_params=None, privacy_attack={}
):
    """
    Load and split image data for a poisoning experiment (but do NOT run the attack itself).

    Returns:
      - buyer_loader:        DataLoader for the buyer’s portion
      - client_loaders:      List of DataLoaders (one per seller) for the clean data
      - test_loader:         DataLoader for the held-out test set
      - class_names:         List of class names in the dataset
    """
    # ——— Logging setup —————————————————————————————————————————————
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Preparing data splits for {dataset_name} | "
                 f"{n_sellers} sellers | {adv_rate * 100:.1f}% adversaries")

    # ——— Compute number of adversaries —————————————————————————————
    n_adversaries = int(n_sellers * adv_rate)

    # ——— Ensure output directory exists ——————————————————————————
    os.makedirs(save_path, exist_ok=True)

    # ——— Call your existing splitter ——————————————————————————
    buyer_loader, client_loaders, _, test_loader, class_names = get_img_distribution(
        dataset_name=dataset_name,
        buyer_percentage=buyer_percentage,
        num_sellers=n_sellers,
        split_method=data_split_mode,
        n_adversaries=n_adversaries,
        save_path=save_path,
        discovery_quality=dm_params["discovery_quality"],
        buyer_data_mode=dm_params["buyer_data_mode"]
    )

    logging.info("Data split complete.")
    logging.info(f" → Buyer samples: {len(buyer_loader.dataset)}")
    logging.info(f" → Each seller ≈ {len(client_loaders[0].dataset)} samples")
    logging.info(f" → Test samples:  {len(test_loader.dataset)}")
    logging.info(f" → Classes:       {class_names}")

    return buyer_loader, client_loaders, test_loader, class_names


def get_img_distribution(
        dataset_name,
        buyer_percentage=0.01,
        num_sellers=10,
        batch_size=64,
        normalize_data=True,
        split_method="discovery",  # Changed default to make the relevant part active
        n_adversaries=0,
        save_path='./result',
        # --- Discovery Split Specific Params ---
        discovery_quality=0.3,
        buyer_data_mode="random",
        buyer_bias_type="dirichlet",  # Added: Specify how buyer bias is generated
        buyer_dirichlet_alpha=0.3,  # Added: Alpha specifically for buyer bias
        # --- Other Split Method Params ---
        seller_dirichlet_alpha=0.7  # Alpha used in the default/other split method
):
    # Define transforms based on the dataset.
    # (Keep your transform definitions here)
    transform = get_transforms(dataset_name, normalize_data=normalize_data)
    print(f"Using transforms for {dataset_name}: {transform}")
    # Load training and test datasets.
    if dataset_name == "FMNIST":
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name == "CIFAR":
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == "MNIST":
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        class_names = [str(i) for i in range(10)]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")

    # --- Derive NUM_CLASSES dynamically ---
    num_classes = len(dataset.classes)
    print(f"Dataset: {dataset_name}, Number of classes: {num_classes}")

    # Determine the number of buyer samples.
    total_samples = len(dataset)
    buyer_count = int(total_samples * buyer_percentage)
    print(f"Allocating {buyer_count} samples ({buyer_percentage * 100:.2f}%) for the buyer.")

    # --- Conditional Data Splitting ---
    if split_method == "discovery":
        print(f"Using 'discovery' split method with buyer bias type: '{buyer_bias_type}'")
        # Generate buyer distribution ONLY when needed
        buyer_biased_distribution = generate_buyer_bias_distribution(
            num_classes=num_classes,  # Use derived num_classes
            bias_type=buyer_bias_type,
            alpha=buyer_dirichlet_alpha  # Use argument for alpha
        )
        print(f"Generated buyer bias distribution: {buyer_biased_distribution}")

        buyer_indices, seller_splits = split_dataset_discovery(
            dataset=dataset,
            buyer_count=buyer_count,
            num_clients=num_sellers,
            noise_factor=discovery_quality,
            buyer_data_mode=buyer_data_mode,
            buyer_bias_distribution=buyer_biased_distribution  # Pass generated dist
        )

    # --- Post-splitting steps ---
    data_distribution_info = print_and_save_data_statistics(dataset, buyer_indices, seller_splits, save_results=True,
                                                            output_dir=save_path)


def main():
    # 1. Set up argparse to accept only the config file path
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment from Config File")
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument("--rerun", type=str, default="false", help="Path to the YAML configuration file")

    cli_args = parser.parse_args()
    print(f"start run with config: {cli_args.config}")

    # 2. Load the configuration file
    config = load_config(cli_args.config)
    if config is None:
        logging.error(f"Failed to load configuration from {cli_args.config}. Exiting.")
        return  # Exit if config loading fails

    # 3. Extract parameters needed for setup (outside the loop)
    experiment_id = config.get('experiment_id', os.path.splitext(os.path.basename(cli_args.config))[0])
    dataset_name = config.get('dataset_name')
    model_structure_name = config.get('model_structure')  # Get model name from config
    base_save_dir = config.get('output', {}).get('save_path_base', './experiment_results')
    n_samples = config.get('n_samples', 3)  # Number of runs with different seeds
    initial_seed = config.get('seed', 42)

    if not dataset_name or not model_structure_name:
        logging.error("Config missing 'dataset_name' or 'model_structure'. Exiting.")
        return

    # Construct the base save path for this specific experiment config
    # experiment_base_path = os.path.join(base_save_dir, experiment_id)
    experiment_base_path = config.get('output', {}).get('final_save_path')
    print(f"Base results directory for this experiment: {experiment_base_path}")

    # Ensure base path exists
    Path(experiment_base_path).mkdir(parents=True, exist_ok=True)

    # 4. Prepare arguments dictionary using the parser function
    # This encapsulates the mapping logic
    attack_func_args = parse_config_for_attack_function(config)
    if attack_func_args is None:
        logging.error("Failed to parse configuration into function arguments. Exiting.")
        return

    # 5. Get Model structure (do this once outside the loop)
    # Pass model structure name or definition from config
    dataset_domain = get_domain(dataset_name)
    attack_func_args['model_structure'] = get_model_name(dataset_name)  # Pass the actual model object/class

    # 6. Save parameters used for this experiment group (optional)
    all_params_to_save = {
        "sybil_params": attack_func_args.get('sybil_params'),
        "local_training_params": attack_func_args.get('local_training_params'),
        "local_attack_params": attack_func_args.get('local_attack_params'),  # Usually None here
        "dm_params": attack_func_args.get('dm_params'),
        "full_config": config  # Save the original config for traceability
    }
    save_to_json(all_params_to_save, f"{experiment_base_path}/experiment_params.json")

    # 7. Loop for multiple runs (if n_samples > 1)
    print(f"Starting, dataset type: {dataset_domain}, {n_samples} run(s) for experiment: {experiment_id}")
    for i in range(n_samples):
        current_seed = initial_seed + i
        set_seed(current_seed)  # Set seed for this specific run

        # Define save path for this specific run
        current_run_save_path = os.path.join(experiment_base_path, f"run_{i}")
        Path(current_run_save_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"\n--- Starting Run {i} (Seed: {current_seed}) ---")
        logging.info(f"Saving results to: {current_run_save_path}")

        # Update arguments that change per run (save_path, potentially seed if needed inside)
        run_specific_args = attack_func_args.copy()
        run_specific_args['save_path'] = current_run_save_path
        # Update seed within the simulated args object if backdoor_attack uses args.seed
        if hasattr(run_specific_args['args'], 'seed'):
            run_specific_args['args'].seed = current_seed

        try:
            if dataset_domain == 'image':
                get_image_data_distribution(**run_specific_args)
            else:
                poisoning_attack_text_reg_distri(**run_specific_args)
            logging.info(f"--- Finished Run {i} ---")
        except Exception as e:
            logging.error(f"!!! Error during Run {i} for experiment {experiment_id} !!!")
            logging.error(f"Config file: {cli_args.config}")
            logging.error(f"Save path: {current_run_save_path}")
            logging.error(f"Exception: {e}", exc_info=True)  # Log traceback
            # Decide if you want to continue to the next run or stop
            # continue

    print(f"\nFinished all {n_samples} run(s) for experiment: {experiment_id}")


if __name__ == "__main__":
    main()
