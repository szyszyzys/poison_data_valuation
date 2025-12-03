# Benchmarking Robust Aggregation in Decentralized Gradient Marketplaces

## 📖 Overview

This repository contains the official implementation of the benchmark framework presented in **"[Insert Paper Title]"**.
It provides a modular, reproducible environment to evaluate the robustness of decentralized learning aggregators against
various adversary models (Sybil, Poisoning, Adaptive) across Image, Text, and Tabular modalities.

## 🏗️ Environment Setup

To ensure reproducibility, we recommend running this framework in a dedicated Conda environment.

### 1\. Prerequisites

* **OS:** Ubuntu 20.04 / 22.04 LTS
* **Package Manager:** [Anaconda](https://www.anaconda.com/)
  or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 2\. Installation Steps

Run the following commands in your terminal to set up the environment:

```bash
# 1. Create a clean Conda environment (using Python 3.10)
conda create -n dgm python=3.10 -y

# 2. Activate the environment
conda activate dgm

# 3. Upgrade pip to avoid dependency resolution errors
pip install --upgrade pip

# 4. Install Python dependencies
pip install -r requirements.txt
```

-----

# 📂 Project Structure & Extensibility

While this benchmark focuses on **Gradient Marketplaces** (where the "product" is a gradient update $\nabla \theta$), the underlying framework is architected as a generic decentralized exchange.

* **Paper Focus (Gradient Market):** The experiments in this submission utilize `marketplace/market/markplace_gradient.py` and `attack/attack_gradient_market`.
* **Extensibility (Raw Data Market):** We provide an experimental implementation for **Raw Row Data** exchange to demonstrate architectural generality. The base classes in `marketplace/` are polymorphic, allowing `data_seller.py` and `attack_data_market` to function alongside the gradient implementations.

## Directory Overview

```text
./
├── LICENSE
├── README.md
├── requirements.txt
├── experiments/
│   ├── configs/
│   │   ├── base_configs.py
│   │   ├── config_generator.py
│   │   ├── config_parser.py
│   │   ├── scenarios.py
│   │   └── tabular_scenarios.py
│   ├── scenarios/
│   │   ├── generate_step1_iid_tuning.py
│   │   ├── generate_step2_find_usable_hps.py
│   │   ├── generate_step3_defense_tuning.py
│   │   ├── generate_step4_attack_sensitivity.py
│   │   ├── generate_step5_advanced_sybil.py
│   │   ├── generate_step6_adaptive_attack.py
│   │   ├── generate_step7_buyer_attacks.py
│   │   ├── generate_step8_scalability.py
│   │   ├── generate_step9_heterogeneity.py
│   │   ├── generate_step10_main_summary.py
│   │   ├── generate_step13_drowning_attack.py
│   │   ├── generate_step14_martfl_collusion.py
│   │   └── step14.py
│   └── visualization/
│       ├── compare_skymasks.py
│       ├── step2_visual.py
│       ├── step3_visual.py
│       ├── step4_visual.py
│       ├── step5_visual_sybil.py
│       ├── step6_visual_adaptive_attack.py
│       ├── step7_fltrust_martfl.py
│       ├── step8_scalability_visual.py
│       ├── step9_heterogeneity_visual.py
│       ├── step10_summary.py
│       ├── step10_valuation.py
│       ├── step13_visual.py
│       └── step14_martfl_visual.py
└── src/
    ├── __init__.py
    ├── attacks/
    │   ├── __init__.py
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── adv.py
    │   │   ├── attack_daved.py
    │   │   └── clip_adv.py
    │   ├── gradient/
    │   │   ├── __init__.py
    │   │   ├── gradient_manipulation.py
    │   │   ├── pfed.py
    │   │   └── sybil_coordinator.py
    │   └── privacy/
    │       ├── __init__.py
    │       ├── inference.py
    │       ├── reconstruction.py
    │       ├── gradient_attack.py
    │       ├── malicious_seller.py
    │       └── privacy_attacker.py
    ├── mechanisms/
    │   ├── __init__.py
    │   ├── defenses/
    │   │   ├── __init__.py
    │   │   ├── fltrust.py
    │   │   ├── martfl.py
    │   │   ├── skymask.py
    │   │   └── fedavg.py
    │   ├── discovery/
    │   │   ├── __init__.py
    │   │   └── daved.py
    │   └── valuation/
    │       ├── __init__.py
    │       ├── shapley.py
    │       ├── influence.py
    │       └── contribution_evaluator.py
    ├── markets/
    │   ├── __init__.py
    │   ├── data_market.py
    │   └── gradient_market.py
    ├── participants/
    │   ├── __init__.py
    │   ├── buyers.py
    │   └── sellers.py
    ├── models/
    │   ├── __init__.py
    │   ├── image_model.py
    │   └── tabular_model.py
    └── utils/
        ├── __init__.py
        ├── data_loader.py
        ├── model_utils.py
        └── logging_utils.py
```

-----

# 🛠️ Scripts & Configurations

### Experiment Orchestrator

The entire benchmark is driven by a custom orchestration script (`entry/run_parallel_experiment.py`) designed for
reliability and high-throughput parallel execution.

**Key Features:**

* **Parallel Execution:** Automatically distributes experiments across available CPU cores or GPU devices.
* **State Management:** Uses file-based locking (`.lock`) and success markers (`.success`) to skip completed runs. You
  can safely interrupt and resume the script.
* **Automated Recovery:** Detects `NaN`/`Inf` divergences and retries up to 3 times.

### 🔧 CLI Arguments

| Argument          | Default             | Description                                                                       |
|:------------------|:--------------------|:----------------------------------------------------------------------------------|
| `--configs_dir`   | `configs_generated` | Path to the directory containing generated YAML configs.                          |
| `--gpu_ids`       | `None`              | Comma-separated list of CUDA device IDs (e.g., `0,1,2`). If omitted, runs on CPU. |
| `--num_processes` | `os.cpu_count()`    | Total number of parallel workers.                                                 |
| `--force_rerun`   | `False`             | Ignores existing success markers and forces re-execution.                         |

**Example Usage:**

```bash
# Run experiments on GPUs 0 and 1 with 4 parallel workers
python experiments/gradient_market/run_parallel_experiment.py --gpu_ids 0,1 --num_processes 4 --configs_dir 'configs_generated/step10_main_summary'
```

-----

# 🎛️ Unified Configuration System

Our framework adopts a **Single-Entry, Data-Driven** architecture. The entire experimental logic is controlled by a
strictly typed hierarchy of configuration objects (`dataclasses`). `main.py` remains static, while the YAML
configuration dictates the experiment logic.

## Configuration Hierarchy

The root configuration object is `AppConfig`, which orchestrates the following modular subsystems:

```mermaid
classDiagram
    class AppConfig {
        +ExperimentConfig experiment
        +TrainingConfig training
        +DataConfig data
        +AggregationConfig aggregation
        +AdversarySellerConfig adversary_seller
    }
    class AggregationConfig {
        +Method: [MartFL, SkyMask, FedAvg]
    }
    class AdversarySellerConfig {
        +PoisoningConfig poisoning
        +SybilConfig sybil
        +AdaptiveAttackConfig adaptive
    }
    AppConfig --> DataConfig
    AppConfig --> AggregationConfig
    AppConfig --> AdversarySellerConfig
```

## Component Overview

| Config Module               | Responsibility                                                                           |
|:----------------------------|:-----------------------------------------------------------------------------------------|
| **`DataConfig`**            | Switches context between **Image** (CIFAR), **Text** (TREC), and **Tabular** (Texas100). |
| **`AggregationConfig`**     | Defines the server's defense (e.g., swapping `fedavg` to `skymask`).                     |
| **`AdversarySellerConfig`** | A unified profile for malicious sellers (Backdoor, Sybil, Drowning).                     |

-----

# ⚙️ Automated Configuration Pipeline

To ensure fair comparison, we use Python scripts to programmatically generate the experiment suites.

**Workflow:**

1. **Generate:** Run a step script (below) to create YAML files in `configs_generated/`.
2. **Execute:** Run `entry/run_parallel_experiment.py` pointing to that folder.

## Generation Script Roadmap

To reproduce specific parts of the paper, use the corresponding generation script to create the configuration files,
then run them using the orchestrator.

| Step   | Script Path                                                  | Purpose                                                                                                                                         | Corresponds To                          |
|:-------|:-------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------|
| **1**  | `experiments/gradient_market/generate_step1_iid_tuning.py`         | **Benign Baselines:** Establishes the ideal optimizer and learning rate for standard `FedAvg` (Benign) across all datasets.                     | **Table 1** (Baseline Performance)      |
| **2**  | `experiments/gradient_market/generate_step2_find_usable_hps.py`    | **Fairness Calibration:** Runs a sweep to find "usable" training HPs (LR, Optimizer) specifically for *each defense* to ensure fair comparison. | **Fairness Methodology**                |
| **3**  | `experiments/gradient_market/generate_step3_defense_tuning.py`     | **Defense Hyperparameter Search:** Uses optimal LRs from Step 2 to tune internal defense parameters (e.g., `clip_norm`, `max_k`).               | **Figures 3 & 4** (Defense Sensitivity) |
| **4**  | `experiments/gradient_market/generate_step4_attack_sensitivity.py` | **Attack Sensitivity:** Evaluates defense performance under varying attack strengths (Poison Rates).                                            | **Attack Analysis**                     |
| **5**  | `experiments/gradient_market/generate_step5_advanced_sybil.py`     | **Sybil Attacks:** Configures coordinated Sybil attacks (Mimicry, Pivot, Knock-out) against the marketplace.                                    | **Sybil Robustness Section**            |
| **6**  | `experiments/gradient_market/generate_step6_adaptive_attack.py`    | **Adaptive Attacks:** Configures sophisticated individual attacks (Stealthy Drowning, Gradient Manipulation).                                   | **Adaptive Defense Section**            |
| **7**  | `experiments/gradient_market/generate_step7_buyer_attacks.py`      | **Buyer Attacks:** Simulates malicious buyers attempting DoS, Starvation, or Class Exclusion attacks.                                           | **Buyer Threat Analysis**               |
| **8**  | `experiments/gradient_market/generate_step8_scalability.py`        | **Scalability Tests:** Measures system throughput and accuracy as the number of sellers increases (10 to 100+).                                 | **Scalability Figures**                 |
| **9**  | `experiments/gradient_market/generate_step9_heterogeneity.py`      | **Heterogeneity Impact:** Tests robustness under varying degrees of Non-IID data distributions (Dirichlet alpha).                               | **Ablation Studies**                    |
| **10** | `experiments/gradient_market/generate_step10_main_summary.py`      | **Main Benchmark:** The comprehensive suite combining the best parameters for the final comparison.                                             | **Figure 5** (Main Results)             |

-----

## 📊 Reproducing Results

After running the experiments, use the analysis scripts to generate the paper figures.

| Paper Content | Script Path                                                                                                                         |
|:-------------|:------------------------------------------------------------------------------------------------------------------------------------|
| **Table 1**  | `experiments/gradient_market/visualization/step3_visual.py`                                                                               |
| **Figure 2** | `experiments/gradient_market/visualization/step10_summary.py`                                                                             |
| **Figure 3** | `experiments/gradient_market/visualization/step3_visual.py`                                                                               |
| **Figure 4** | `experiments/gradient_market/visualization/step4_visual.py`                                                                               |
| **Figure 5** | `experiments/gradient_market/visualization/step8_scalability_visual.py`                                                                   |
| **Figure 6** | `experiments/gradient_market/visualization/step5_visual_sybil.py` & `experiments/gradient_market/visualization/step6_visual_adaptive_attack.py` |
| **Figure 7** | `experiments/gradient_market/visualization/step7_fltrust_martfl.py`                                                                       |
| **Figure 8** | `experiments/gradient_market/visualization/step7_fltrust_martfl.py`                                                                       |
| **Figure 9** | `experiments/gradient_market/visualization/step9_heterogeneity_visual.py`                                                                 |
| **Figure 10** | `experiments/gradient_market/visualization/step10_valuation.py`                                                                           |

-----

Yes, absolutely. Adding an **"Architecture & Extensibility"** section derived from this text will significantly strengthen the README.

It demonstrates to reviewers that the code isn't just a static script, but a **framework** designed for longevity (Design Principle \#3).

Here is the drafted section. I have translated your academic LaTeX definitions into practical "Developer Instructions" and added a diagram to visualize the modules described in your paper.

### Copy-Paste this section into your README (e.g., before "Installation" or at the end as "Developer Guide")

-----

# 🧩 Architecture & Extensibility

## High-Level Design
The system consists of five pluggable modules that interact via strict interfaces:

```mermaid
graph TD
    subgraph Market["The Marketplace"]
        Asset[("1. Market Asset<br>(Dataset/Model)")]
        Regulator["3. The Regulator<br>(Aggregator/Defense)"]
        Economy["4. The Economy<br>(Valuation Engine)"]
    end
    
    subgraph Participants
        Supply["2. The Supply Side<br>(Honest & Malicious Sellers)"]
        Buyer["Buyer<br>(Root Trust)"]
    end
    
    subgraph Audit
        Auditor["5. The Auditor<br>(Privacy/Integrity Checks)"]
    end

    Supply -->|Gradients| Regulator
    Buyer -->|Trust Signal| Regulator
    Regulator -->|Selected Gradients| Asset
    Regulator -->|Rewards| Economy
    Asset -->|State Update| Auditor
````

## 🔌 How to Extend the Framework

### 1\. Adding a New Defense (Regulator)

To implement a new aggregation mechanism (e.g., a new robust mean estimation):

1.  **Create Class:** Inherit from the base `Aggregator` class in `marketplace/market_mechanism/aggregators/`.
2.  **Implement Interface:** Override the `run(gradients, trusted_signal)` method.
3.  **Register:** Add your class key to `common/factories.py`.
4.  **Config:** You can now use it in YAML: `aggregation.method: "my_new_defense"`.

### 2\. Adding a New Attack (Supply Side)

We decouple the *Payload* (atomic action) from the *Coordination* (group strategy).

  * **To add a Payload (e.g., a new Backdoor):** Add a function in `attack/attack_gradient_market/poison_attack/`.
  * **To add Coordination (e.g., a generic Sybil strategy):** Inherit from `SybilStrategy` and implement the logic to manipulate the vector distribution of the malicious coalition.

### 3\. Adding a New Dataset (Market Asset)

The framework uses a **Configuration-Driven Factory** to ensure attacks are dataset-agnostic.

1.  **Wrapper:** Create a lightweight wrapper in `common/datasets/` that handles loading and standardizing your data (Image/Text/Tabular).
2.  **Model:** Define the compatible architecture in `model/`.
3.  **Config:** Update `DataConfig` to recognize the new string identifier.

## 📊 Marketplace Metrics

Beyond standard accuracy, this framework allows you to plug in custom **Economic** and **Integrity** metrics.

  * **Mechanism Integrity:** Track `Malicious Selection Rate (MSR)` vs `Benign Selection Rate (BSR)` to measure revenue theft and collateral damage.
  * **Value-Selection Alignment:** The `ValuationEngine` runs in parallel to the training loop. You can plug in new "Ground Truth" auditors (e.g., Shapley Value approximations) in `marketplace/market_mechanism/valuation/`.

<!-- end list -->

```

-----
## License

This project is licensed under the [MIT License](https://www.google.com/search?q=./LICENSE).

