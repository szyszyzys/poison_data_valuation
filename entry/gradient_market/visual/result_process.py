import ast
import ast
import logging
import logging
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import pandas as pd
import re
import re
import seaborn as sns
import seaborn as sns
from collections import Counter
from collections import Counter
from pathlib import Path
from pathlib import Path
from scipy.stats import spearmanr
from scipy.stats import spearmanr
from typing import Any, Dict, List, Optional, Tuple
from typing import Any, Dict, List, Optional, Tuple

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Plotting Style ---
sns.set_theme(style="whitegrid", palette="viridis")


def safe_literal_eval(val: Any) -> Any:
    """
    Safely evaluate strings to Python literals (list/dict/float) or return original.
    """
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                pass
        try:
            return float(s)
        except ValueError:
            return s
    return val


def load_all_results(base_dir: str, csv_filename: str = "round_results.csv") -> Dict[str, List[pd.DataFrame]]:
    """
    Traverse base_dir/<experiment>/<run> and load CSVs into DataFrames.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        logging.error(f"Base directory '{base_dir}' not found.")
        return {}

    all_results: Dict[str, List[pd.DataFrame]] = {}
    logging.info(f"Loading results from: {base_path}")

    for exp_path in sorted(base_path.iterdir()):
        if not exp_path.is_dir():
            continue
        exp_name = exp_path.name
        run_dfs: List[pd.DataFrame] = []
        logging.info(f"Processing experiment: {exp_name}")

        # Identify directories containing runs
        subdirs = [d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith(exp_name)]
        search_dirs = subdirs or [exp_path]

        for sd in search_dirs:
            for run_dir in sorted(sd.glob('run_*')):
                if not run_dir.is_dir():
                    continue
                csv_file = run_dir / csv_filename
                if not csv_file.is_file():
                    continue
                try:
                    df = pd.read_csv(csv_file)
                    if df.empty:
                        continue
                    df['run_id'] = run_dir.name
                    df['experiment_setup'] = exp_name
                    run_dfs.append(df)
                except Exception as e:
                    logging.error(f"Failed to load {csv_file}: {e}")
        if run_dfs:
            all_results[exp_name] = run_dfs
        else:
            logging.warning(f"No valid runs for experiment: {exp_name}")

    if not all_results:
        logging.error("No experiments loaded.")
    return all_results


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and extract metrics from raw run DataFrame.
    """
    logging.debug(f"Preprocessing columns: {df.columns.tolist()}")

    # --- Literal eval columns ---
    eval_cols = {
        'list': ['selected_sellers', 'seller_ids_all'],
        'dict': ['perf_global', 'perf_local', 'selection_rate_info', 'gradient_inversion_log']
    }
    for col in df.columns:
        if col in eval_cols['list'] + eval_cols['dict']:
            df[col] = df[col].apply(safe_literal_eval)

    # --- Ensure types ---
    for col in eval_cols['list']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    for col in eval_cols['dict']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else {})

    # --- Extract global metrics ---
    if 'perf_global' in df.columns:
        df['global_acc'] = df['perf_global'].apply(lambda d: d.get('accuracy', np.nan))
        df['global_loss'] = df['perf_global'].apply(lambda d: d.get('loss', np.nan))
        df['global_asr'] = df['perf_global'].apply(lambda d: d.get('attack_success_rate', np.nan))

    # --- Extract privacy metrics ---
    if 'gradient_inversion_log' in df.columns:
        def _get_metric(log: dict, key: str) -> float:
            return log.get('metrics', {}).get(key, np.nan)

        df['attack_psnr'] = df['gradient_inversion_log'].apply(lambda d: _get_metric(d, 'psnr'))
        df['attack_ssim'] = df['gradient_inversion_log'].apply(lambda d: _get_metric(d, 'ssim'))
        df['attack_label_acc'] = df['gradient_inversion_log'].apply(lambda d: _get_metric(d, 'label_accuracy'))

    # --- Numeric conversions ---
    for col in ['round_number', 'round_duration_sec', 'num_sellers_selected']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'round_number' in df.columns and not df['round_number'].dropna().empty:
        df['round_number'] = df['round_number'].astype(int)

    # --- Parse experiment name ---
    df['exp_adv_rate'] = 0.0
    if 'experiment_setup' in df.columns and isinstance(df['experiment_setup'].iloc[0], str):
        name = df['experiment_setup'].iloc[0]
        m = re.search(r'_adv(\d+)pct', name)
        if m:
            df['exp_adv_rate'] = int(m.group(1)) / 100.0

    # --- Identify sellers ---
    seller_ids = df['seller_ids_all'].iloc[0] if 'seller_ids_all' in df.columns and len(df) > 0 else []
    benign_ids = [s for s in seller_ids if isinstance(s, str) and s.startswith('bn_')]
    adv_ids = [s for s in seller_ids if isinstance(s, str) and s.startswith('adv_')]

    df['num_benign'] = len(benign_ids)
    df['num_malicious'] = len(adv_ids)
    df['benign_selected_count'] = df['selected_sellers'].apply(
        lambda sel: sum(1 for s in sel if s in benign_ids)
    ) if 'selected_sellers' in df.columns else 0
    df['malicious_selected_count'] = df['selected_sellers'].apply(
        lambda sel: sum(1 for s in sel if s in adv_ids)
    ) if 'selected_sellers' in df.columns else 0

    df['benign_selection_rate'] = (
            df['benign_selected_count'] / df['num_sellers_selected'].replace(0, np.nan)
    )
    df['malicious_selection_rate'] = (
            df['malicious_selected_count'] / df['num_sellers_selected'].replace(0, np.nan)
    )

    return df


def calculate_cost_of_convergence(df_run: pd.DataFrame, target_acc: float) -> Optional[float]:
    """
    Return cumulative #selected until global_acc >= target_acc, else NaN.
    """
    if 'global_acc' not in df_run.columns or 'num_sellers_selected' not in df_run.columns:
        return np.nan
    meets = df_run[df_run['global_acc'] >= target_acc]
    if meets.empty:
        return np.nan
    r0 = meets['round_number'].min()
    return df_run.loc[df_run['round_number'] <= r0, 'num_sellers_selected'].sum()


def calculate_fairness_differential(df_run: pd.DataFrame) -> Optional[float]:
    """
    Avg per-seller benign selection rate minus malicious.
    """
    n_rounds = df_run['round_number'].max() if 'round_number' in df_run else 0
    b = df_run['benign_selected_count'].sum()
    m = df_run['malicious_selected_count'].sum()
    nb = df_run['num_benign'].iloc[0] if 'num_benign' in df_run else 0
    nm = df_run['num_malicious'].iloc[0] if 'num_malicious' in df_run else 0
    if n_rounds <= 0 or nm == 0:
        return np.nan
    return (b / nb / n_rounds) - (m / nm / n_rounds)


def get_seller_divergence(sid: str, exp_name: str) -> Optional[float]:
    """Placeholder for divergence lookup."""
    m = re.search(r'_f([\d\.]+)_', sid)
    if m:
        return float(m.group(1))
    return None


def calculate_fairness_correlation(df_run: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """
    Spearman correlation between benign selection frequency and divergence.
    """
    if 'selected_sellers' not in df_run or 'seller_ids_all' not in df_run:
        return None
    benign_ids = [s for s in df_run['seller_ids_all'].iloc[0] if s.startswith('bn_')]
    counts = Counter(sum(df_run['selected_sellers'], []))
    freqs: List[float] = []
    divs: List[float] = []
    for bid in benign_ids:
        div = get_seller_divergence(bid, df_run['experiment_setup'].iloc[0])
        if div is None:
            continue
        freq = counts.get(bid, 0) / (df_run['round_number'].max() or 1)
        freqs.append(freq)
        divs.append(div)
    logging.debug(f"Correlation sample size: {len(freqs)}/{len(benign_ids)}")
    if len(freqs) < 2:
        return None
    rho, p = spearmanr(freqs, divs)
    return rho, p


def gini(x: np.ndarray) -> float:
    """Gini coefficient of array x."""
    x = np.array(x, dtype=float)
    if x.min() < 0:
        x -= x.min()
    if (x == 0).all():
        return 0.0
    x_sorted = np.sort(x)
    n = len(x)
    cum = np.cumsum(x_sorted)
    return (2 * np.sum((np.arange(1, n + 1) * x_sorted)) / (n * cum[-1]) - (n + 1) / n)


def calculate_gini_coefficient(df_run: pd.DataFrame) -> Optional[float]:
    """Gini over total selections per seller."""
    if 'seller_ids_all' not in df_run:
        return np.nan
    ids = df_run['seller_ids_all'].iloc[0]
    counts = Counter(sum(df_run['selected_sellers'], []))
    vals = [counts.get(s, 0) for s in ids]
    if len(vals) < 2:
        return 0.0
    return gini(np.array(vals))


def calculate_selection_entropy(df_run: pd.DataFrame) -> Optional[float]:
    """Normalized Shannon entropy of selection frequencies."""
    if 'seller_ids_all' not in df_run:
        return np.nan
    ids = df_run['seller_ids_all'].iloc[0]
    freq = np.array(list(Counter(sum(df_run['selected_sellers'], [])).values()), dtype=float)
    total = freq.sum()
    if total == 0:
        return 0.0
    p = freq / total
    p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    return H / np.log2(len(ids))


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard index between two sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def calculate_selection_stability(df_run: pd.DataFrame) -> Optional[float]:
    """Avg Jaccard between consecutive rounds."""
    sels = [set(s) for s in df_run.sort_values('round_number')['selected_sellers']]
    if len(sels) < 2:
        return np.nan
    js = [jaccard_similarity(sels[i], sels[i + 1]) for i in range(len(sels) - 1)]
    return float(np.mean(js))


def plot_metric_comparison(
        results: Dict[str, List[pd.DataFrame]],
        metric: str,
        title: str,
        xlabel: str = "Round",
        ylabel: Optional[str] = None,
        ci: str = 'sd',
        save_path: Optional[Path] = None
) -> None:
    """Line plot of a metric over rounds with mean ± CI."""
    all_dfs = []
    for name, runs in results.items():
        for df in runs:
            if metric in df:
                tmp = df[['round_number', metric]].copy()
                tmp['experiment'] = name
                all_dfs.append(tmp)
    if not all_dfs:
        logging.warning(f"No data for metric {metric}")
        return
    df_all = pd.concat(all_dfs, ignore_index=True).dropna()
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_all,
        x='round_number',
        y=metric,
        hue='experiment',
        estimator=np.mean,
        errorbar=ci
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel or metric)
    plt.legend(title='Experiment', loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_final_round_comparison(
        results: Dict[str, List[pd.DataFrame]],
        metric: str,
        title: str,
        higher_is_better: bool = True,
        save_path: Optional[Path] = None
) -> None:
    """Bar plot comparing final-round metric across experiments."""
    means, errs, names = [], [], []
    for name, runs in results.items():
        vals = []
        for df in runs:
            if metric in df:
                dfv = df.dropna(subset=[metric, 'round_number'])
                if not dfv.empty:
                    last = dfv.loc[dfv['round_number'].idxmax(), metric]
                    vals.append(last)
        if vals:
            names.append(name)
            means.append(np.mean(vals))
            errs.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
    if not names:
        logging.warning(f"No final data for {metric}")
        return
    idx = np.arange(len(names))
    palette = sns.color_palette(None, len(names))
    best = int(np.argmax(means) if higher_is_better else np.argmin(means))
    colors = [palette[i] if i != best else 'red' for i in range(len(names))]
    plt.figure(figsize=(10, 6))
    plt.bar(names, means, yerr=errs, capsize=5, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel(metric)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_scatter_comparison(
        results: Dict[str, List[pd.DataFrame]],
        x: str,
        y: str,
        title: str,
        save_path: Optional[Path] = None
) -> None:
    """Scatter of two final-round metrics across experiments."""
    rows = []
    for name, runs in results.items():
        xs, ys = [], []
        for df in runs:
            dfv = df.dropna(subset=[x, y, 'round_number'])
            if not dfv.empty:
                last = dfv.loc[dfv['round_number'].idxmax()]
                xs.append(last[x])
                ys.append(last[y])
        if xs and ys:
            rows.append({'experiment': name, x: np.mean(xs), y: np.mean(ys)})
    if not rows:
        logging.warning(f"No data for scatter {x} vs {y}")
        return
    dfp = pd.DataFrame(rows)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=dfp, x=x, y=y, hue='experiment', s=100)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_aggregated_metric_bar(
        df: pd.DataFrame,
        metric_base: str,
        title: str,
        ylabel: str,
        higher_is_better: bool,
        save_path: Path
) -> None:
    """Bar plot of aggregated marketplace metrics (mean ± std)."""
    mean_col = f"{metric_base}_mean"
    std_col = f"{metric_base}_std"
    if mean_col not in df.columns:
        logging.warning(f"Missing column {mean_col}")
        return
    dfx = df.set_index('experiment_setup')[[mean_col, std_col]].dropna()
    if dfx.empty:
        return
    names = dfx.index.tolist()
    means = dfx[mean_col].values
    errs = dfx[std_col].values
    best = int(np.nanargmax(means) if higher_is_better else np.nanargmin(means))
    palette = sns.color_palette(None, len(names))
    colors = [palette[i] if i != best else 'red' for i in range(len(names))]
    plt.figure(figsize=(10, 6))
    plt.bar(names, means, yerr=errs, capsize=5, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def safe_literal_eval(val: Any) -> Any:
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                pass
        try:
            return float(s)
        except ValueError:
            return s
    return val


def load_all_results(base_dir: str, csv_filename: str = "round_results.csv") -> Dict[str, List[pd.DataFrame]]:
    base_path = Path(base_dir)
    if not base_path.is_dir():
        logging.error(f"Base directory '{base_dir}' not found.")
        return {}

    all_results: Dict[str, List[pd.DataFrame]] = {}
    logging.info(f"Loading results from: {base_path}")

    for exp_path in sorted(base_path.iterdir()):
        if not exp_path.is_dir(): continue
        exp_name = exp_path.name
        run_dfs: List[pd.DataFrame] = []
        logging.info(f"Processing experiment: {exp_name}")
        subdirs = [d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith(exp_name)]
        search_dirs = subdirs or [exp_path]
        for sd in search_dirs:
            for run_dir in sorted(sd.glob('run_*')):
                csv_file = run_dir / csv_filename
                if not csv_file.is_file(): continue
                try:
                    df = pd.read_csv(csv_file)
                    if df.empty: continue
                    df['run_id'] = run_dir.name
                    df['experiment_setup'] = exp_name
                    run_dfs.append(df)
                except Exception as e:
                    logging.error(f"Failed to load {csv_file}: {e}")
        if run_dfs:
            all_results[exp_name] = run_dfs
        else:
            logging.warning(f"No valid runs for experiment: {exp_name}")
    if not all_results:
        logging.error("No experiments loaded.")
    return all_results


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug(f"Preprocessing columns: {df.columns.tolist()}")
    # Literal-eval
    lists = ['selected_sellers', 'seller_ids_all']
    dicts = ['perf_global', 'perf_local', 'selection_rate_info', 'gradient_inversion_log']
    for col in lists + dicts:
        if col in df.columns:
            df[col] = df[col].apply(safe_literal_eval)
    for col in lists:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    for col in dicts:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else {})
    # Extract global metrics
    if 'perf_global' in df.columns:
        df['global_acc'] = df['perf_global'].apply(lambda d: d.get('accuracy', np.nan))
        df['global_asr'] = df['perf_global'].apply(lambda d: d.get('attack_success_rate', np.nan))
    # Extract privacy
    if 'gradient_inversion_log' in df.columns:
        df['attack_psnr'] = df['gradient_inversion_log'].apply(lambda d: d.get('metrics', {}).get('psnr', np.nan))
    # Numerics
    for col in ['round_number', 'num_sellers_selected']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'round_number' in df.columns and not df['round_number'].dropna().empty:
        df['round_number'] = df['round_number'].astype(int)
    # Parse adv rate
    df['exp_adv_rate'] = 0.0
    if 'experiment_setup' in df.columns:
        m = re.search(r'_adv(\d+)pct', df['experiment_setup'].iloc[0] or '')
        if m: df['exp_adv_rate'] = int(m.group(1)) / 100.0
    # Sellers
    ids = df['seller_ids_all'].iloc[0] if 'seller_ids_all' in df.columns and len(df) > 0 else []
    benign = [s for s in ids if isinstance(s, str) and s.startswith('bn_')]
    adv = [s for s in ids if isinstance(s, str) and s.startswith('adv_')]
    df['num_benign'], df['num_malicious'] = len(benign), len(adv)
    if 'selected_sellers' in df.columns:
        df['benign_selected_count'] = df['selected_sellers'].apply(lambda sel: sum(1 for s in sel if s in benign))
        df['malicious_selected_count'] = df['selected_sellers'].apply(lambda sel: sum(1 for s in sel if s in adv))
        df['benign_selection_rate'] = df['benign_selected_count'] / df['num_sellers_selected'].replace(0, np.nan)
        df['malicious_selection_rate'] = df['malicious_selected_count'] / df['num_sellers_selected'].replace(0, np.nan)
    return df


# Metric calculators and plotting omitted for brevity; assume they are defined above

if __name__ == "__main__":
    RESULTS_BASE_DIR = "./experiment_results"
    OUTPUT_DIR = Path("./analysis_plots_marketplace")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_ACC = 0.75
    # Load & preprocess
    raw = load_all_results(RESULTS_BASE_DIR)
    processed = {n: [preprocess_data(df) for df in dfs] for n, dfs in raw.items()}
    # Marketplace summary
    summary = []
    for n, runs in processed.items():
        vals = {'coc': [], 'fairness_diff': [], 'gini': [], 'entropy': [], 'stability': [], 'fair_corr_rho': []}
        for df in runs:
            from math import isnan

            # assume functions return nan if not available
            vals['coc'].append(calculate_cost_of_convergence(df, TARGET_ACC))
            vals['fairness_diff'].append(calculate_fairness_differential(df))
            vals['gini'].append(calculate_gini_coefficient(df))
            vals['entropy'].append(calculate_selection_entropy(df))
            vals['stability'].append(calculate_selection_stability(df))
            corr = calculate_fairness_correlation(df)
            if corr: vals['fair_corr_rho'].append(corr[0])
        rec = {'experiment_setup': n}
        for k, arr in vals.items():
            arr_np = np.array(arr, dtype=float)
            rec[f"{k}_mean"] = np.nanmean(arr_np)
            rec[f"{k}_std"] = np.nanstd(arr_np)
        summary.append(rec)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "marketplace_metrics_summary.csv", index=False)
    # Define experiment groups
    all_keys = list(processed.keys())
    exp_groups = {
        "Baselines": [k for k in all_keys if k.startswith("baseline_")],
        "Backdoor_Attacks_All": [k for k in all_keys if k.startswith("backdoor_")],
        "LabelFlip_Attacks_All": [k for k in all_keys if k.startswith("label_flip_")],
        "Compare_Agg_CIFAR_Backdoor_30pct": [k for k in all_keys if
                                             k.startswith('backdoor_cifar_') and '_adv30pct' in k],
        "Compare_AdvRate_CIFAR_Backdoor_FLTrust": [k for k in all_keys if k.startswith('backdoor_cifar_fltrust_')]
    }
    exp_groups = {g: ks for g, ks in exp_groups.items() if ks}
    # Plot per group
    for g, keys in exp_groups.items():
        grp_dir = OUTPUT_DIR / g
        grp_dir.mkdir(exist_ok=True)
        subset = {k: processed[k] for k in keys}
        # Standard metrics
        plot_metric_comparison(subset, 'global_acc', f"{g} - Global Acc", save_path=grp_dir / "global_acc.png")
        plot_final_round_comparison(subset, 'global_acc', f"{g} - Final Global Acc",
                                    save_path=grp_dir / "final_global_acc.png")
        plot_metric_comparison(subset, 'global_asr', f"{g} - Global ASR", save_path=grp_dir / "global_asr.png")
        plot_final_round_comparison(subset, 'global_asr', f"{g} - Final Global ASR", higher_is_better=False,
                                    save_path=grp_dir / "final_global_asr.png")
        plot_metric_comparison(subset, 'attack_psnr', f"{g} - Attack PSNR", save_path=grp_dir / "attack_psnr.png")
        plot_final_round_comparison(subset, 'attack_psnr', f"{g} - Final Attack PSNR",
                                    save_path=grp_dir / "final_attack_psnr.png")
        # Marketplace aggregated bars
        group_market = summary_df[summary_df['experiment_setup'].isin(keys)]
        plot_aggregated_metric_bar(group_market, 'coc', f"{g} - Cost of Convergence (>= {TARGET_ACC})",
                                   "Total Selections", False, grp_dir / "coc.png")
        plot_aggregated_metric_bar(group_market, 'fairness_diff', f"{g} - Fairness Differential",
                                   "BenignRate - MalRate", True, grp_dir / "fairness_diff.png")
        plot_aggregated_metric_bar(group_market, 'gini', f"{g} - Payment Gini", "Gini (0=Equal)", False,
                                   grp_dir / "gini.png")
        plot_aggregated_metric_bar(group_market, 'entropy', f"{g} - Selection Entropy", "Norm Entropy", True,
                                   grp_dir / "entropy.png")
        plot_aggregated_metric_bar(group_market, 'stability', f"{g} - Selection Stability", "Avg Jaccard", True,
                                   grp_dir / "stability.png")
        plot_aggregated_metric_bar(group_market, 'fair_corr_rho', f"{g} - Fairness Corr (Rho)", "Spearman Rho", False,
                                   grp_dir / "fairness_corr_rho.png")
        # Scatter plots
        plot_scatter_comparison(subset, 'global_asr', 'global_acc', f"{g}: ASR vs Acc",
                                save_path=grp_dir / "scatter_asr_acc.png")
        if any('malicious_selection_rate' in df.columns for runs in subset.values() for df in runs):
            plot_scatter_comparison(subset, 'malicious_selection_rate', 'global_acc', f"{g}: Mal Select Rate vs Acc",
                                    save_path=grp_dir / "scatter_mal_rate_acc.png")
    logging.info("All plots generated.")
