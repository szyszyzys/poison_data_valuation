import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. GLOBAL STYLE SETTINGS ---
def set_publication_style():
    """Sets the 'Big & Bold' professional style for all figures."""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.2)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'legend.title_fontsize': 18,
        'axes.linewidth': 2,
        'lines.linewidth': 3,
        'lines.markersize': 10,
        'figure.figsize': (8, 6), # Default size
    })

# --- 2. STANDARDIZED NAMES ---
# Map your code strings to pretty paper strings
PRETTY_NAMES = {
    # Defenses
    "fedavg": "FedAvg",
    "fltrust": "FLTrust",
    "martfl": "MARTFL",
    "skymask": "SkyMask",
    "skymask_small": "SkyMask", # Map the new small version to the main name

    # Attacks
    "min_max": "Min-Max",
    "min_sum": "Min-Sum",
    "labelflip": "Label Flip",
    "label_flip": "Label Flip",
    "fang_krum": "Fang-Krum",
    "fang_trim": "Fang-Trim",
    "scaling": "Scaling Attack",
    "dba": "DBA",
    "badnet": "BadNet",
    "pivot": "Targeted Pivot",
    "0. Baseline": "No Attack (Baseline)"
}

def format_label(label: str) -> str:
    """Returns the standardized paper name or title-cases it if unknown."""
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

# --- 3. CONSISTENT COLORS ---
# Assign specific colors so readers can track methods across figures
DEFENSE_COLORS = {
    "FedAvg": "#7f8c8d",   # Grey (Baseline)
    "FLTrust": "#3498db",  # Blue
    "MARTFL": "#2ecc71",   # Green
    "SkyMask": "#e74c3c",  # Red (Your Method - Highlighted)
}

# Attacks get a palette
ATTACK_PALETTE = "magma"

def get_defense_order():
    return ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def get_defense_palette():
    return DEFENSE_COLORS