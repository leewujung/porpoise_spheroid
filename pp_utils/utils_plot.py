import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STAT_FILENAME_MIDDLE = {
    "A": "A_decision_time",
    "B1": "B1_decision_range_target",
    "B2": "B2_decision_range_clutter",
    "C": "C_buzz_time",
    "D": "D_buzz_range",
    "E": "E_scan_ch0",
    "F": "F_scan_ch1",
}

STAT_PLOT_XPOS = [[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]]

STAT_POSITION = ["TC"] * 3 + ["CT"] * 3


# Between spheroids
STAT_CONTRAST_DIFF_AR = [
    "AR=1.3 - AR=2.9",
    "AR=1.1 - AR=2.9",
    "AR=1.1 - AR=1.3",
] * 2

STAT_CONTRAST_RATIO_AR = [
    "AR=1.3 / AR=2.9",
    "AR=1.1 / AR=2.9",
    "AR=1.1 / AR=1.3",
] * 2

STAT_CONTRAST_STR_AR = [
    "TC: AR=2.9 v. AR=1.3: p=",
    "TC: AR=2.9 v. AR=1.1: p=",
    "TC: AR=1.3 v. AR=1.1: p=",
    "CT: AR=2.9 v. AR=1.3: p=",
    "CT: AR=2.9 v. AR=1.1: p=",
    "CT: AR=1.3 v. AR=1.1: p=",
]

# Between AR=1.3 clusters
STAT_CONTRAST_DIFF = [
    "(Curved-1) - Straight",
    "(Curved-2) - Straight",
    "(Curved-1) - (Curved-2)",
] * 2

STAT_CONTRAST_RATIO = [
    "(Curved-1) / Straight",
    "(Curved-2) / Straight",
    "(Curved-1) / (Curved-2)",
] * 2

STAT_CONTRAST_STR = [
    "TC-Straight v. TC-Curved-1: p=",
    "TC-Straight v. TC-Curved-2: p=",
    "TC-Curved-1 v. TC-Curved-2: p=",
    "CT-Straight v. CT-Curved-1: p=",
    "CT-Straight v. CT-Curved-2: p=",
    "CT-Curved-1 v. CT-Curved-2: p=",
]


# Functions for annotating p values
def get_p_val_position(df: pd.DataFrame):
    """
    Get p value from TC vs CT position contrast
    """
    return df["p.value"].values[0]


def get_p_val_group(df: pd.DataFrame, position: str, contrast: str):
    """
    Get p values from contrasts between clusters
    """
    return df[
        (df["position"] == position)
        & (df["contrast"] == contrast)
    ]["p.value"].values[0]


# Functions for other plots
def plot_track(axx: plt.Axes, df: pd.DataFrame, color: str, alpha=0.1, lw=2):
    axx.plot(
        df[df["before_touch"]]["DTAG_X"],
        df[df["before_touch"]]["DTAG_Y"],
        alpha=alpha, color=color, lw=lw
    )


def plot_vio(
    axx: plt.Axes, vals: np.ndarray, seq: int,
    color: str, widths=0.5, alpha=0.3, qbar_color="k", qbar_alpha=0.3
):    
    vals = vals[~np.isnan(vals)]  # remove NaNs
    
    q1 = np.quantile(vals, 0.25)
    q3 = np.quantile(vals, 0.75)
    vio = axx.violinplot(
        vals, [seq], widths=widths,
        showmeans=False, showmedians=False, showextrema=False
    )
    vio["bodies"][0].set_facecolor(color)
    vio["bodies"][0].set_edgecolor("k")
    vio["bodies"][0].set_alpha(alpha)
    axx.plot(seq*np.ones(2), [q1, q3], lw=3, color=qbar_color, alpha=qbar_alpha)


def add_jitter(d: np.ndarray, width=0.25):
    np.random.seed(100)
    return d + np.random.uniform(low=-width/2, high=width/2, size=(1, d.size))


def plot_jitter(
    axx: plt.Axes, x_pos: float, vals: np.ndarray,
    color: str, width=0.25, **kwargs
):
    axx.plot(
        add_jitter(x_pos * np.ones_like(vals), width).T,
        vals,
        ls="none", color=color, **kwargs
    )

