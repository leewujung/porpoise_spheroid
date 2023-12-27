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
    "AR=2.9 - AR=1.3",
    "AR=2.9 - AR=1.1",
    "AR=1.3 - AR=1.1",
] * 2

STAT_CONTRAST_RATIO_AR = [
    "AR=2.9 / AR=1.3",
    "AR=2.9 / AR=1.1",
    "AR=1.3 / AR=1.1",
] * 2

STAT_CONTRAST_STR_AR = [
    "TC: AR=2.9 vs AR=1.3",
    "TC: AR=2.9 vs AR=1.1",
    "TC: AR=1.3 vs AR=1.1",
    "CT: AR=2.9 vs AR=1.3",
    "CT: AR=2.9 vs AR=1.1",
    "CT: AR=1.3 vs AR=1.1",
]

# Between AR=1.3 clusters
STAT_CONTRAST_DIFF = [
    "Straight - (Curved-1)",
    "Straight - (Curved-2)",
    "(Curved-1) - (Curved-2)",
] * 2

STAT_CONTRAST_RATIO = [
    "Straight / (Curved-1)",
    "Straight / (Curved-2)",
    "(Curved-1) / (Curved-2)",
] * 2

STAT_CONTRAST_STR = [
    "R+/Straight vs R+/Curved-1",
    "R+/Straight vs R+/Curved-2",
    "R+/Curved-1 vs R+/Curved-2",
    "L+/Straight vs L+/Curved-1",
    "L+/Straight vs L+/Curved-2",
    "L+/Curved-1 vs L+/Curved-2",
]


# =======================================
# Functions for annotating p values
# =======================================
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


def get_p_star(p_val, p_range=[0.05, 0.01, 0.001, 0.0001]):
    """
    Return star annotation for a p value.
    
    The levels are:
        - ns    P > 0.05
        - *     P ≤ 0.05
        - **    P ≤ 0.01
        - ***   P ≤ 0.001
        - ****  P ≤ 0.0001
    """
    if p_val <= p_range[3]:
        return r"$\asterisk\asterisk\asterisk\asterisk$"
        # return "****"
    elif p_val <= p_range[2]:
        return r"$\asterisk\asterisk\asterisk$"
        # return "***"
    elif p_val <= p_range[1]:
        return r"$\asterisk\asterisk$"
        # return "**"
    elif p_val <= p_range[0]:
        return r"$\asterisk$"
        # return "*"
    else:
        return "ns"


def add_TCCT_text(ax: plt.Axes, y_height: float=1.02):
    ylim = ax.get_ylim()
    ax.text(1, ylim[1]*y_height, "TC", fontsize=14, ha="center", fontweight="bold")
    ax.text(4, ylim[1]*y_height, "CT", fontsize=14, ha="center", fontweight="bold")


def annotate_p_val_cluster(
    ax: plt.Axes,
    df_stat: pd.DataFrame,
    star_only: bool=True,
    ratio: bool=False,
    fontsize: float=10,
    vert_h: np.ndarray=np.array([0.92 , 0.86, 0.80, 0.92 , 0.86, 0.80])
    # vert_h: np.ndarray=np.array([0.9 , 0.82, 0.74, 0.9 , 0.82, 0.74])
):
    """
    ax : plt.Axes
        axis to annotate
    df_stat : pd.DataFrame
        dataframe holding p values
    star_only : bool
        use * to denote p value range
    ratio : bool
        whether contrast is a ratio (from glm)
    vert_h : np.ndarray
        vertical offset of p value annotation
    """
    # Get p values
    constrast_str = STAT_CONTRAST_RATIO if ratio else STAT_CONTRAST_DIFF
    p_val_cluster = [
        get_p_val_group(df_stat, STAT_POSITION[idx], constrast_str[idx])
        for idx in np.arange(6)
    ]
    if star_only:
        p_val_cluster = [get_p_star(p) for p in p_val_cluster]
    
    # Scale vertical position
    ylim = ax.get_ylim()
    vert_h = vert_h * (ylim[1] - ylim[0]) + ylim[0]

    # Annotate
    for idx in np.arange(6):
        cmp_str = p_val_cluster[idx] if star_only else f"{p_val_cluster[idx]:2.2E}"        
        ax.plot(STAT_PLOT_XPOS[idx], vert_h[idx] * np.ones(2), color="k", lw=0.5)
        ax.text(
            np.mean(STAT_PLOT_XPOS[idx]), vert_h[idx], cmp_str,
            ha="center", va="bottom", fontsize=fontsize
        )            


def annotate_p_val_spheroid(
    ax: plt.Axes,
    df_stat: pd.DataFrame,
    ratio: bool=False,
    star_only: bool=True,
    fontsize: float=10,
    vert_h: np.ndarray=np.array([0.92 , 0.86, 0.80, 0.92 , 0.86, 0.80])
    # vert_h: np.ndarray=np.array([0.9 , 0.82, 0.74, 0.9 , 0.82, 0.74])
):
    """
    ax : plt.Axes
        axis to annotate
    df_stat : pd.DataFrame
        dataframe holding p values
    star_only : bool
        use * to denote p value range
    ratio : bool
        whether contrast is a ratio (from glm)
    vert_h : np.ndarray
        vertical offset of p value annotation
    """
    # Get p values
    constrast_str = STAT_CONTRAST_RATIO_AR if ratio else STAT_CONTRAST_DIFF_AR
    p_val_spheroid = [
        get_p_val_group(df_stat, STAT_POSITION[idx], constrast_str[idx])
        for idx in np.arange(6)
    ]
    if star_only:
        p_val_spheroid = [get_p_star(p) for p in p_val_spheroid]
    
    # Scale vertical position
    ylim = ax.get_ylim()
    vert_h = vert_h * (ylim[1] - ylim[0]) + ylim[0]

    # Annotate
    for idx in np.arange(6):
        ax.plot(STAT_PLOT_XPOS[idx], vert_h[idx] * np.ones(2), color="k", lw=0.5)
        cmp_str = p_val_spheroid[idx] if star_only else f"{p_val_spheroid[idx]:2.2E}"
        ax.text(
            np.mean(STAT_PLOT_XPOS[idx]), vert_h[idx], cmp_str,
            ha="center", va="bottom", fontsize=fontsize
        )


def annotate_p_val_position(
    ax: plt.Axes, df_stat: pd.DataFrame, star_only: bool=True, fontsize: float=10, y_height: float=1.02
):
    """
    ax : plt.Axes
        axis to annotate
    df_stat : pd.DataFrame
        dataframe holding p values
    """
    p_val_position = get_p_val_position(df_stat)
    if star_only:
        p_val_position = get_p_star(p_val_position)

    ax.annotate('', xy=(0.32, y_height), xycoords='axes fraction', xytext=(0.68, 1.02),
        arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    
    ylim = ax.get_ylim()
    cmp_str = p_val_position if star_only else f"{p_val_position:2.2E}"
    ax.text(
        2.5, ylim[1]*y_height, cmp_str,
        ha="center", va="bottom", fontsize=fontsize
    )


def annotate_p_val_scan(
    ax: plt.Axes,
    df_stat_position: pd.DataFrame,
    df_stat_group: pd.DataFrame,
    group: str="cluster",
    ratio: bool=False,
    star_only: bool=True,
    fontsize: float=11,
    vert_text: float=-0.3,
    vert_text_gap: float=0.1,
    horz_text_left: float=0,
    horz_text_right: float=6,
    **kwargs
):
    """
    ax : plt.Axes
        axis to annotate
    df_stat_position : pd.DataFrame
        dataframe holding p values for TC vs CT comparison
    df_stat_group : pd.DataFrame
        dataframe holding p values for comparisons across spheroids (if group="spheroid")
        or across AR=1.3 clusters (if group="cluster")
    star_only : bool
        use * to denote p value range
    ratio : bool
        whether contrast is a ratio (from glm)
    """
    # Get p values
    if group == "cluster":
        constrast_str = STAT_CONTRAST_RATIO if ratio else STAT_CONTRAST_DIFF
    elif group == "spheroid":
        constrast_str = STAT_CONTRAST_RATIO_AR if ratio else STAT_CONTRAST_DIFF_AR
    else:
        raise ValueError(f"{group} is not allowed as the comparison group specifier!")

    p_val_group = [
        get_p_val_group(df_stat_group, STAT_POSITION[idx], constrast_str[idx])
        for idx in np.arange(6)
    ]
    p_val_position = get_p_val_position(df_stat_position)
    
    if star_only:
        p_val_group = [get_p_star(p) for p in p_val_group]
        p_val_position = get_p_star(p_val_position)

    # TC vs CT
    tcct_str = (
        f"R+ v. L+: {p_val_position}" if star_only
        else f"R+ v. C+: {p_val_position:2.2E}"
    )
    ax.text(
        horz_text_left, vert_text, tcct_str,
        ha="left", va="center", fontsize=fontsize
    )

    # TC comparisons
    contrast_str = STAT_CONTRAST_STR if group == "cluster" else STAT_CONTRAST_STR_AR
    for idx in np.arange(3):
        cmp_str = (
            f"{contrast_str[idx]}: {p_val_group[idx]}" if star_only
            else f"{contrast_str[idx]}: {p_val_group[idx]:2.2E}"
        )
        ax.text(
            horz_text_left, vert_text - vert_text_gap*(idx+1), cmp_str,
            ha="left", va="center", fontsize=fontsize
        )
    # CT comparisons
    for idx in np.arange(3):
        cmp_str = (
            f"{contrast_str[idx+3]}: {p_val_group[idx+3]}" if star_only
            else f"{contrast_str[idx+3]}: {p_val_group[idx+3]:2.2E}"
        )
        ax.text(
            horz_text_right, vert_text - vert_text_gap*(idx+1), cmp_str,
            ha="left", va="center", fontsize=fontsize
        )


# =======================================
# Functions for other plots
# =======================================
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
