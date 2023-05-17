import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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