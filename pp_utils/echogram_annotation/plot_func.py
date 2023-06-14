"""
Functions in here are largely identical those in class `Echogram`,
just modified to suit independent use for the range marking GUI.
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def plot_time_on_axis(df_dtag, df_hydro_ch0, df_hydro_ch1, df_track, ax=None):
    """
    Plot hydrophone click detections and valid track times.
    """
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(14, 4))
        ax_opt = ax  # for ref later on whether to return fig, ax
    else:
        ax_opt = 1

    # drop invalid time points on df_track
    df_track_wanted = df_track.dropna(subset=["DTAG_X", "ROSTRUM_X"], how="any").copy()

    # RL
    ax[0].plot(
        df_hydro_ch0["time_corrected"],
        df_hydro_ch0["RL"],
        marker="o",
        markerfacecolor="none",
        markersize=4,
        alpha=0.6,
        label="Ch0: target",
    )
    ax[0].plot(
        df_hydro_ch1["time_corrected"],
        df_hydro_ch1["RL"],
        marker="o",
        markerfacecolor="none",
        markersize=4,
        alpha=0.6,
        label="Ch1: clutter",
    )
    ax[0].set_ylabel("Received level\nSPL (dB)", fontsize=12)
    ax[0].legend(fontsize=12)

    # Viz for click recorded time
    ax[1].plot(
        df_hydro_ch0["time_corrected"],
        np.ones(len(df_hydro_ch0)) * 0,
        marker=".",
        markersize=6,
        alpha=0.3,
        ls="none",
        label="Ch0: target",
    )
    ax[1].plot(
        df_hydro_ch1["time_corrected"],
        np.ones(len(df_hydro_ch1)) * 1,
        marker=".",
        markersize=6,
        alpha=0.3,
        ls="none",
        label="Ch1: clutter",
    )
    ax[1].plot(
        df_dtag["time_corrected"],
        np.ones(len(df_dtag)) * 2,
        marker=".",
        markersize=6,
        alpha=0.3,
        ls="none",
        label="Dtag",
    )
    ax[1].plot(
        df_track_wanted["time_corrected"],
        np.ones(len(df_track_wanted)) * 3,
        marker=".",
        markersize=6,
        alpha=0.3,
        ls="none",
        label="track",
    )
    ax[1].set_ylim(-0.5, 3.5)
    ax[1].set_xlabel("Time corrected (sec)", fontsize=12)
    ax[1].set_ylabel("Data exists", fontsize=12)
    ax[1].legend(fontsize=12)

    if ax_opt is None:
        return fig, ax


def plot_echogram_axis(
    echogram_mtx, time_vec, range_vec, interpolation="antialiased", ax=None
):
    """
    Plot echogram on an axis.
    """
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(14, 8))
        ax_opt = ax  # for ref later on whether to return fig, ax
    else:
        ax_opt = 1

    ax.imshow(
        echogram_mtx,
        aspect="auto",
        interpolation=interpolation,
        vmin=-90,
        vmax=-60,
        cmap="jet",
        origin="lower",
        extent=(
            time_vec[0],
            time_vec[-1],
            range_vec[0],
            range_vec[-1],
        ),
    )
    ax.grid()
    ax.set_ylabel("Echo range (m)", fontsize=12)

    if ax_opt is None:
        return fig, ax


def plot_echogram_with_detection(
    df_dtag,
    df_hydro_ch0,
    df_hydro_ch1,
    df_track,
    echogram_mtx,
    time_vec,
    range_vec,
    time_lim=None,
    range_lim=None,
    interpolation="antialiased",
    figsize=(14, 12),
):
    """
    Plot echogram along with hydrophone click detections and valid track times.

    Returns
    -------
    Figure handle and a list of axes in this final outputs,
    in the form of (fig, [ax_echogram, ax_RL, ax_track])
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 1, figure=fig)
    ax_echogram = fig.add_subplot(gs[:4, :])
    ax_RL = fig.add_subplot(gs[4, :], sharex=ax_echogram)
    ax_track = fig.add_subplot(gs[5, :], sharex=ax_echogram)

    # Plot echogram
    plot_echogram_axis(
        echogram_mtx, time_vec, range_vec, interpolation=interpolation, ax=ax_echogram
    )

    # Plot hydrophone click detection and track points
    plot_time_on_axis(
        df_dtag, df_hydro_ch0, df_hydro_ch1, df_track, ax=[ax_RL, ax_track]
    )

    if time_lim:
        ax_echogram.set_xlim(time_lim)
    if range_lim:
        ax_echogram.set_ylim(range_lim)

    # Only showing until touching time
    touch_time = df_hydro_ch0[df_hydro_ch0["before_touch"]]["time_corrected"].values[-1]
    plt.xlim(time_vec[0], touch_time)

    plt.show()

    return fig, [ax_echogram, ax_RL, ax_track]
