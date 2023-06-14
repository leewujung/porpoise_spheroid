import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.gridspec import GridSpec
from scipy import interpolate, signal


class Echogram:
    def __init__(
        self,
        df_dtag,
        dtag_wav_path,
        sound_speed=1500,
        echo_range=25,
    ) -> None:
        """
        Parameters
        ----------
        df_dtag
            A dataframe containing labeled Dtag clicks from a particular trial
        dtag_wav_path
            Path to the Dtag wav file from a particular trial
            This file is preferably bandpass filtered at 110-145 kHz for cleaner echogram
        """

        # Constants
        self.sound_speed = sound_speed  # [m/s]
        self.echo_range = echo_range  # [m]

        self.dtag_wav_path = dtag_wav_path
        self.df_dtag = df_dtag

        # Load dtag signal
        self.sig_dtag, self.fs_dtag = sf.read(self.dtag_wav_path)
        self.sig_dtag_t = np.arange(len(self.sig_dtag)) / self.fs_dtag
        self.sig_dtag_env = np.abs(signal.hilbert(self.sig_dtag))

        # Init other params
        self.echo_mtx_raw = None
        self.echo_mtx_fine = None
        self.range_vec = None
        self.time_vec = None
        self.time_fine = None

    def build_echogram(self, time_step=20e-3):
        """
        Build echogram and interpolate to allow plotting correct time using imshow.
        """
        echo_smpl_wanted = np.ceil(
            self.echo_range * 2 / self.sound_speed * self.fs_dtag
        ).astype(int)

        # Clean up duplicates in df_dtag
        self.df_dtag.drop_duplicates(subset=["sample_loc"], inplace=True)

        # Get echo_mtx
        clk_idx_drop = len(self.df_dtag)
        echo_mtx = np.ones((echo_smpl_wanted, len(self.df_dtag)))
        idx_cut = np.arange(echo_smpl_wanted)
        for clk_idx, start_smpl in enumerate(self.df_dtag["sample_loc"]):
            if start_smpl + idx_cut[-1] >= self.sig_dtag_env.size:
                clk_idx_drop = clk_idx
                break
            else:
                echo_mtx[:, clk_idx] = self.sig_dtag_env[start_smpl + idx_cut]

        # Interpolate echo_mtx for imshow
        echo_mtx = echo_mtx[:, :clk_idx_drop]
        self.time_vec = self.df_dtag["time_corrected"].values[:clk_idx_drop]
        self.range_vec = (
            np.arange(echo_smpl_wanted) / self.fs_dtag * self.sound_speed / 2
        )
        f_echo_mtx = interpolate.RectBivariateSpline(
            x=self.range_vec, y=self.time_vec, z=20 * np.log10(echo_mtx)
        )
        self.time_fine = np.arange(self.time_vec[0], self.time_vec[-1], time_step)

        self.echo_mtx_raw = echo_mtx
        self.echo_mtx_fine = f_echo_mtx(self.range_vec, self.time_fine)

    def plot_time_on_axis(self, df_hydro_ch0, df_hydro_ch1, df_track, ax=None):
        """
        Plot hydrophone click detections and valid track times.
        """
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(14, 4))
            ax_opt = ax  # for ref later on whether to return fig, ax
        else:
            ax_opt = 1

        # drop invalid time points on df_track
        df_track_wanted = df_track.dropna(
            subset=["DTAG_X", "ROSTRUM_X"], how="any"
        ).copy()

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
            self.df_dtag["time_corrected"],
            np.ones(len(self.df_dtag)) * 2,
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
        ax[1].legend(fontsize=12)

        if ax_opt is None:
            return fig, ax

    def plot_echogram_axis(self, interpolation="antialiased", ax=None):
        """
        Plot echogram on an axis.
        """
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(14, 8))
            ax_opt = ax  # for ref later on whether to return fig, ax
        else:
            ax_opt = 1

        ax.imshow(
            self.echo_mtx_fine,
            aspect="auto",
            interpolation=interpolation,
            vmin=-90,
            vmax=-60,
            cmap="jet",
            origin="lower",
            extent=(
                self.time_vec[0],
                self.time_vec[-1],
                self.range_vec[0],
                self.range_vec[-1],
            ),
        )
        ax.grid()

        if ax_opt is None:
            return fig, ax

    def plot_echogram_with_detection(
        self,
        df_hydro_ch0,
        df_hydro_ch1,
        df_track,
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
        self.plot_echogram_axis(interpolation=interpolation, ax=ax_echogram)

        # Plot hydrophone click detection and track points
        self.plot_time_on_axis(
            df_hydro_ch0, df_hydro_ch1, df_track, ax=[ax_RL, ax_track]
        )

        if time_lim:
            ax_echogram.set_xlim(time_lim)
        if range_lim:
            ax_echogram.set_ylim(range_lim)

        plt.show()

        return fig, [ax_echogram, ax_RL, ax_track]
