from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from .core import ANGLE_MAP


def select_clicks(
    df_clk: pd.DataFrame,
    time_cut: Union[str, int, float] = "before_touch",
    threshold: dict = {"ICI": 13e-3, "SNR": 25},
):
    """
    Select clicks to be used to estimate ensonification angles.

    The criteria include ICI threshold, SNR threshold, and a time constraint.

    Parameters
    ----------
    df_clk
        A dataframe with matched clicks
    time_cut
        Time constraints.
        A number: before a particular time point (time_corrected) in the trial
        "before_touch": all clicks before the animal touched a target
        "no_position": all clicks before there was video track
    threshold
        A dictionary containign the ICI and SNR threshold

    Examples
    --------
    Use this function to generate df_click_all to initialize HydroLocalizer:
    >>> ti = get_trial_ti(trial_idx=trial_idx)
    >>> ti.get_track_hydro_features()

    >>> # Select clicks
    >>> df_hydro_ch0 = _filter(ti.tp.df_hydro_ch0, time_cut=1441).copy()
    >>> df_hydro_ch1 = _filter(ti.tp.df_hydro_ch1, time_cut=1441).copy()

    >>> # Match clicks
    >>> df_click_all = hydro_clicks.match_clicks(df_hydro_ch0, df_hydro_ch1)

    >>> # only keep matching clicks
    >>> df_click_all = df_click_all.dropna(subset=["the_other_ch_index"])
    """
    # Set time constraint
    if time_cut == "before_touch":
        time_sel = df_clk["before_touch"]
    elif time_cut == "no_position":
        time_sel = df_clk["DTAG_X"].isnull()
    else:
        time_sel = df_clk["time_corrected"] <= time_cut

    return df_clk[
        (df_clk["ICI"] > threshold["ICI"])  # regular clicks
        & (df_clk["SNR"] > threshold["SNR"])  # high SNR
        & time_sel  # time constraint
    ]


class HydroLocalizer:
    def __init__(
        self,
        df_click_all: pd.DataFrame,
        wav_file_path: Union[Path, str],
        scenario: str,
        threshold_peak_height=0.5,
        sound_speed=1500,
    ) -> None:
        """
        Parameters
        ----------
        df_click_all
            A dataframe containing matched clicks from a particular trial
        wav_fil_path
            Path to the hydrophone recording file from a particular trial
        scenario
            Experimental scenario (e.g., "TC3")
        """

        # Store click selection parameters
        self.threshold_peak_height = threshold_peak_height

        # Other params
        self.scenario = scenario
        self.sound_speed = sound_speed

        # Load and select clicks for localization
        self.df_click_all = df_click_all
        self.df_ch0 = df_click_all[df_click_all["CH"] == 0].copy()
        self.df_ch1 = df_click_all[df_click_all["CH"] == 1].copy()

        # Load and filter hydrophone recording
        self.wav_file_path = wav_file_path
        self.sig, self.fs_sig = sf.read(wav_file_path)
        self.filter_sig()  # filter self.sig and save to self.sig_filt

        # Other inits
        self.pos_xy = None

    def filter_sig(self):
        """
        Filter signal to only forward click band: 110-160 kHz
        """
        sos = signal.butter(
            N=40, Wn=[110e3, 160e3], btype="band", output="sos", fs=self.fs_sig
        )
        sig_filt = np.ones(self.sig.shape) * np.nan
        for ch in range(2):
            sig_filt[:, ch] = signal.sosfiltfilt(sos, self.sig[:, ch])
        self.sig_filt = sig_filt

    def _get_clk_pos(self, click_seq_idx: int):
        """
        Get labeled sample location of a specific click.

        Parameters
        ----------
        click_seq_idx
            Sequential number (not index) of the click in self.df_ch0
            Using sequential number here for convenience to loop through all matched clicks.
        """
        return {
            "0": self.df_ch0.iloc[click_seq_idx]["sample_loc"],
            "1": self.df_click_all.loc[
                self.df_ch0.iloc[click_seq_idx]["the_other_ch_index"].astype(int)
            ]["sample_loc"],
        }

    def _get_clk(
        self, clk_ch0_pos: int, sig_cut_len: np.array = np.arange(-512, 512, 1)
    ):
        """
        Cut out click from both channels

        Parameters
        ----------
        clk_ch0_pos
            Labeled sample location in ch0 for a particular click (as returned in get_clk_pos)
        sig_cut_len
            Length of signal to use for cross-correlation
        """
        return self.sig_filt[clk_ch0_pos + sig_cut_len, :]

    @staticmethod
    def _find_dip_index(s, th_max=0.5):
        """
        Find the "dip" between the first and second peaks of a signal envelope.

        Parameters
        ----------
        s
            signal
        th_max
            threshold to find peaks from maximum (e.g., 0.5 of envelope max level)
        """
        s_env = np.abs(signal.hilbert(s))
        pks, _ = signal.find_peaks(s_env, height=s_env.max() * th_max)

        # if multiple peaks, find first "dip"
        if len(pks) > 1:
            return np.argmin(s_env[pks[0] : pks[1]]) + pks[0]
        else:
            return None

    def _c2t_shift_single(self, click_seq_idx: int) -> int:
        """
        Find lag to shift ch1 (clutter) signal to align with ch0 (target) for a single click.

        Parameters
        ----------
        click_seq_idx
            Sequential number (not index) of the click in self.df_ch0
            Using sequential number here for convenience to loop through all matched clicks.
        """
        clk_pos = self._get_clk_pos(click_seq_idx)
        clk = self._get_clk(clk_pos["0"])
        dip = [
            self._find_dip_index(clk[:, ch], th_max=self.threshold_peak_height)
            for ch in range(2)
        ]

        sig_corr = signal.correlate(in1=clk[: dip[0], 0], in2=clk[: dip[1], 1])
        sig_corr_lags = signal.correlation_lags(
            in1_len=clk[: dip[0], 0].size, in2_len=clk[: dip[1], 0].size
        )
        sig_corr_env = np.abs(signal.hilbert(sig_corr))

        return sig_corr_lags[np.argmax(sig_corr_env)]

    def find_clutter2target_shift(self) -> np.array:
        """
        Find lag to shift ch1 (clutter) signal to align with ch0 (target).
        """
        shift_c2t = []
        for seq_idx in range(len(self.df_ch0)):
            shift_c2t.append(self._c2t_shift_single(click_seq_idx=seq_idx))
        return np.array(shift_c2t)

    def shift2xy_triangle(
        self, shift_c2t: np.array, range_marked: np.array
    ) -> np.array:
        """
        Use shift in time to localize animal (x,y) position assuming triangle,
        using hand-marked range to closest target.

        Parameters
        ----------
        shift_c2t
            Clutter (ch1) to target (ch0) shift in time
        range_marked
            Hand-marked range of animal to closest target

        Returns
        -------
            Target ensonification angle based on ch1-ch0 lag in globbl coordinate
        """

        def calc_x_y(row, scenario):
            r = row["animal_range"]
            c2t = row["c2t"]

            def case_minus():
                y = (-2 * r * dr - dr**2 - 1) / 2
                x = np.sqrt(r**2 - (y + 1) ** 2)
                return x, y

            def case_plus():
                y = (2 * r * dr + dr**2 - 1) / 2
                x = np.sqrt(r**2 - y**2)
                return x, y

            dr = np.abs(c2t) / self.fs_sig * self.sound_speed
            if scenario[:2] == "TC":
                if c2t > 0:
                    return case_plus()
                else:
                    return case_minus()
            else:  # "CT"
                if c2t < 0:
                    return case_plus()
                else:
                    return case_minus()

        df_c2t = pd.DataFrame(shift_c2t, columns=["c2t"])
        df_c2t["animal_range"] = range_marked
        df_c2t["x_est"], df_c2t["y_est"] = zip(
            *df_c2t.apply(calc_x_y, axis=1, scenario=self.scenario)
        )

        return df_c2t["x_est"].values, df_c2t["y_est"].values

    def xy2angle_triangle(self, x, y):
        """
        Convert animal (x, y) position to global angle.

        Parameters
        ----------
        x, y
            Animal (x, y) position

        Returns
        -------
        Ensonification angle in global coordinate for target and clutter
        """
        theta_top = np.arctan2(-y - 1, x)
        theta_bot = np.arctan2(-y, x)
        if self.scenario[:2] == "TC":
            return np.array([theta_top, theta_bot]).T / np.pi * 180
        else:  # "CT"
            return np.array([theta_bot, theta_top]).T / np.pi * 180

    def angle_global2target(self, theta_global: np.array):
        """
        Transform estimated angle from global to target coordinate.

        The global coordinate is the one defined by having the top target at (0, 1)
        and bottom target at (0, 0), with positive angle being counter-clockwise from the x-axis,
        and negative angle being clockwise from the x-axis.

        The target coordinate is in the same coordinate as the ensonification angles,
        see `TrailProcessor._add_hydro_ch_info()`
        and `track_features.get_ensonification_angle()`

        Parameters
        ----------
        theta_global
            Target ensonification angle based on ch1-ch0 lag in global coordinate

        Returns
        -------
            Target ensonification angle in the target coordinate
        """
        return -theta_global + 90 + ANGLE_MAP[int(self.scenario[-1])]

    def angle_target2global(self, theta_target: np.array):
        """
        Transform estimated angle from target to global coordinate, inverse of angle_global2target.

        The global coordinate is the one defined by having the top target at (0, 1)
        and bottom target at (0, 0), with positive angle being counter-clockwise from the x-axis,
        and negative angle being clockwise from the x-axis.

        The target coordinate is in the same coordinate as the ensonification angles,
        see `TrailProcessor._add_hydro_ch_info()`
        and `track_features.get_ensonification_angle()`

        Parameters
        ----------
        theta_target
            Target ensonification angle based on ch1-ch0 lag in target coordinate

        Returns
        -------
            Target ensonification angle in the global coordinate
        """
        return -theta_target + 90 + ANGLE_MAP[int(self.scenario[-1])]

    def get_xy_angle(self, range_marked: np.array):
        """
        Estimate (x,y) position and ensonification angle from hydrophone clicks.

        This method saves outputs into the object itself.

        Parameters
        ----------
        range_marked
            Hand-marked range of animal to closest target
        """
        # Get ch1 to ch0 shift in sample numbers
        shift_c2t = self.find_clutter2target_shift()

        # Angle estimates using triangle approach
        self.pos_xy = self.shift2xy_triangle(shift_c2t, range_marked)
        theta_global = self.xy2angle_triangle(self.pos_xy[0], self.pos_xy[1])
        theta_target = self.angle_global2target(theta_global)

        # Store values to df_ch0/1
        def _store(df, ch):
            df["c2t"] = shift_c2t
            df["pos_x"] = self.pos_xy[0]
            df["pos_y"] = self.pos_xy[1]
            df["range_marked"] = range_marked
            df["theta_global"] = theta_global[:, ch]
            df["theta_target"] = theta_target[:, ch]

        _store(self.df_ch0, 0)
        _store(self.df_ch1, 1)


# def upsample(sig_in, upsample_factor, fs_sig, LPF_fc=50e3):
#     sig = np.hstack(
#         (
#             np.expand_dims(sig_in, axis=1),
#             np.zeros(shape=(sig_in.size, upsample_factor - 1))
#         )
#     ).reshape(1, -1).squeeze()

#     # LPF
#     sos_lpf = signal.butter(
#         N=40, Wn=LPF_fc, btype='lowpass', output='sos', fs=fs_sig
#     )

#     return signal.sosfiltfilt(sos_lpf, sig)
