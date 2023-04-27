from typing import Dict
import datetime
import re
from pathlib import Path, PosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from .core import DATA_PATH, RAW_PATH
from . import track_features as tf
from . import file_handling as fh

# re matcher for getting calibration object type
CAL_OBJ_STR = (
    "\d{8}_s\d{1}_t\d{1,2}_\w{8}_xypressure_(?P<cal_obj>\w+)_transformed.csv"  # noqa
)
CAL_OBJ_MATCHER = re.compile(CAL_OBJ_STR)


# Source pre-processed data paths under DATA_PATH["main"]
SRC_PATH = {
    "hydro": DATA_PATH["main"] / "hydrophone/click_reclean_202103/points_cleaned_20211019",
    "dtag": DATA_PATH["main"] / "dtag/dtag_reclean_202104/points_cleaned",
    "track": DATA_PATH["main"] / "tracks/xypressure_cal_transformed",
    "target": DATA_PATH["main"] / "tracks/targets_cal_transformed",    
}

# Destination paths to store synchronized outputs
OUT_PATH = {
    "csv": "click_sync/sync_csv",
    "spectrogram": "click_sync/sync_spectrogram",
    "df_time": "click_sync/sync_df_time",
    "track": "click_sync/sync_track",
}


class ClickSynchronizer:
    """
    Test click_synchronizer for one trial:
        from click_synchronizer.click_synchronizer import click_synchronizer
        c = click_synchronizer()
        c.get_sampling_rate()
        c.curr_trial_idx = 35
        c.get_filenames()
        c.sync_curr_trial()
    """

    def __init__(
            self,
            df_master: pd.DataFrame,
            trial_idx: int,
            raw_path: Path = None,
            src_path: Dict = None,
            out_path: Dict = None
        ):

        self.trial_idx = trial_idx
        self.trial_series = df_master.iloc[trial_idx, :]
        self.raw_path = RAW_PATH if raw_path is None else raw_path
        self.source_path = SRC_PATH if src_path is None else src_path
        self.output_path = OUT_PATH if out_path is None else out_path

        # file paths
        self.hydro_file = None
        self.dtag_file = None
        self.track_file = None
        self.video_file = None

        # sampling rates
        self.fs_dtag = None
        self.fs_hydro = None
        self.fs_video = None

    def get_filenames(self):
        # hydrophone click detection file
        hydro_file = self.source_path["hydro"].joinpath(
            f"t{self.trial_idx:03d}_hydrophone_cleaned.npz"
        )
        if not hydro_file.exists():
            print("cannot find the hydrophone detection file!")
        else:
            self.hydro_file = hydro_file
            print("* hydrophone click detection file:")
            print("  - %s" % self.hydro_file.name)

        # dtag click detection file
        dtag_file = self.source_path["dtag"] / f"t{self.trial_idx:03d}_dtag_cleaned.npz"
        if not dtag_file.exists():
            print("cannot find the dtag detection file!")
        else:
            self.dtag_file = dtag_file
            print("* dtag click detection file:")
            print("  - %s" % self.dtag_file.name)

        # track file
        track_file = sorted(list(
            self.source_path["track"].glob(self.trial_series["fname_prefix"] + "_*.csv")
        ))
        if track_file:
            self.track_file = track_file
            print("* track file(s):")
            for ff in self.track_file:
                print("  - %s" % ff.name)
        else:
            print("cannot find corresponding track file!")

    def get_sampling_rate(self):
        hydro_wav = (
            self.raw_path
            / self.trial_series['DATE'].astype(str)
            / f"session{self.trial_series['SESSION']:d}/target"
            / self.trial_series["hydro_filename"]
        )
        dtag_wav = self.raw_path / self.trial_series["dtag_wav_file"]
        video_mp4 = self.raw_path / self.trial_series["gopro_video"]

        _, self.fs_hydro = sf.read(hydro_wav, 1)
        _, self.fs_dtag = sf.read(dtag_wav, 1)
        self.fs_video = fh.get_fs_video(str(video_mp4))  # opencv only takes str path

        # print sampling rate
        print("Hydrophone was sampled at %d Hz" % self.fs_hydro)
        print("DTAG was sampled at %d Hz" % self.fs_dtag)
        print("Video was sampled at %f Hz" % self.fs_video)

    def get_scenario(self):
        print("* experiment scenario:")
        if self.trial_series["TARGET_ANGLE"][:2] == "TC":
            print("  - TC (target top, clutter bottom)")
            return "TC"
        elif self.trial_series["TARGET_ANGLE"][:2] == "CT":
            print("  - CT (clutter top, target bottom)")
            return "CT"
        else:
            print("  - only the target was in water and not the clutter")
            return None

    def get_chirp(self):
        sig_chirp_t = np.arange(0, 5e-3, 1 / self.fs_dtag)
        sig_chirp = signal.chirp(
            t=sig_chirp_t, f0=170e3, t1=sig_chirp_t.max(), f1=200e3
        )
        return sig_chirp, sig_chirp_t

    def suppress_click(
        self, clk, sig_dtag_seg, sig_suppress_len_time=1024e-6, perc_front=0.2
    ):
        clk = clk[clk < sig_dtag_seg.size]  # get only the first section loaded
        sig_suppress_len_smpl = int(sig_suppress_len_time * self.fs_dtag)
        idx_to_suppress_tmp = np.arange(sig_suppress_len_smpl) - int(
            perc_front * sig_suppress_len_smpl
        )
        sig_dtag_seg_clean = sig_dtag_seg.copy()
        for clk_loc in clk:
            idx_to_suppress = idx_to_suppress_tmp + clk_loc
            idx_to_suppress = idx_to_suppress[
                (idx_to_suppress >= 0) & (idx_to_suppress < sig_dtag_seg.size)
            ]
            sig_dtag_seg_clean[idx_to_suppress] = 0
        return sig_dtag_seg_clean

    def assemble_hydro_df_ch(self, hydro_clk, ch_num, sync_time):
        df_h = pd.DataFrame(data=hydro_clk["location"][ch_num], columns=["sample_loc"])
        df_h["sample_pk"] = hydro_clk["peak"][ch_num]
        df_h["time"] = df_h["sample_loc"] / self.fs_hydro
        df_h["time_corrected"] = df_h["time"] + sync_time
        return df_h

    def get_target_df(self, cal_obj_list):
        df_targets = []
        for cal_obj in cal_obj_list:
            target_file = (
                self.source_path["target"]
                / f"{self.trial_series['fname_prefix']}_targets_{cal_obj}_transformed.csv"
            )
            df_targets.append(
                fh.assemble_target_df(
                    target_file, cal_obj, self.trial_series["TARGET_ANGLE"][:2]
                )
            )
        # element is empty if only one target was present
        if len(df_targets) > 1 and df_targets[0] is not None:
            df_targets = pd.merge(df_targets[0], df_targets[1])
        else:
            df_targets = df_targets[0]

        return df_targets

    def load_dtag_sig_seg(self, sync_time_seg2dtag: datetime.datetime, len_to_load_time: datetime.timedelta, data_dtag_file: PosixPath):
        # load just the beginning portion including the chirp
        start_read_smpl = int(sync_time_seg2dtag * self.fs_dtag)
        # if len_to_load_time is reasonable
        if len_to_load_time.total_seconds() < 5 and len_to_load_time.days >= 0:
            len_to_load_smpl = int((len_to_load_time.total_seconds()) * self.fs_dtag)
            stop_read_smpl = start_read_smpl + len_to_load_smpl
        # either when len_to_load_time is too long or negative
        # (when chirp happened before dtag trial start)
        else:
            start_read_smpl = int(
                (sync_time_seg2dtag + len_to_load_time.total_seconds() - 2.5)
                * self.fs_dtag
            )
            stop_read_smpl = int(
                (sync_time_seg2dtag + len_to_load_time.total_seconds() + 2.5)
                * self.fs_dtag
            )
        sig_dtag_seg, _ = sf.read(
            data_dtag_file, start=start_read_smpl, stop=stop_read_smpl
        )
        sig_dtag_seg_t = np.arange(sig_dtag_seg.size) / self.fs_dtag
        return sig_dtag_seg, sig_dtag_seg_t, start_read_smpl

    def get_dtag_sync_time(self):
        # tap sync time to go from Dtag time to video time
        sync_time_dtag2video = self.trial_series["tap_sync_time_dtag2video"]

        # start of dtag segment time in total dtag time
        sync_time_seg2dtag = datetime.datetime.strptime(
            self.trial_series["dtag_trial_start_time"], "%M:%S.%f"
        )

        # convert sync_time_seg2dtag to be based on seconds
        sync_time_seg2dtag = (
            sync_time_seg2dtag.minute * 60
            + sync_time_seg2dtag.second
            + sync_time_seg2dtag.microsecond / 1e6
        )

        return sync_time_dtag2video, sync_time_seg2dtag

    def get_info_for_hydro_sync(self):
        # rough chirp time in total dtag time
        rough_chirp_time_in_dtag = datetime.datetime.strptime(
            self.trial_series["rough_chirp_time"], "%M:%S.%f"
        )

        # start of dtag segment time in total dtag time
        sync_time_seg2dtag = datetime.datetime.strptime(
            self.trial_series["dtag_trial_start_time"], "%M:%S.%f"
        )

        # rough chirp time in dtag segment
        rough_chirp_time_in_seg = rough_chirp_time_in_dtag - sync_time_seg2dtag

        # rough start location of the chirp
        len_to_load_time = (
            rough_chirp_time_in_dtag
            - sync_time_seg2dtag
            + datetime.timedelta(seconds=1)
        )  # add 1 sec to be safe

        return rough_chirp_time_in_seg, len_to_load_time

    def get_hydro_corr(self, sig_dtag_seg_clean, sig_chirp):
        sig_clean_corr = signal.correlate(sig_dtag_seg_clean, sig_chirp)
        sig_clean_corr_env = np.abs(signal.hilbert(sig_clean_corr))
        sig_corr_lags = signal.correlation_lags(sig_dtag_seg_clean.size, sig_chirp.size)
        sig_corr_t = sig_corr_lags / self.fs_dtag

        return sig_clean_corr_env, sig_corr_t

    @staticmethod
    def interpolate_xy(df_pos, df_track, cal_obj, track_label="ROSTRUM"):
        df_pos[cal_obj + "_pos_x"] = np.interp(
            df_pos["time_corrected"],
            df_track["time_corrected"],
            df_track[track_label + "_X"],
        )
        df_pos[cal_obj + "_pos_y"] = np.interp(
            df_pos["time_corrected"],
            df_track["time_corrected"],
            df_track[track_label + "_Y"],
        )

    def plot_chirp_detection(self, seg_t, seg, corr_t, corr, chirp_idx, title_text):
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex="col")
        plt.subplots_adjust(hspace=0.1)
        # time series and detection
        ax[0].plot(seg_t, seg * 10, label="raw", lw=1)
        ax[0].plot(corr_t, corr + 0.4, label="corr clean", lw=1)
        ax[0].plot(
            corr_t[chirp_idx],
            corr[chirp_idx] + 0.4,
            "rx",
            label="chirp time",
            markersize=6,
        )
        ax[0].set_xlim(0, corr_t.max())
        ax[0].set_ylim(-1, 2)
        ax[0].legend(fontsize=12)
        ax[0].set_title(title_text)
        # spectrogram
        _ = ax[1].specgram(seg, NFFT=256, Fs=self.fs_dtag, vmax=-100, vmin=-160)
        ax[1].set_xlabel("Time (sec)", fontsize=12)
        ax[1].set_xlim(0, corr_t.max())
        plt.show()
        return fig

    @staticmethod
    def plot_click_sync_time(tr, dtag, ch0, ch1, title_text):
        fig = plt.figure(figsize=(8, 6))
        plt.plot(tr["time_corrected"].dropna(), "k", label="track")
        plt.plot(dtag["time_corrected"], ".", markersize=3, label="dtag")
        plt.plot(ch0["time_corrected"], ".", markersize=3, label="hydro 0")
        plt.plot(ch1["time_corrected"], ".", markersize=3, label="hydro 1")
        plt.xlabel("Point / detection", fontsize=12)
        plt.ylabel("Time in video (sec)", fontsize=12)
        plt.grid()
        plt.legend(fontsize=12)
        plt.title(title_text)
        plt.show()
        return fig

    @staticmethod
    def plot_track_with_clicks(dfs, cal_obj, df_track, title_text, track_label):
        fig, ax = plt.subplots(figsize=(9, 6))
        df_track.plot(
            ax=ax,
            x=track_label + "_X",
            y=track_label + "_Y",
            label="track " + track_label,
            color="k",
            lw=0.5,
        )
        dfs["dtag"].plot(
            ax=ax,
            x=cal_obj + "_pos_x",
            y=cal_obj + "_pos_y",
            marker="x",
            linestyle="none",
            color="k",
            label="dtag clicks",
            markersize=5,
        )
        dfs["hydro_ch0"].plot(
            ax=ax,
            x=cal_obj + "_pos_x",
            y=cal_obj + "_pos_y",
            marker=".",
            linestyle="none",
            color="r",
            label="hydro 0 clicks",
            markersize=5,
        )
        dfs["hydro_ch1"].plot(
            ax=ax,
            x=cal_obj + "_pos_x",
            y=cal_obj + "_pos_y",
            marker="o",
            markerfacecolor="none",
            color="b",
            linestyle="none",
            label="hydro 1 clicks",
            markersize=5,
        )
        if dfs["targets"] is not None:  # if both target and clutter were present
            dfs["targets"].plot(
                ax=ax,
                x="target_" + cal_obj + "_pos_x",
                y="target_" + cal_obj + "_pos_y",
                linestyle="none",
                marker="*",
                color="c",
                markersize=10,
                label="target",
            )
            dfs["targets"].plot(
                ax=ax,
                x="clutter_" + cal_obj + "_pos_x",
                y="clutter_" + cal_obj + "_pos_y",
                linestyle="none",
                marker="o",
                color="c",
                markersize=6,
                label="clutter",
            )
        plt.axis("equal")
        plt.gca().invert_yaxis()
        plt.title("%s, %s" % (title_text, cal_obj))
        plt.xlabel("Distance (m)", fontsize=12)
        plt.ylabel("Distance (m)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        plt.show()
        return fig

    def sync_curr_trial(
        self,
        track_label,
        proc_track=True,
        offset_seg_time=0.2,
        peak_height_th=0.5,
        peak_spacing_th=0.1,
        plot_opt=True,
    ):
        """
        Synchronize hydrophone detection with chirp received on DTAG.

        Parameters
        ----------
        track_label : str {"ROSTRUM", "DTAG"}
            label to use for track
        proc_track : bool
            whether or not the raw track is nan-filled and smoothed
        """
        # get dtag filename for this trial
        data_dtag_file = self.raw_path / self.trial_series["dtag_wav_file"]

        # get click detection info
        if (
            self.trial_idx == 81
        ):  # need special handling for t81 due to the 2 hydro detection files
            hydro_clk = np.load(
                self.source_path["hydro"] / "t081_1_hydrophone_cleaned.npz",
                allow_pickle=True,
            )
        else:
            hydro_clk = np.load(self.hydro_file, allow_pickle=True)
        dtag_clk = np.load(self.dtag_file, allow_pickle=True)

        # sync time for dtag
        sync_time_dtag2video, sync_time_seg2dtag = self.get_dtag_sync_time()

        # sync time for hydrophone
        rough_chirp_time_in_seg, len_to_load_time = self.get_info_for_hydro_sync()

        # load a portion that includes the chirp signal
        sig_dtag_seg, sig_dtag_seg_t, seg_read_smpl = self.load_dtag_sig_seg(
            sync_time_seg2dtag, len_to_load_time, data_dtag_file
        )

        # chirp template
        sig_chirp, sig_chirp_t = self.get_chirp()

        # suppress click locations
        # note: smpl_diff = 0 if segment read from dtag_trial_start_time
        smpl_diff = seg_read_smpl - int(sync_time_seg2dtag * self.fs_dtag)
        dtag_clk_shifted = (
            dtag_clk["location"][0][dtag_clk["location"][0] > smpl_diff] - smpl_diff
        )
        sig_dtag_seg_clean = self.suppress_click(dtag_clk_shifted, sig_dtag_seg)

        # find hydrophone to dtag sync time
        sig_clean_corr_env, sig_corr_t = self.get_hydro_corr(
            sig_dtag_seg_clean, sig_chirp
        )
        offset_seg_smpl = int(offset_seg_time * self.fs_dtag)
        sig_clean_corr_env = sig_clean_corr_env[offset_seg_smpl:]
        sig_corr_t = sig_corr_t[offset_seg_smpl:]
        chirp_loc, pk = signal.find_peaks(
            sig_clean_corr_env,
            height=sig_clean_corr_env.max() * peak_height_th,
            distance=peak_spacing_th * self.fs_dtag,
        )
        # possible_chirp_time = [datetime.timedelta(seconds=tt) for tt in sig_corr_t[chirp_loc]]
        diff_time = np.array(
            [
                (
                    rough_chirp_time_in_seg - datetime.timedelta(seconds=tt)
                ).total_seconds()
                for tt in sig_corr_t[chirp_loc]
            ]
        )
        sync_time_hydro2seg = sig_corr_t[chirp_loc][np.abs(diff_time).argmin()]
        sync_time_hydro2vid = (
            seg_read_smpl / self.fs_dtag + sync_time_hydro2seg + sync_time_dtag2video
        )

        # Dataframe for dtag click detections
        df_dtag = pd.DataFrame(data=dtag_clk["location"][0], columns=["sample_loc"])
        df_dtag["time"] = df_dtag["sample_loc"] / self.fs_dtag
        df_dtag["time_corrected"] = (
            df_dtag["time"] + sync_time_seg2dtag + sync_time_dtag2video
        )

        # If trial 81, update to use the combined hydro detection file
        if self.trial_idx == 81:
            del hydro_clk
            hydro_clk = np.load(self.hydro_file, allow_pickle=True)

        # Dataframe for hydrophone click detections
        df_hydro_ch0 = self.assemble_hydro_df_ch(hydro_clk, 0, sync_time_hydro2vid)
        df_hydro_ch1 = self.assemble_hydro_df_ch(hydro_clk, 1, sync_time_hydro2vid)

        # Interpolate for x-y positions from track
        cal_obj_list = []
        df_track_list = []
        for tr in self.track_file:
            cal_obj = CAL_OBJ_MATCHER.match(tr.name)
            cal_obj = cal_obj["cal_obj"]
            cal_obj_list.append(cal_obj)
            df_track = pd.read_csv(tr, index_col=0)
            if proc_track:  # whether to fill in small gaps and smoothing track
                df_track = tf.interp_smooth_track(df_track)
            df_track["time"] = df_track.index / self.fs_video
            df_track["time_corrected"] = (
                df_track["time"] + self.trial_series["FRAME_NUM_START"] / self.fs_video
            )
            df_track_list.append(df_track)
            # interpolate x-y for all detections
            self.interpolate_xy(df_dtag, df_track, cal_obj, track_label=track_label)
            self.interpolate_xy(
                df_hydro_ch0, df_track, cal_obj, track_label=track_label
            )
            self.interpolate_xy(
                df_hydro_ch1, df_track, cal_obj, track_label=track_label
            )

        # Dataframe for target locations
        df_targets = self.get_target_df(cal_obj_list)

        if plot_opt:
            # Plot to check chirp detection
            fig_spectrogram = self.plot_chirp_detection(
                sig_dtag_seg_t,
                sig_dtag_seg,
                sig_corr_t,
                sig_clean_corr_env,
                chirp_loc[np.abs(diff_time).argmin()],
                "trial %d - %s" % (self.trial_idx, self.trial_series["fname_prefix"]),
            )

            # Plot to check sync timing
            fig_df_time = self.plot_click_sync_time(
                df_track,
                df_dtag,
                df_hydro_ch0,
                df_hydro_ch1,
                "trial %d - %s" % (self.trial_idx, self.trial_series["fname_prefix"]),
            )

            # Plot to check click detection on track
            if df_targets is None:
                fig_track = None
            else:
                dfs = {
                    "dtag": df_dtag,
                    "hydro_ch0": df_hydro_ch0,
                    "hydro_ch1": df_hydro_ch1,
                    "targets": df_targets,
                }
                fig_track = dict()
                for cal_obj, df_track_cal_obj in zip(cal_obj_list, df_track_list):
                    fig_track[cal_obj] = self.plot_track_with_clicks(
                        dfs=dfs,
                        cal_obj=cal_obj,
                        df_track=df_track_cal_obj,
                        title_text="trial %d - %s"
                        % (self.trial_idx, self.trial_series["fname_prefix"]),
                        track_label=track_label,
                    )
        else:
            fig_spectrogram, fig_df_time, fig_track = None, None, None

        return df_dtag, df_hydro_ch0, df_hydro_ch1, fig_spectrogram, fig_df_time, fig_track

    def sync_curr_trial_LED(self, track_label, proc_track=True, plot_opt=True):
        """
        Synchronize hydrophone detection with LED flash at the beginning of the track video.

        Parameters
        ----------
        track_label : str {"ROSTRUM", "DTAG"}
            label to use for track
        proc_track : bool
            whether or not the raw track is nan-filled and smoothed
        """

        # get click detection info
        hydro_clk = np.load(self.hydro_file, allow_pickle=True)
        dtag_clk = np.load(self.dtag_file, allow_pickle=True)

        # sync time for dtag
        sync_time_dtag2video, sync_time_seg2dtag = self.get_dtag_sync_time()

        # sync time for hydrophone: recording started at track video start (LED flash)
        sync_time_hydro2vid = self.trial_series["FRAME_NUM_START"] / self.fs_video

        # Dataframe for dtag click detections
        df_dtag = pd.DataFrame(data=dtag_clk["location"][0], columns=["sample_loc"])
        df_dtag["time"] = df_dtag["sample_loc"] / self.fs_dtag
        df_dtag["time_corrected"] = (
            df_dtag["time"] + sync_time_seg2dtag + sync_time_dtag2video
        )

        # Dataframe for hydrophone click detections
        df_hydro_ch0 = self.assemble_hydro_df_ch(hydro_clk, 0, sync_time_hydro2vid)
        df_hydro_ch1 = self.assemble_hydro_df_ch(hydro_clk, 1, sync_time_hydro2vid)

        # Interpolate for x-y positions from track
        cal_obj_list = []
        df_track_list = []
        for tr in self.track_file:
            cal_obj = CAL_OBJ_MATCHER.match(tr.name)
            cal_obj = cal_obj["cal_obj"]
            cal_obj_list.append(cal_obj)
            df_track = pd.read_csv(tr, index_col=0)
            if proc_track:  # whether to fill in small gaps and smoothing track
                df_track = tf.interp_smooth_track(df_track)
            df_track["time"] = df_track.index / self.fs_video
            df_track["time_corrected"] = (
                df_track["time"] + self.trial_series["FRAME_NUM_START"] / self.fs_video
            )
            df_track_list.append(df_track)
            # interpolate x-y for all detections
            self.interpolate_xy(df_dtag, df_track, cal_obj, track_label=track_label)
            self.interpolate_xy(
                df_hydro_ch0, df_track, cal_obj, track_label=track_label
            )
            self.interpolate_xy(
                df_hydro_ch1, df_track, cal_obj, track_label=track_label
            )

        # Dataframe for target locations
        df_targets = self.get_target_df(cal_obj_list)

        if plot_opt:
            # Plot to check sync timing
            fig_df_time = self.plot_click_sync_time(
                df_track,
                df_dtag,
                df_hydro_ch0,
                df_hydro_ch1,
                "trial %d - %s" % (self.trial_idx, self.trial_series["fname_prefix"]),
            )

            # Plot to check click detection on track
            if df_targets is None:
                fig_track = None
            else:
                dfs = {
                    "dtag": df_dtag,
                    "hydro_ch0": df_hydro_ch0,
                    "hydro_ch1": df_hydro_ch1,
                    "targets": df_targets,
                }
                fig_track = dict()
                for cal_obj, df_track_cal_obj in zip(cal_obj_list, df_track_list):
                    fig_track[cal_obj] = self.plot_track_with_clicks(
                        dfs=dfs,
                        cal_obj=cal_obj,
                        df_track=df_track_cal_obj,
                        title_text="trial %d - %s"
                        % (self.trial_idx, self.trial_series["fname_prefix"]),
                        track_label=track_label,
                    )
        else:
            fig_df_time, fig_track = None, None

        return df_dtag, df_hydro_ch0, df_hydro_ch1, fig_df_time, fig_track
