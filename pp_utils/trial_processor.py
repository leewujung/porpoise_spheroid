from typing import Union, Dict
from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import soundfile as sf

from .core import RAW_PATH, DATA_PATH, ANGLE_MAP, HYDRO_PARAMS
from .file_handling import get_trial_info, assemble_target_df, get_fs_video
from . import track_features as tf
from . import hydro_clicks as hc
from .seawater import calc_absorption
from . import misc



class TrialProcessor:

    def __init__(
        self,
        df_master: pd.DataFrame,
        trial_idx: int,
        data_path: Dict = DATA_PATH,
        raw_path: Path = RAW_PATH,
        track_label: str = "DTAG",
        hydro_params: Dict = HYDRO_PARAMS,
    ):
        
        self.trial_idx = trial_idx
        self.track_label = track_label
        self.raw_path = raw_path
        self.data_paths = data_path
        self.trial_series = df_master.iloc[trial_idx, :]
        self.hydro_params = hydro_params

        # Get all trial flags, params, and paths
        self.flags, self.params, self.paths = get_trial_info(
            df_master, self.data_paths, self.trial_idx
        )

        # Get all trial dataframes
        self.get_all_trial_dfs()

        # Get sampling rates
        self.get_sampling_rates()

        # Print trial info
        self._print_file_paths()
        self._print_scenario()
        self._print_choice()

        # Other init
        self.touch_time_corrected = None


    def get_all_trial_dfs(self):
        """
        Obtain all dataframes for this trial.
        """
        # Get dataframes from stored paths and track label selection
        self.df_track = (
            pd.read_csv(self.paths["track"], index_col=0) if self.paths["track"] else None
        )
        self.df_dtag = (
            pd.read_csv(self.paths[f"dtag_{self.track_label}"], index_col=0)
            if self.paths[f"dtag_{self.track_label}"] else None
        )
        self.df_hydro_ch0 = (
            pd.read_csv(self.paths[f"hydro_ch0_{self.track_label}"], index_col=0)
            if self.paths[f"hydro_ch0_{self.track_label}"] else None
        )
        self.df_hydro_ch1 = (
            pd.read_csv(self.paths[f"hydro_ch1_{self.track_label}"], index_col=0)
            if self.paths[f"hydro_ch1_{self.track_label}"] else None
        )

        # Assemble target dataframe
        target_pos_comb = self.trial_series["TARGET_ANGLE"][:2]  # "TC" or "CT"
        self.df_targets = (
            assemble_target_df(self.paths["target"], self.params["cal_obj"], target_pos_comb)
            if self.paths["target"] else None
        )

    def _print_file_paths(self):
        """
        Print file paths
        """        
        print("* cal object selected: %s" % self.params['cal_obj'])
        print("* track file:     %s" % self.paths["track"].name)
        print("* dtag file:      %s" % self.paths[f"dtag_{self.track_label}"].name)
        print("* hydro ch0 file: %s" % self.paths[f"hydro_ch0_{self.track_label}"].name)
        print("* hydro ch1 file: %s" % self.paths[f"hydro_ch1_{self.track_label}"].name)

    def _print_scenario(self):
        print("* scenario: %s" % self.trial_series["TARGET_ANGLE"])

    def _print_choice(self):
        print("* choice: %d" % self.params["chose_target"])

    def save_trial(self, save_path: Union[str, Path]):
        """
        Save content of the TrialProcessor object for later use.

        Parameters
        ----------
        save_path : str or Path
            path to save the object to
        """
        fname = (
            Path(save_path) / f"t{self.trial_idx:03d}_{self.trial_series['fname_prefix']}.pickle"
        )
        print(f"Saving object to: {fname.name}")

        with open(fname, "wb") as file_out:
            pickle.dump(self, file_out)

        return fname

    def get_sampling_rates(self, verbose=False):
        """
        Get sampling rate for all recording systems.

        Parameters
        ----------
        s : pandas.Series
            pandas series of a particular trial
        data_path : str or Path
            path to data folder
        verbose : bool
            whether or not to print sampling rate for all systems

        Returns
        -------
        fs_video, fs_dtag, fs_hydro
        """
        hydro_raw_path = self.raw_path / f"{self.trial_series['DATE']}/session{self.trial_series['SESSION']:d}/target"
        hydro_raw_file = hydro_raw_path / self.trial_series["hydro_filename"]
        dtag_raw_file = self.raw_path / self.trial_series["dtag_wav_file"]
        vid_raw_file = self.raw_path / self.trial_series["gopro_video"]

        _, self.fs_hydro = sf.read(hydro_raw_file, 1)
        _, self.fs_dtag = sf.read(dtag_raw_file, 1)
        self.fs_video = get_fs_video(str(vid_raw_file))

        # print sampling rate
        if verbose:
            print("Hydrophone was sampled at %d Hz" % self.fs_hydro)
            print("DTAG was sampled at %d Hz" % self.fs_dtag)
            print("Video was sampled at %f Hz" % self.fs_video)

    def process_track(self):
        """
        Interp, smooth, and add synced timing info to track.
        """
        if self.df_track is None:
            print("Calibrated track does not exist!")
        else:
            df_track = self.df_track.copy()

            # Fill in NaN and smooth track
            df_track = tf.interp_smooth_track(df_track)

            # Get times for track
            df_track["time"] = df_track.index / self.fs_video
            frame_start_time = (
                self.df_master.loc[
                    self.df_master["fname_prefix"] == self.params["fname_prefix"],
                    ["FRAME_NUM_START"],
                ].values.squeeze()
                / self.fs_video
            )
            df_track["time_corrected"] = df_track["time"] + frame_start_time

            # Save touch_time_corrected
            self.touch_time_corrected = df_track.iloc[self.params["idx_touch"]][
                "time_corrected"
            ]
            self.df_track = df_track
        
    def add_track_features(self):
        """
        Add all track features except for DTAG-related info.
        """
        df_track = self.df_track.copy()
        df_targets = self.df_targets.copy()
        cal_obj = self.params["cal_obj"]

        for track_label in ["DTAG", "ROSTRUM"]:
            # Add distance measure
            df_track[f"{track_label}_dist_to_target"] = tf.get_dist_to_object(
                df_track, df_targets, "target", cal_obj, track_label,
            )
            df_track[f"{track_label}_dist_to_clutter"] = tf.get_dist_to_object(
                df_track, df_targets, "clutter", cal_obj, track_label,
            )
            # Add elliptical distance
            df_track[f"{track_label}_dist_elliptical"] = (
                df_track[f"{track_label}_dist_to_target"]
                + df_track[f"{track_label}_dist_to_clutter"]
            )
            # Add speed
            df_track[f"{track_label}_speed"] = tf.get_speed(df_track, track_label)

        # Add heading info
        self.df_track["angle_heading_to_target"] = tf.get_angle_heading_to_object(
            df_track, df_targets, "target", cal_obj,
        )
        self.df_track["angle_heading_to_clutter"] = tf.get_angle_heading_to_object(
            df_track, df_targets, "clutter", cal_obj,
        )
        self.df_track["absolute_heading"] = tf.get_absolute_heading(df_track)

    def add_hydro_info(self):
        """
        Add hydrophone-derived info to df_hydro_ch0 or df_hydro_ch1.
        """
        if not self.df_hydro_ch0 or not self.df_hydro_ch1:
            # No detection on hydro channels
            print("No hydro click detected on hydrophone channels! Skip trial...")
        else:
            # Add info for both channels
            for df, obj_type in zip([self.df_hydro_ch0, self.df_hydro_ch1], ["target", "clutter"]):
                # Interpolate track to hydro dfs
                df["DTAG_X"], df["DTAG_Y"] = tf.interpolate_track_xy(
                    df_in=self.df_track, df_out=df, track_label="DTAG"
                )
                df["ROSTRUM_X"], df["ROSTRUM_Y"] = tf.interpolate_track_xy(
                    df_in=self.df_track, df_out=df, track_label="ROSTRUM"
                )

                # Get distance to hydrophone
                df["dist_to_hydro"] = tf.get_dist_to_object(
                    df_track=df,
                    df_targets=self.df_targets,
                    obj_type=obj_type,
                    cal_obj=self.params["cal_obj"],
                    track_label="DTAG",
                )

                # Compute ICI
                df["ICI"] = df["time_corrected"].diff()

                # Compute inspection angle (based on DTAG)
                df["angle_yaxis_DTAG"] = tf.get_angle_from_yaxis(
                    df_hydro=df,
                    df_targets=self.df_targets,
                    track_label="DTAG",
                    obj_type=obj_type,
                    cal_obj=self.params["cal_obj"],
                )

                # Compute ensonification angle
                df["enso_angle"] = df.apply(
                    tf.get_ensonification_angle,
                    axis=1,
                    args=(ANGLE_MAP[self.trial_series["ANGLE"]], "angle_yaxis_DTAG"),
                )

                # Compute echo reception angle by animal
                df["angle_heading_to_hydro"] = tf.get_angle_heading_to_object(
                    df_track=df,
                    df_targets=self.df_targets,
                    obj_type=obj_type,
                    cal_obj=self.params["cal_obj"],
                )

    def add_SNR_p2p(self):
        """
        Add click SNR and p2p voltage to both hydro channels.
        """
        if not self.df_hydro_ch0 or not self.df_hydro_ch1:
            # No detection on hydro channels
            print("No hydro click detected on hydrophone channels! Skip trial...")
        else:
            # Add SNR and p2p info on both hydro channels
            for ch, df in zip([0, 1], [self.df_hydro_ch0, self.df_hydro_ch1]):

                # Load click matrix
                fname_prefix = self.params["fname_prefix"]
                clk_fname = "%s_extracted_clicks_ch%d.npy" % (fname_prefix, ch)
                cm = np.load(self.paths["extracted_click_path"] / clk_fname)

                # Add SNR
                clk_var = hc.get_clk_variance(
                    clk_mtx=cm,
                    clk_sel_len_sec=self.hydro_params["clk_sel_len_sec"],
                    perc_before_pk=self.hydro_params["perc_before_pk"],
                    fs_hydro=self.fs_hydro,
                )
                bkg_var = hc.get_bkg_variance(
                    clk_mtx=cm,
                    bkg_len_sec=self.hydro_params["bkg_len_sec"],
                    fs_hydro=self.fs_hydro,
                )
                df["SNR"] = 10 * np.log10(clk_var) - 10 * np.log10(bkg_var)

                # Add p2p voltage
                df["p2p"] = hc.get_clk_p2p(
                    clk_mtx=cm,
                    clk_sel_len_sec=self.hydro_params["clk_sel_len_sec"],
                    perc_before_pk=self.hydro_params["perc_before_pk"],
                    fs_hydro=self.fs_hydro,
                )

    def add_RL_ASL_pointEL(
        self,
        frequency=130e3,
        temperature=16,
        salinity=28,
        pressure=1,
        pH=8,
        formula_source="FG",
    ):
        """
        Add click apparent source level (ASL) to hydro channels.

        The default values of environmental parameters are set to
        match the absorption 0.04 dB m^-1 at 130 kHz specified in
        Malinka et al. 2021 JEB paper.
        """
        if not self.df_hydro_ch0 or not self.df_hydro_ch1:
            # No detection on hydro channels
            print("No hydro click detected on hydrophone channels! Skip trial...")
        else:
            # Add info on both hydro channels
            for df in [self.df_hydro_ch0, self.df_hydro_ch1]:

                # Get transmission loss
                absorption_1way_1m = calc_absorption(
                    frequency=frequency,
                    temperature=temperature,
                    salinity=salinity,
                    pressure=pressure,
                    pH=pH,
                    formula_source=formula_source,
                )
                df["absorption_1way"] = absorption_1way_1m * df["dist_to_hydro"]
                df["spreading_1way"] = 20 * np.log10(df["dist_to_hydro"])

                # Receive level
                df["RL"] = (
                    20 * np.log10(df["p2p"])
                    - self.hydro_params["hydro_sens"]
                    - self.hydro_params["recording_gain"]
                )

                # Apparent source level
                df["ASL"] = df["RL"] + df["absorption_1way"] + df["spreading_1way"]

                # point scatterer echo level
                df["pointEL"] = df["RL"] - df["absorption_1way"] - df["spreading_1way"]

    def add_before_touch_to_all_dfs(self):
        """
        Add before_touch column to df if not already exist.
        """
        # Add "before_touch" column to track df
        if "before_touch" not in self.df_track:
            self.df_track["before_touch"] = misc.get_before_touch_column(
                df=self.df_track, idx_touch=self.trial_paths["idx_touch"]
            )
        # Add "before_touch" column to dtag and hydro dfs
        for df in [self.df_dtag, self.df_hydro_ch0, self.df_hydro_ch1]:
            if "before_touch" not in df:
                df["before_touch"] = misc.get_before_touch_column(
                    df=df, time_touch=self.touch_time_corrected
                )
