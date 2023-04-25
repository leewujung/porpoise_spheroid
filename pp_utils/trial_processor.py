from typing import Union, Dict
from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import soundfile as sf

from .core import RAW_PATH, DATA_PATH, ANGLE_MAP, HYDRO_PARAMS, ENV_PARAMS, SCAN_PARAMS
from .file_handling import get_trial_info, assemble_target_df, get_fs_video
from . import track_features as tf
from . import hydro_clicks as hc
from . import inspection_angle as ia
from .seawater import calc_absorption
from . import misc



class TrialProcessor:

    def __init__(
        self,
        df_master: pd.DataFrame,
        trial_idx: int,
        data_path: Dict = None,
        raw_path: Path = None,
        track_label: str = "DTAG",
        hydro_params: Dict = None,
        env_params: Dict = None,
        scan_params: Dict = None,
    ):
        
        self.trial_idx = trial_idx
        self.track_label = track_label
        self.raw_path = RAW_PATH if raw_path is None else raw_path
        self.data_paths = DATA_PATH if data_path is None else data_path
        self.trial_series = df_master.iloc[trial_idx, :]
        self.hydro_params = HYDRO_PARAMS if hydro_params is None else hydro_params
        self.env_params = ENV_PARAMS if env_params is None else env_params
        self.scan_params = SCAN_PARAMS if scan_params is None else scan_params

        # Get all trial flags, params, and paths
        self.flags, self.params, self.paths = get_trial_info(
            df_master, self.data_paths, self.trial_idx
        )

        # Get all trial dataframes
        self.get_all_trial_dfs()

        # Get sampling rates
        self.get_sampling_rates()

        # Get timing for time_corrected on track and touch_time_corrected
        self.get_timing()

        # Print trial info
        self._print_file_paths()
        self._print_scenario()
        self._print_choice()

        # Other init
        self.touch_time_corrected = None
        self.df_click_all = None  # all selected clicks from both hydro channels
        self.df_click_scan = None  # clicks retained after scan determination
        self.df_scan = None  # scan number
        self.last_scan_start = None  # start time of last scan
        self.last_scan_end = None  # end time of last scan



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

    def get_timing(self):
        """
        Get time_corrected for df_track and overall touch_time_corrected.
        """
        if self.df_track is None:
            print("Calibrated track does not exist!")
        else:
            # Get times for track
            self.df_track["time"] = self.df_track.index / self.fs_video
            frame_start_time = self.trial_series["FRAME_NUM_START"] / self.fs_video
            self.df_track["time_corrected"] = self.df_track["time"] + frame_start_time

            # Save touch_time_corrected
            self.touch_time_corrected = self.df_track.iloc[self.params["idx_touch"]]["time_corrected"]
        
    def add_track_features(self):
        """
        Add all track features except for DTAG-related info.
        """
        if self.df_track is None:
            print("Calibrated track does not exist!")
        else:
            # Fill in NaN and smooth track
            self.df_track = tf.interp_smooth_track(self.df_track)

            for track_label in ["DTAG", "ROSTRUM"]:
                # Add distance measure
                self.df_track[f"{track_label}_dist_to_target"] = tf.get_dist_to_object(
                    self.df_track, self.df_targets, "target", self.params["cal_obj"], track_label,
                )
                self.df_track[f"{track_label}_dist_to_clutter"] = tf.get_dist_to_object(
                    self.df_track, self.df_targets, "clutter", self.params["cal_obj"], track_label,
                )
                # Add elliptical distance
                self.df_track[f"{track_label}_dist_elliptical"] = (
                    self.df_track[f"{track_label}_dist_to_target"]
                    + self.df_track[f"{track_label}_dist_to_clutter"]
                )
                # Add speed
                self.df_track[f"{track_label}_speed"] = tf.get_speed(self.df_track, track_label)

            # Add heading info
            self.df_track["angle_heading_to_target"] = tf.get_angle_heading_to_object(
                self.df_track, self.df_targets, "target", self.params["cal_obj"],
            )
            self.df_track["angle_heading_to_clutter"] = tf.get_angle_heading_to_object(
                self.df_track, self.df_targets, "clutter", self.params["cal_obj"],
            )
            self.df_track["absolute_heading"] = tf.get_absolute_heading(self.df_track)

    def add_hydro_features(self):
        """
        Add hydrophone-derived info to df_hydro_ch0 or df_hydro_ch1.
        """
        if self.df_hydro_ch0 is None or self.df_hydro_ch1 is None:
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
                    df, self.df_targets, obj_type, self.params["cal_obj"], track_label="DTAG",
                )

                # Compute inspection angle (based on DTAG position)
                df["angle_yaxis_DTAG"] = tf.get_angle_from_yaxis(
                    df, self.df_targets, obj_type, self.params["cal_obj"], track_label="DTAG",
                )

                # Compute ensonification angle
                df["enso_angle"] = df.apply(
                    tf.get_ensonification_angle,
                    axis=1,
                    args=(ANGLE_MAP[self.trial_series["ANGLE"]], "angle_yaxis_DTAG"),
                )

                # Compute echo reception angle by animal
                df["angle_heading_to_hydro"] = tf.get_angle_heading_to_object(
                    df, self.df_targets, obj_type, self.params["cal_obj"],
                )

                # Compute ICI
                df["ICI"] = df["time_corrected"].diff()


    def add_SNR_p2p(self):
        """
        Add click SNR and p2p voltage to both hydro channels.
        """
        if self.df_hydro_ch0 is None or self.df_hydro_ch1 is None:
            # No detection on hydro channels
            print("No hydro click detected on hydrophone channels! Skip trial...")
        else:
            # Add SNR and p2p info on both hydro channels
            for ch, df in zip([0, 1], [self.df_hydro_ch0, self.df_hydro_ch1]):

                # Load click matrix
                cm = np.load(self.paths[f"extracted_click_ch{ch}"])

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

    def add_RL_ASL_pointEL(self):
        """
        Add click apparent source level (ASL) to hydro channels.

        """
        if self.df_hydro_ch0 is None or self.df_hydro_ch1 is None:
            # No detection on hydro channels
            print("No hydro click detected on hydrophone channels! Skip trial...")
        else:
            # Add info on both hydro channels
            for df in [self.df_hydro_ch0, self.df_hydro_ch1]:

                # Get transmission loss
                absorption_1way_1m = calc_absorption(
                    frequency=self.env_params["frequency"],
                    temperature=self.env_params["temperature"],
                    salinity=self.env_params["salinity"],
                    pressure=self.env_params["pressure"],
                    pH=self.env_params["pH"],
                    formula_source=self.env_params["absorption_formula_source"],
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
                df=self.df_track, idx_touch=self.params["idx_touch"]
            )
        # Add "before_touch" column to dtag and hydro dfs
        for df in [self.df_dtag, self.df_hydro_ch0, self.df_hydro_ch1]:
            if "before_touch" not in df:
                df["before_touch"] = misc.get_before_touch_column(
                    df=df, time_touch=self.touch_time_corrected
                )

    def get_desired_track_portion(
        self,
        dist_max=("DTAG_dist_elliptical", 12),
        dist_min=("ROSTRUM_dist_to_target", 0.1),
    ):
        """
        Get a desired track portion based on distance criteria.
        """
        # Check if trying to find the portion makes sense
        df_track = self.df_track[self.df_track["before_touch"]].copy()
        idx_first = df_track["angle_heading_to_target"].first_valid_index()
        idx_last = df_track["angle_heading_to_target"].last_valid_index()

        # Criteria
        min_ok = (
            True
            if dist_min[1] is None
            else df_track.loc[idx_last, "ROSTRUM_dist_to_target"] < dist_min[1]
        )
        max_ok = (
            True
            if dist_max[1] is None
            else df_track.loc[idx_first, "DTAG_dist_elliptical"] >= dist_max[1]
        )

        if (
            # Skip if missing some markers comletely
            idx_first is not None
            and idx_last is not None
            # Skip if doesn't meet the selection criteria
            and min_ok
            and max_ok
        ):
            idx_dist_min, idx_dist_max = tf.get_track_index_based_on_dist(
                df_track,
                dist_name_max=dist_max[0],
                dist_name_min=dist_min[0],
                dist_th_max=dist_max[1],
                dist_th_min=dist_min[1],
            )
            # pandas slicing include BOTH the start and the stop
            # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html#pandas-dataframe-loc  # noqa
            return self.df_track.loc[idx_dist_max:idx_dist_min]
        else:
            return None

    def get_hydro_scan_num(self):
        """
        Count the number of hydrophone scans using filtered clicks.

        Set the following dfs:
        - self.df_click_all
        - self.df_click_scan
        - self.df_scan
        """
        # Match clicks and assign scanned channel to each click
        df_h0 = hc.select_scan_clicks(self.df_hydro_ch0, th_RL=self.scan_params["th_RL"])
        df_h1 = hc.select_scan_clicks(self.df_hydro_ch1, th_RL=self.scan_params["th_RL"])
        df_h_all = hc.match_clicks(df_h0, df_h1)
        df_h_all = hc.assign_scan(df_h_all, RL_tolerance=self.scan_params["RL_tolerance"])
        selector_matched = ~df_h_all["the_other_ch_index"].isna()  # have matched channel

        # Sanity check
        selector_ch0 = df_h_all["CH"] == 0  # ch0
        selector_ch1 = df_h_all["CH"] == 1  # ch1
        if not len(df_h_all[selector_ch0 & selector_matched]) == len(
            df_h_all[selector_ch1 & selector_matched]
        ):
            raise ValueError("Number of clicks in ch0 and ch1 do not match!")

        # Set up RL difference across channels
        diff_matched = (
            df_h_all[selector_ch0 & selector_matched]["RL"].values
            - df_h_all[selector_ch1 & selector_matched]["RL"].values
        )
        df_h_all["RL_diff"] = np.nan
        df_h_all.loc[selector_ch0 & selector_matched, "RL_diff"] = diff_matched
        df_h_all.loc[selector_ch1 & selector_matched, "RL_diff"] = diff_matched

        # Store ALL selected clicks from both channels with attached info
        self.df_click_all = df_h_all.copy()

        # Get streaks (strides of consecutive clicks, which are the scans)
        df_h_all = hc.get_streaks(df_h_all)  # get streaks the first time
        df_h_all = hc.clean_streaks(  # remove streaks with only few clicks
            df_h_all, th_num_clk=self.scan_params["th_num_clk"]
        )
        df_h_all = hc.get_streaks(df_h_all)  # get streaks again

        # Scan validity:
        #  - False: both targets ensonified
        #  - True: only one target ensonified
        scan_validity = df_h_all.groupby("streak_id").apply(
            hc.is_true_scan,
            th_RL_diff=self.scan_params["true_scan_th_RL_diff"],
            max_num_click_has_RL_diff=self.scan_params["true_scan_max_num_click_has_RL_diff"],
        )

        # Assemble scan number dataframe:
        #   - index is scan number (1-based)
        #   - include column for whether it is a true scan (valid=True/False)
        df_scan = df_h_all[df_h_all["start_of_streak"]][["scan_ch", "time_corrected"]]
        df_scan["valid"] = scan_validity.values
        df_scan.index = np.arange(len(df_scan)) + 1
        self.df_scan = df_scan.copy()

        # Store subset of selected clicks used to determine scans
        self.df_click_scan = df_h_all.copy()

        # TODO: don't have to keep both df_click_all and df_click_scan
        #  if we can fill in streak_id of the previous streak rather than deleting
        #  the with unwanted streak_id under _clean_streaks

    def sort_df_scan_to_channel(self, ch):
        """
        Sort scans to each channel.
        
        df_scan["valid"] = True are scan only to 1 channel
        df_scan["valid"] = False are scan toward both channels
        """
        df_scan = self.df_scan
        return pd.concat((
            df_scan[~df_scan["valid"]],
            df_scan[df_scan["valid"] & (df_scan["scan_ch"] == ch)]
        ))

    def decision_hydro_click_from_scan(self):
        """
        Get last click from the last scan to the non-selected target.

        Can use info with this click for decision time or range to target, based on scans.

        Returns
        -------
        A pandas series containing info for the decision click based on hydrophone data and scans.
        """
        last_streak = self.df_click_scan["streak_id"] == self.df_scan.index.max() - 1
        return self.df_click_scan[last_streak].iloc[-1]

    def get_dtag_buzz_onset(self, th_num_buzz=30, th_ICI=13e-3):
        """
        Identify buzz onset using Dtag click detections.

        There has to be `>= th_num_buzz` number of consecutive clicks with ICI<th_ICI
        for a given streak to be consider onset of buzz.

        Parameters
        ----------
        th_num_buzz
            minimum number of clicks to quality as a buzz-like section
        th_ICI
            ICI threshold between regular clicks and buzzes

        Returns
        -------
        A pandas series containing info for the buzz onset click based on dtag data.
        """
        # get dataframe to manipulate
        df_dtag = self.df_dtag.copy()
        df_dtag["ICI"] = df_dtag["time_corrected"].diff()
        df_dtag = df_dtag[df_dtag["before_touch"]]

        # only want clicks at the end of trial
        df_dtag = df_dtag.loc[
            (df_dtag.iloc[-1]["time_corrected"] - df_dtag["time_corrected"]) < 5
        ]

        df_dtag_for_buzz = df_dtag[df_dtag["ICI"] < th_ICI].copy()

        df_dtag_for_buzz = hc.add_buzz_streak(df_dtag_for_buzz, th_ICI=th_ICI)
        df_dtag_for_buzz = hc.get_streaks(df_dtag_for_buzz, col_name="streak")
        df_dtag_for_buzz = hc.clean_streaks(
            df_dtag_for_buzz, th_num_clk=th_num_buzz, type="ICI"
        )

        return df_dtag_for_buzz.iloc[0]

    def get_hydro_buzz_onset(self, th_num_buzz=30, th_ICI=13e-3):
        """
        Identify buzz onset using hydrophone click detections.

        There has to be `>= th_num_buzz` number of consecutive clicks with ICI<th_ICI
        for a given streak to be consider onset of buzz.

        Parameters
        ----------
        th_num_buzz
            minimum number of clicks to quality as a buzz-like section
        th_ICI
            ICI threshold between regular clicks and buzzes

        Returns
        -------
        A pandas series containing info for the buzz onset click based on hydrophone data.
        """
        # only want clicks at the last 2 click streaks
        df_hydro_for_buzz = self.df_click_scan[
            self.df_click_scan["streak_id"].isin(self.df_scan.index[-2:])
        ].copy()

        # only want clicks at the end of trial
        df_hydro_for_buzz = hc.add_buzz_streak(df_hydro_for_buzz, th_ICI=th_ICI)
        df_hydro_for_buzz = hc.get_streaks(df_hydro_for_buzz, col_name="streak")
        df_hydro_for_buzz = hc.clean_streaks(
            df_hydro_for_buzz, th_num_clk=th_num_buzz, type="ICI"
        )

        return df_hydro_for_buzz.iloc[0]

    def get_inspection_angle_in_view(
        self,
        time_stop: float,
        th_RL: Union[int, float] = 140,
        time_binning_delta: float = 50e-3,
    ):

        """
        Get range of inspection angle when porpoise was in camera view.

        Parameters
        ----------
        time_stop
            end of time to count inspection angle in time_corrected
            e.g., decision time
        th_RL
            receive level (RL) threshold for clicks to consider
            default to 140 dB
        time_binning_delta
            binning interval for clicks in seconds
            default to 50 ms
        angle_bins
            bins for ensonification angle
            default to np.arange(0, 365, 2.5)

        Returns
        -------
        angle_h0, angle_h1
            two numpy arrays holding the inspection angle for channel 0 and 1.
            angle_h0/h1 can be empty if labeled track is too short and do not
            overlap with the times where clicks are detected in ch0/ch1.
        """

        def filter_clicks(df_h, th_RL, time_stop):
            """
            Filter clicks based on start/stop time and receive level.
            """
            return df_h[(df_h["time_corrected"] < time_stop) & (df_h["RL"] > th_RL)]

        # Filter hydro clicks
        df_h0 = filter_clicks(
            self.df_hydro_ch0.copy(), th_RL=th_RL, time_stop=time_stop
        )
        df_h1 = filter_clicks(
            self.df_hydro_ch1.copy(), th_RL=th_RL, time_stop=time_stop
        )

        # Binning by time interval and get median of each bin
        tc_bins_h0 = ia.get_time_corrected_bins(
            df_h=df_h0, bin_delta=time_binning_delta
        )
        tc_bins_h1 = ia.get_time_corrected_bins(
            df_h=df_h1, bin_delta=time_binning_delta
        )
        angle_h0 = ia.groupby_ops(df_h0, "enso_angle", tc_bins_h0, "median").values.squeeze()
        angle_h1 = ia.groupby_ops(df_h1, "enso_angle", tc_bins_h1, "median").values.squeeze()

        return angle_h0, angle_h1

    def get_timing_last_scan_of_nonselect(self):
        """
        Get timing info of the last scan toward the non-selected target.

        Returns
        -------
        Start and end time of the last scan before decision.
        """

        def find_cut(th_click_gap):
            """
            th_click_gap
                used to ensure click trains broken up into multiple pieces are merged together
            """
            ICI = last_scan_nonchosen["time_corrected"].diff()
            return last_scan_nonchosen[ICI > th_click_gap]["time_corrected"].values

        last_streak = self.df_click_scan["streak_id"] == self.df_scan.index.max() - 1
        last_scan_nonchosen = self.df_click_scan[last_streak]

        inspect_end = last_scan_nonchosen["time_corrected"].iloc[-1]

        cut_time = find_cut(th_click_gap=0.11)
        if len(cut_time) == 0:  # no gap in scan
            inspect_start = last_scan_nonchosen["time_corrected"].iloc[0]
        else:
            inspect_start = cut_time
        if cut_time.size >= 1:
            if self.tp.trial_idx == 197:
                inspect_start = inspect_start[0]
            else:
                inspect_start = inspect_start[-1]

        # Store start and end time of last scan before decision
        self.last_scan_start = inspect_start
        self.last_scan_end = inspect_end

    def duration_last_scan_of_nonselect(self):
        """
        Get duration of the last scan toward the non-selected target.

        Returns
        -------
        Duration (in seconds) of the last scan toward the non-selected target.
        """
        # If timing info not available, run function to get info
        if self.last_scan_start is None and self.last_scan_end is None:
            self.get_timing_last_scan_of_nonselect()

        return self.last_scan_end - self.last_scan_start

    def angle_span_last_scan_of_nonselect(self):
        """
        Get span of inspection angle during the last scan toward the non-selected target.

        Returns
        -------
        Span of inspection angle of the last scan toward the non-selected target.
        """
        # If timing info not available, run function to get info
        if self.last_scan_start is None and self.last_scan_end is None:
            self.get_timing_last_scan_of_nonselect()

        # Get last scan clicks in the assigned scanned channel
        last_scan_clicks_in_ch = self.df_click_scan[
            (self.df_click_scan["time_corrected"] >= self.last_scan_start)
            & (self.df_click_scan["time_corrected"] <= self.last_scan_end)
            & (self.df_click_scan["CH"] == self.df_click_scan["scan_ch"])
        ]
        enso_angle_consider = last_scan_clicks_in_ch["enso_angle"]

        return enso_angle_consider.max() - enso_angle_consider.min()
