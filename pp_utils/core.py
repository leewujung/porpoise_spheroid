"""
Core processing params and data paths
"""

from pathlib import Path

import numpy as np


# Path to raw data
RAW_PATH = Path("/Volumes/SSD_2TB/MURI/fb2019_data")


# Paths to processed data
data_path_main = Path("/Volumes/SSD_2TB/MURI/fb2019_analysis")
DATA_PATH = {
    "main": data_path_main,
    # paths to hydro clicks synced by LED flash
    "LED": data_path_main / "click_sync_LED/sync_csv/",
    # paths to hydro clicks synced by chirp
    "chirp": data_path_main / "click_sync/sync_csv/",
    # paths to calibrated tracks
    "track": data_path_main / "tracks/xypressure_cal_transformed",
    # paths to calibrated target positions
    "target": data_path_main / "tracks/targets_cal_transformed",
    "click_attr": data_path_main / "hydrophone_clicks/extracted_clicks_attr_df",
    "extracted_click": data_path_main / "hydrophone_clicks/extracted_clicks",
}


# ICI params
ICI_PARAMS = {
    "buzz_reg_switch": 13e-3,  # ICI threshold between buzz and regular clicks
    "num_buzz_for_onset": 30,  # minimum number of clicks to qualify as buzz onset
}


# # Decision params
# DECISION_PARAMS = {}
# DECISION_PARAMS["hydro_ASL_th"] = th_ASL


# Hydrophone click processing params
HYDRO_PARAMS = {
    "bkg_len_sec": 32e-6,  # background length from the beginning of extracted click
    "clk_sel_len_sec": 128e-6,  # click length around the middle of extracted click vector
    "perc_before_pk": 30,  # percentage before peak to count as click
    "hydro_sens": -211,  # hydrophone sensitivity
    "recording_gain": 40,  # gain of hydro recording system
}


# Mapping of angle code with actual angle from Y-axis
ANGLE_MAP = dict(zip(np.arange(4) + 1, np.arange(4) * 45))

