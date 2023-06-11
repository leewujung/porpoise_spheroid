"""
Core processing params and data paths
"""

from pathlib import Path

import numpy as np


# --------------------------------------
# Parameters that are fed into TrialProcessor.__init__ or used directly

# # Path to raw data
RAW_PATH = Path("/Volumes/SSD_2TB/MURI/fb2019_data")


# Paths to processed data
def generate_data_path_dict(main_path: Path):
    return {
        "main": main_path,
        # path to info csv
        "info_csv": main_path / "all_info_csv",
        # path to hydro clicks synced by LED flash
        "LED": main_path / "click_sync_LED/sync_csv/",
        # path to hydro clicks synced by chirp
        "chirp": main_path / "click_sync/sync_csv/",
        # path to calibrated tracks
        "track": main_path / "tracks/xypressure_cal_transformed",
        # path to calibrated target positions
        "target": main_path / "tracks/targets_cal_transformed",
        # path to extracted hydrophone clicks
        "extracted_clicks": main_path / "hydrophone/extracted_clicks",
    }


data_path_main = Path("/Volumes/SSD_2TB/MURI/fb2019_analysis")
# fb2019_analysis/info_csv does not exist but required for tests to work out
# do not want to change what's in fb2019_analysis
# because the structure is used in many previous analysis/exploration notebooks
DATA_PATH = generate_data_path_dict(main_path=data_path_main)


# Mapping of angle code with actual angle from Y-axis
ANGLE_MAP = dict(zip(np.arange(4) + 1, np.arange(4) * 45))



# --------------------------------------
# Processing parameters that are fed into individual TrialProcessor methods
# Those listed below are the default set

# Misc processing params
MISC_PARAMS = {    
    "th_RL": 140,  # receive level (RL) threshold for clicks to consider
    "time_binning_delta": 50e-3,  # binning interval for clicks in seconds
    "buzz_reg_switch": 13e-3,  # ICI threshold between buzz and regular clicks
    "num_buzz_for_onset": 30,  # minimum number of clicks to qualify as buzz onset
    "dist_max": ("DTAG_dist_elliptical", 12),  # track max dist selection criteria
    "dist_min": ("ROSTRUM_dist_to_target", 0.1),  # track min dist selection criteria
    "hydro_angle_th_SNR": 25,  # SNR threshold for computing inspection angle from clicks
    "hydro_angle_th_range": 6,  # minimum range (hand-marked range to target from Dtag) to be
                                # considered outside of camera view, in meters
}

# Hydrophone click processing params
HYDRO_PARAMS = {
    "bkg_len_sec": 32e-6,  # background length from the beginning of extracted click
    "clk_sel_len_sec": 128e-6,  # click length around the middle of extracted click vector
    "perc_before_pk": 30,  # percentage before peak to count as click
    "hydro_sens": -211,  # hydrophone sensitivity
    "recording_gain": 40,  # gain of hydro recording system
}

# Environmental params for sound absorption
# These parameters are set to match the absorption 0.04 dB m^-1 at 130 kHz
# specified in Malinka et al. 2021 JEB paper.
ENV_PARAMS = {
    "frequency": 130e3,
    "temperature": 16,
    "salinity": 28,
    "pressure": 1,
    "pH": 8,
    "absorption_formula_source": "FG",
}

# Scan params
SCAN_PARAMS = {
    # RL tolerance, below which do not switch the assigned scan channel from the previous click
    "RL_tolerance": 5,
    # minimum number of clicks to be considered a streak (stride of clicks)
    "th_num_clk": 5,
    # RL difference to accept as true scan (beam alternating between targets)
    "true_scan_th_RL_diff": 5,
    # max num of overlapping to be consider a true scan
    "true_scan_max_num_click_has_RL_diff": 3,
}