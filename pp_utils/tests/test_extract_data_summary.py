"""
This script is used to debug running through the routines in extract_data_summary_20230425.ipynb
"""

from pp_utils.core import DATA_PATH, RAW_PATH, HYDRO_PARAMS, MISC_PARAMS, ENV_PARAMS, SCAN_PARAMS
from pp_utils.file_handling import df_main_loader
from pp_utils.misc import interp_xy
from pp_utils.trial_processor import TrialProcessor


# Process trial
trial_idx = 2
df_main = df_main_loader(folder=DATA_PATH["main"], filename="analysis_master_info_append_09.csv")
tp = TrialProcessor(df_main, trial_idx, data_path=DATA_PATH, raw_path=RAW_PATH)


# Add track and hydrophone features
tp.add_track_features()
tp.add_hydro_features()
tp.add_SNR_p2p(hydro_params=HYDRO_PARAMS)
tp.add_RL_ASL_pointEL(hydro_params=HYDRO_PARAMS, env_params=ENV_PARAMS)
tp.add_before_touch_to_all_dfs()

# Scan
tp.get_hydro_scan_num(th_RL=MISC_PARAMS["th_RL"], scan_params=SCAN_PARAMS)
df_scan_ch0 = tp.sort_df_scan_to_channel(ch=0)
df_scan_ch1 = tp.sort_df_scan_to_channel(ch=1)

# Decision
decision_click = tp.decision_hydro_click_from_scan()

# Buzz onset
buzz_onset_dtag = tp.get_dtag_buzz_onset(
    buzz_reg_switch=MISC_PARAMS["buzz_reg_switch"],
    num_buzz_for_onset=MISC_PARAMS["num_buzz_for_onset"]
)
buzz_onset_hydro = tp.get_hydro_buzz_onset(
    buzz_reg_switch=MISC_PARAMS["buzz_reg_switch"],
    num_buzz_for_onset=MISC_PARAMS["num_buzz_for_onset"]
)

# Positions
decision_pos = interp_xy(decision_click["time_corrected"], tp.df_track)
buzz_onset_dtag_pos = interp_xy(buzz_onset_dtag["time_corrected"], tp.df_track)
buzz_onset_hydro_pos = interp_xy(buzz_onset_hydro["time_corrected"], tp.df_track)

# Inspection angle
angle_ch0_in_cam, angle_ch1_in_cam = tp.get_inspection_angle_in_view(
    time_stop=decision_click["time_corrected"],
    th_RL=MISC_PARAMS["th_RL"],
    time_binning_delta=MISC_PARAMS["time_binning_delta"]
)

# Last scan start/end time
tp.get_timing_last_scan_of_nonselect()

# track portion
track_portion_entry = tp.get_desired_track_portion(
    dist_max=MISC_PARAMS["dist_max"], dist_min=MISC_PARAMS["dist_min"]
)

print("Run through all functions no problem!")