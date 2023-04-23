from pp_utils.core import DATA_PATH
from pp_utils.file_handling import get_trial_info, df_master_loader

import pandas as pd


flags_expected = {
    "has_LED_or_chirp_sync": True,
    "can_identify_touch_frame": True,
    "has_track_file": True,
    "has_target_file": True,
    "has_dtag_clicks": True,
    "has_hydro_clicks_ch0": True,
    "has_hydro_clicks_ch1": True,
}

params_expected = {
    "fname_prefix": "20190628_s2_t2",
    "sync_source": "chirp",  # {None, "LED", "chirp"}
    "idx_touch": 273,  # index of which the animal touches target
    "chose_target": False,  # whether choice is correct, {True, False}        
    "cal_obj": "hula_flip",  # {"hula_flip" "hula_noflip", "cross"}
}

paths_expected = {
    "target": DATA_PATH["target"] / "20190628_s2_t2_targets_hula_flip_transformed.csv",
    "track": DATA_PATH["track"] / "20190628_s2_t2_GOPR1476_xypressure_hula_flip_transformed.csv",
    "dtag_DTAG": DATA_PATH["chirp"] / "DTAG/20190628_s2_t2_dtag.csv",
    "dtag_ROSTRUM": DATA_PATH["chirp"] / "ROSTRUM/20190628_s2_t2_dtag.csv",
    "hydro_ch0_DTAG": DATA_PATH["chirp"] / "DTAG/20190628_s2_t2_hydro_ch0.csv",
    "hydro_ch1_DTAG": DATA_PATH["chirp"] / "DTAG/20190628_s2_t2_hydro_ch1.csv",
    "hydro_ch0_ROSTRUM": DATA_PATH["chirp"] / "ROSTRUM/20190628_s2_t2_hydro_ch0.csv",
    "hydro_ch1_ROSTRUM": DATA_PATH["chirp"] / "ROSTRUM/20190628_s2_t2_hydro_ch1.csv",
}

def test_get_trial_info():

    df_master = df_master_loader()
    trial_idx = 100

    flags, params, paths = get_trial_info(
        df_master=df_master, data_path=DATA_PATH, trial_idx=trial_idx
    )

    assert flags == flags_expected
    assert params == params_expected
    assert paths == paths_expected