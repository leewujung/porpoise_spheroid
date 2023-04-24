import pytest
from pathlib import Path


@pytest.fixture
def test_data_path():
    data_path_main = Path("pp_utils/test_data/fb2019_analysis")
    return {
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

@pytest.fixture
def test_raw_path():
    return Path("pp_utils/test_data/fb2019_data")

@pytest.fixture
def t100_flags():
    return {
        "has_LED_or_chirp_sync": True,
        "can_identify_touch_frame": True,
        "has_track_file": True,
        "has_target_file": True,
        "has_dtag_clicks": True,
        "has_hydro_clicks_ch0": True,
        "has_hydro_clicks_ch1": True,
    }

@pytest.fixture
def t100_params():
    return {
        "fname_prefix": "20190628_s2_t2",
        "sync_source": "chirp",  # {None, "LED", "chirp"}
        "idx_touch": 273,  # index of which the animal touches target
        "chose_target": False,  # whether choice is correct, {True, False}        
        "cal_obj": "hula_flip",  # {"hula_flip" "hula_noflip", "cross"}
    }

@pytest.fixture
def t100_paths(test_data_path):
    return {
        "target": test_data_path["target"] / "20190628_s2_t2_targets_hula_flip_transformed.csv",
        "track": test_data_path["track"] / "20190628_s2_t2_GOPR1476_xypressure_hula_flip_transformed.csv",
        "dtag_DTAG": test_data_path["chirp"] / "DTAG/20190628_s2_t2_dtag.csv",
        "dtag_ROSTRUM": test_data_path["chirp"] / "ROSTRUM/20190628_s2_t2_dtag.csv",
        "hydro_ch0_DTAG": test_data_path["chirp"] / "DTAG/20190628_s2_t2_hydro_ch0.csv",
        "hydro_ch1_DTAG": test_data_path["chirp"] / "DTAG/20190628_s2_t2_hydro_ch1.csv",
        "hydro_ch0_ROSTRUM": test_data_path["chirp"] / "ROSTRUM/20190628_s2_t2_hydro_ch0.csv",
        "hydro_ch1_ROSTRUM": test_data_path["chirp"] / "ROSTRUM/20190628_s2_t2_hydro_ch1.csv",
    }
