import pytest

from pp_utils.core import DATA_PATH


@pytest.fixture
def trial_100_flags():
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
def trial_100_params():
    return {
        "fname_prefix": "20190628_s2_t2",
        "sync_source": "chirp",  # {None, "LED", "chirp"}
        "idx_touch": 273,  # index of which the animal touches target
        "chose_target": False,  # whether choice is correct, {True, False}        
        "cal_obj": "hula_flip",  # {"hula_flip" "hula_noflip", "cross"}
    }

@pytest.fixture
def trial_100_paths():
    return {
        "target": DATA_PATH["target"] / "20190628_s2_t2_targets_hula_flip_transformed.csv",
        "track": DATA_PATH["track"] / "20190628_s2_t2_GOPR1476_xypressure_hula_flip_transformed.csv",
        "dtag_DTAG": DATA_PATH["chirp"] / "DTAG/20190628_s2_t2_dtag.csv",
        "dtag_ROSTRUM": DATA_PATH["chirp"] / "ROSTRUM/20190628_s2_t2_dtag.csv",
        "hydro_ch0_DTAG": DATA_PATH["chirp"] / "DTAG/20190628_s2_t2_hydro_ch0.csv",
        "hydro_ch1_DTAG": DATA_PATH["chirp"] / "DTAG/20190628_s2_t2_hydro_ch1.csv",
        "hydro_ch0_ROSTRUM": DATA_PATH["chirp"] / "ROSTRUM/20190628_s2_t2_hydro_ch0.csv",
        "hydro_ch1_ROSTRUM": DATA_PATH["chirp"] / "ROSTRUM/20190628_s2_t2_hydro_ch1.csv",
    }
