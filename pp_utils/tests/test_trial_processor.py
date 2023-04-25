import pickle
import numpy as np
import pandas as pd

from pp_utils.file_handling import df_master_loader
from pp_utils.trial_processor import TrialProcessor
from pp_utils.core import HYDRO_PARAMS, SCAN_PARAMS, ENV_PARAMS


def test_trial_processor(t100_flags, t100_params, t100_paths, test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)

    assert tp.flags == t100_flags
    assert tp.params == t100_params
    assert tp.paths == t100_paths

    for df_name in ["df_track", "df_dtag", "df_hydro_ch0", "df_hydro_ch1", "df_targets"]:
        assert isinstance(getattr(tp, df_name), pd.DataFrame)


def test_save_tp(t100_flags, t100_params, t100_paths, test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100
    save_path = "/Users/wujung/Downloads"

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)

    # Pickle tp object
    save_full_path = tp.save_trial(save_path)

    # Load saved object and compare with the original
    with open(save_full_path, "rb") as file_in:
        tp_load = pickle.load(file_in)

    assert isinstance(tp_load, TrialProcessor)
    assert tp_load.flags == t100_flags
    assert tp_load.params == t100_params
    assert tp_load.paths == t100_paths
    for df_name in ["df_track", "df_dtag", "df_hydro_ch0", "df_hydro_ch1", "df_targets"]:
        assert isinstance(getattr(tp_load, df_name), pd.DataFrame)
        assert getattr(tp_load, df_name).equals(getattr(tp, df_name))

    # Remove pickle file
    save_full_path.unlink()


def test_get_sampling_rate(test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)

    assert np.isclose(tp.fs_video, 29.97)
    assert tp.fs_dtag == 576000
    assert tp.fs_hydro == 500000
    

def test_add_track_features(test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)

    df_track_original = tp.df_track.copy()

    # Get timing from track
    tp.get_timing()
    # Attribute was added
    assert tp.touch_time_corrected is not None
    # New columns were added
    for attr in ["time", "time_corrected"]:
        assert attr in tp.df_track

    # Add track features
    tp.add_track_features()
    # New columns were added
    for track_label in ["DTAG", "ROSTRUM"]:
        for attr in [
            f"{track_label}_dist_to_target",
            f"{track_label}_dist_to_clutter",
            f"{track_label}_dist_elliptical",
            f"{track_label}_speed",
            "angle_heading_to_target",
            "angle_heading_to_clutter",
            "absolute_heading",
        ]:
            assert attr in tp.df_track
    # Track modified
    for attr in df_track_original.columns:
        assert not tp.df_track[attr].equals(df_track_original[attr])


def test_add_hydro_features(test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)
    tp.get_timing()  # this adds the "time_corrected" column to df_track required for interpolate_track_xy

    # Add hydro info
    tp.add_hydro_features()
    # New columns were added
    for attr in [
        "DTAG_X", "DTAG_Y", "ROSTRUM_X", "ROSTRUM_Y", "dist_to_hydro",
        "angle_yaxis_DTAG", "enso_angle", "angle_heading_to_hydro", "ICI",
    ]:
        assert attr in tp.df_hydro_ch0
        assert attr in tp.df_hydro_ch1

    # Add SNR and p2p
    tp.add_SNR_p2p(hydro_params=HYDRO_PARAMS)
    # New columns were added
    for attr in ["SNR", "p2p"]:
        assert attr in tp.df_hydro_ch0
        assert attr in tp.df_hydro_ch1

    # Add RL, ASL, and pointEL
    tp.add_RL_ASL_pointEL(env_params=ENV_PARAMS, hydro_params=HYDRO_PARAMS)
    # New columns were added
    for attr in ["RL", "ASL", "pointEL"]:
        assert attr in tp.df_hydro_ch0
        assert attr in tp.df_hydro_ch1


def test_add_before_touch_to_all_dfs(test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)
    tp.get_timing()  # this adds the "time_corrected" column to df_track required for interpolate_track_xy

    # Add "before_touch" column
    tp.add_before_touch_to_all_dfs()
    for df_name in ["df_track", "df_dtag", "df_hydro_ch0", "df_hydro_ch1"]:
        assert "before_touch" in getattr(tp, df_name)


def test_get_desired_track_portion(test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)
    tp.add_track_features()
    tp.get_timing()  # this adds the "time_corrected" column to df_track required for interpolate_track_xy
    tp.add_before_touch_to_all_dfs()

    # if track fits dist max/min criteria, check extract portion values
    df_track_extract = tp.get_desired_track_portion(
        dist_max=("DTAG_dist_elliptical", 8), dist_min=("ROSTRUM_dist_to_target", 1.1)
    )
    assert df_track_extract["DTAG_dist_elliptical"].max() < 8
    assert df_track_extract["ROSTRUM_dist_to_target"].min() > 1.1

    # if tracak does NOT fit dist max/min criteria, should return None
    df_track_extract = tp.get_desired_track_portion(
        dist_max=("DTAG_dist_elliptical", 12), dist_min=("ROSTRUM_dist_to_target", 0.1)
    )
    assert df_track_extract is None


def test_hydro_scan(test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)
    tp.get_timing()  # this adds the "time_corrected" column to df_track required for interpolate_track_xy
    tp.add_track_features()
    tp.add_hydro_features()
    tp.add_SNR_p2p(hydro_params=HYDRO_PARAMS)
    tp.add_RL_ASL_pointEL(hydro_params=HYDRO_PARAMS, env_params=ENV_PARAMS)
    tp.add_before_touch_to_all_dfs()

    tp.get_hydro_scan_num(th_RL=140, scan_params=SCAN_PARAMS)
    # Check 3 dfs should not be None
    assert tp.df_click_all is not None
    assert tp.df_click_scan is not None
    assert tp.df_scan is not None
    # Check length of dfs
    assert len(tp.df_click_all) == 914
    assert len(tp.df_click_scan) == 909
    assert len(tp.df_scan) == 11

    df_scan_ch0 = tp.sort_df_scan_to_channel(ch=0)
    df_scan_ch1 = tp.sort_df_scan_to_channel(ch=1)
    assert len(df_scan_ch0) == 6
    assert len(df_scan_ch1) == 6