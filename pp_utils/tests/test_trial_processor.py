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


def test_hydro_scan_decision(test_data_path, test_raw_path):

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

    decision_click = tp.decision_hydro_click_from_scan()
    assert isinstance(decision_click, pd.Series)
    assert np.isclose(decision_click["time_corrected"], 791.236871)


def test_get_dtag_buzz_onset(test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)
    tp.get_timing()  # this adds the "time_corrected" column to df_track required for interpolate_track_xy
    tp.add_before_touch_to_all_dfs()

    # tp.get_dtag_buzz_onset needs "before_touch" column
    buzz_reg_switch = 13e-3
    num_buzz_for_onset = 30
    buzz_onset_time = tp.get_dtag_buzz_onset(buzz_reg_switch, num_buzz_for_onset)

    assert isinstance(buzz_onset_time, pd.Series)
    assert np.isclose(buzz_onset_time["time_corrected"], 792.181)
    assert buzz_onset_time["sample_loc"] == 9800441
    df_dtag = tp.df_dtag.copy()
    df_dtag["ICI"] = df_dtag["time_corrected"].diff()
    df_dtag = df_dtag[df_dtag["before_touch"]]
    assert np.all(
        df_dtag[df_dtag["sample_loc"] > 9800441]["ICI"][:num_buzz_for_onset+10].values
        < buzz_reg_switch
    )


def test_get_hydro_buzz_onset(test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)
    tp.get_timing()  # this adds the "time_corrected" column to df_track required for interpolate_track_xy
    tp.add_before_touch_to_all_dfs()

    # tp.get_hydro_buzz_onset needs tp.df_click_scan
    tp.add_hydro_features()
    tp.add_SNR_p2p(hydro_params=HYDRO_PARAMS)
    tp.add_RL_ASL_pointEL(hydro_params=HYDRO_PARAMS, env_params=ENV_PARAMS)
    tp.get_hydro_scan_num(th_RL=140, scan_params=SCAN_PARAMS)
    
    buzz_reg_switch = 13e-3
    num_buzz_for_onset = 30
    buzz_onset_time = tp.get_hydro_buzz_onset(buzz_reg_switch, num_buzz_for_onset)

    assert isinstance(buzz_onset_time, pd.Series)
    assert np.isclose(buzz_onset_time["time_corrected"], 792.182)
    assert buzz_onset_time["sample_loc"] == 7892284
    assert np.all(
        tp.df_click_scan[tp.df_click_scan["sample_loc"] > 7892284]["ICI"][:num_buzz_for_onset+10]
        < buzz_reg_switch
    )


def test_get_inspection_angle_in_view(test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)
    tp.get_timing()  # this adds the "time_corrected" column to df_track required for interpolate_track_xy

    #  RL is needed for selecting clicks in tp.get_inspection_angle_in_view
    tp.add_hydro_features()
    tp.add_SNR_p2p(hydro_params=HYDRO_PARAMS)
    tp.add_RL_ASL_pointEL(hydro_params=HYDRO_PARAMS, env_params=ENV_PARAMS)

    # stop_time = 791.236871 from test_hydro_scan_decision
    angle_ch0, angle_ch1 = tp.get_inspection_angle_in_view(
        time_stop=791.236871, th_RL=140, time_binning_delta=50e-3
    )

    assert isinstance(angle_ch0, np.ndarray)
    assert isinstance(angle_ch1, np.ndarray)
    assert len(angle_ch0) == 50
    assert len(angle_ch1) == 62


def test_last_scan(test_data_path, test_raw_path):

    df_master = df_master_loader(folder=test_data_path["main"])
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx, data_path=test_data_path, raw_path=test_raw_path)
    tp.get_timing()  # this adds the "time_corrected" column to df_track required for interpolate_track_xy

    # tp.df_click_scan is needed for last scan calculations
    tp.add_track_features()
    tp.add_hydro_features()
    tp.add_SNR_p2p(hydro_params=HYDRO_PARAMS)
    tp.add_RL_ASL_pointEL(hydro_params=HYDRO_PARAMS, env_params=ENV_PARAMS)
    tp.add_before_touch_to_all_dfs()
    tp.get_hydro_scan_num(th_RL=140, scan_params=SCAN_PARAMS)

    tp.get_timing_last_scan_of_nonselect()
    assert tp.last_scan_start is not None
    assert tp.last_scan_end is not None

    duration_last_scan = tp.duration_last_scan_of_nonselect()
    assert duration_last_scan == tp.last_scan_end - tp.last_scan_start

    angle_span_last_scan = tp.angle_span_last_scan_of_nonselect()
    assert np.isclose(angle_span_last_scan, 8.811212)