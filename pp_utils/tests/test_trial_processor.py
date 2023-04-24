import pickle
import numpy as np
import pandas as pd

from pp_utils.file_handling import df_master_loader
from pp_utils.trial_processor import TrialProcessor


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
    