import numpy as np
import pandas as pd
from pp_utils.core import MISC_PARAMS, HYDRO_PARAMS, ENV_PARAMS
from pp_utils.inspection_angle import enso2spheroid_angle, get_angle_dist
from pp_utils.file_handling import df_main_loader
from pp_utils.trial_processor import TrialProcessor



def test_enso2spheroid_angle():

    assert enso2spheroid_angle(120) == 30
    assert enso2spheroid_angle(135) == 45
    assert enso2spheroid_angle(180) == 90
    assert enso2spheroid_angle(45) == -45
    assert enso2spheroid_angle(0) == -90
    assert enso2spheroid_angle(60) == -30
    assert enso2spheroid_angle(255) == 15
    assert enso2spheroid_angle(90) == 0

    assert np.array_equal(
        enso2spheroid_angle(np.arange(45, 225 + 15, 15)),
        np.hstack((np.arange(-45, 90 + 5, 15), np.array([75, 60, 45]))),
    )


def test_inspection_angle(test_data_path, test_raw_path):

    df_main = df_main_loader(folder=test_data_path["info_csv"])
    trial_idx = 96

    tp = TrialProcessor(df_main, trial_idx, data_path=test_data_path, raw_path=test_raw_path)
    #  RL is needed for selecting clicks in tp.get_inspection_angle_in_view
    tp.add_hydro_features()
    tp.add_SNR_p2p(hydro_params=HYDRO_PARAMS)
    tp.add_RL_ASL_pointEL(hydro_params=HYDRO_PARAMS, env_params=ENV_PARAMS)

    # expected angle distributions
    h0_dist_expected = np.zeros((1, 72))
    h0_dist_expected[0, 21] = 6
    h0_dist_expected[0, 22] = 11
    h1_dist_expected = np.zeros((1, 72))
    h1_dist_expected[0, 17] = 11
    h1_dist_expected[0, 18] = 3
    h1_dist_expected[0, 20] = 16

    def filter_clicks(df_h, th_RL, time_stop):
        return df_h[(df_h["time_corrected"] < time_stop) & (df_h["RL"] > th_RL)]

    time_stop = 1219.7852333333333  # from previous test_inspection_angle
    df_ch0 = filter_clicks(tp.df_hydro_ch0, th_RL=MISC_PARAMS["th_RL"], time_stop=time_stop)
    df_ch1 = filter_clicks(tp.df_hydro_ch1, th_RL=MISC_PARAMS["th_RL"], time_stop=time_stop)

    ch0_dist = get_angle_dist(df_click=df_ch0, col_name="enso_angle")
    ch1_dist = get_angle_dist(df_click=df_ch1, col_name="enso_angle")

    assert (ch0_dist == h0_dist_expected).all()
    assert (ch1_dist == h1_dist_expected).all()
