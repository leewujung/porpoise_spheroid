import pandas as pd

from pp_utils.core import DATA_PATH
from pp_utils.file_handling import df_main_loader
from pp_utils.click_synchronizer import ClickSynchronizer




def test_click_synchronizer_before_sync(test_data_path, test_raw_path):

    df_main = df_main_loader(
        folder=test_data_path["info_csv"], filename="main_info_append_09.csv"
    )

    c = ClickSynchronizer(
        df_main=df_main,
        trial_idx=100,
        raw_path=test_raw_path,
        data_path=test_data_path,
    )
    c.get_filenames()
    c.get_sampling_rate()

    # synchronization requires the full dtag wav file that is too large
    # so here only testing the path setting and file finding mechanisms
    assert isinstance(c.source_path, dict)
    for attr in [
            "fs_hydro", "fs_dtag", "fs_video", 
            "hydro_file", "dtag_file", "track_file"
        ]:
        assert getattr(c, attr) is not None


def test_click_synchronizer_all(test_data_path):

    df_main = df_main_loader(
        folder=test_data_path["info_csv"], filename="main_info_append_09.csv"
    )

    c = ClickSynchronizer(df_main=df_main, trial_idx=30)
    c.get_filenames()
    c.get_sampling_rate()
    df_dtag, df_hydro_ch0, df_hydro_ch1, _, _, _ = c.sync_curr_trial(
        track_label="DTAG", plot_opt=False
    )
    assert isinstance(df_dtag, pd.DataFrame)
    assert isinstance(df_hydro_ch0, pd.DataFrame)
    assert isinstance(df_hydro_ch1, pd.DataFrame)
