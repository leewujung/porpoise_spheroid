import pandas as pd

from pp_utils.core import DATA_PATH
from pp_utils.file_handling import df_master_loader
from pp_utils.click_synchronizer import ClickSynchronizer


df_master = df_master_loader(folder=DATA_PATH["main"])


def test_click_synchronizer():
    c = ClickSynchronizer(df_master=df_master, trial_idx=30)
    c.get_filenames()
    c.get_sampling_rate()
    df_dtag, df_hydro_ch0, df_hydro_ch1, _, _, _ = c.sync_curr_trial(
        track_label="DTAG", plot_opt=False
    )
    assert isinstance(df_dtag, pd.DataFrame)
    assert isinstance(df_hydro_ch0, pd.DataFrame)
    assert isinstance(df_hydro_ch1, pd.DataFrame)
