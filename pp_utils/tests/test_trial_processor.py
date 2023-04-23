import pandas as pd

from pp_utils.file_handling import df_master_loader
from pp_utils.trial_processor import TrialProcessor


def test_trial_processor(trial_100_flags, trial_100_params, trial_100_paths):

    df_master = df_master_loader()
    trial_idx = 100

    tp = TrialProcessor(df_master, trial_idx)

    assert tp.flags == trial_100_flags
    assert tp.params == trial_100_params
    assert tp.paths == trial_100_paths    

    for df_name in ["df_track", "df_dtag", "df_hydro_ch0", "df_hydro_ch1", "df_targets"]:
        assert isinstance(getattr(tp, df_name), pd.DataFrame)
