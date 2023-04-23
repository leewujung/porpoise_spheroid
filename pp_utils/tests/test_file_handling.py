from pp_utils.core import DATA_PATH
from pp_utils.file_handling import get_trial_info, df_master_loader, assemble_target_df


def test_get_trial_info(trial_100_flags, trial_100_params, trial_100_paths):

    df_master = df_master_loader()
    trial_idx = 100

    flags, params, paths = get_trial_info(
        df_master=df_master, data_path=DATA_PATH, trial_idx=trial_idx
    )

    assert flags == trial_100_flags
    assert params == trial_100_params
    assert paths == trial_100_paths


def test_assemble_target_df():
    # trial_idx 100
    target_file = DATA_PATH["target"] / "20190628_s2_t2_targets_hula_flip_transformed.csv"
    cal_obj = "hula_flip"
    target_pos_comb = "TC"

    df_targets = assemble_target_df(target_file, cal_obj, target_pos_comb)

    assert list(df_targets.columns) == [
        'index', 'DATE', 'SESSION', 'TRIAL', 'TARGETS_CSV_FILENAME',
        'target_hula_flip_pos_x', 'target_hula_flip_pos_y',
        'target_hula_flip_pos_z', 'clutter_hula_flip_pos_x',
        'clutter_hula_flip_pos_y', 'clutter_hula_flip_pos_z'
    ]
