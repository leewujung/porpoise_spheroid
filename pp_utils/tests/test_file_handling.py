from pp_utils.file_handling import get_trial_info, df_main_loader, assemble_target_df


def test_get_trial_info(t100_flags, t100_params, t100_paths, test_data_path):

    df_main = df_main_loader(folder=test_data_path["info_csv"])
    trial_idx = 100

    flags, params, paths = get_trial_info(
        df_main=df_main, data_path=test_data_path, trial_idx=trial_idx
    )

    assert flags == t100_flags
    assert params == t100_params
    assert paths == t100_paths


def test_assemble_target_df(test_data_path):
    # trial_idx 100
    target_file = test_data_path["target"] / "20190628_s2_t2_targets_hula_flip_transformed.csv"
    cal_obj = "hula_flip"
    target_pos_comb = "TC"

    df_targets = assemble_target_df(target_file, cal_obj, target_pos_comb)

    assert list(df_targets.columns) == [
        'index', 'DATE', 'SESSION', 'TRIAL', 'TARGETS_CSV_FILENAME',
        'target_hula_flip_pos_x', 'target_hula_flip_pos_y',
        'target_hula_flip_pos_z', 'clutter_hula_flip_pos_x',
        'clutter_hula_flip_pos_y', 'clutter_hula_flip_pos_z'
    ]
