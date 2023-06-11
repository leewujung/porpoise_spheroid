import pytest

import numpy as np
import pandas as pd

from pp_utils.core import ANGLE_MAP
import pp_utils.track_features as tf


@pytest.mark.parametrize(
    ["df_hydro", "angle_yaxis_DTAG_expected"],
    [
        # 45 deg from Y-axis
        (pd.DataFrame([1, -1], ["DTAG_X", "DTAG_Y"]).T, 45),
        # 120 deg from Y-axis
        (pd.DataFrame([np.sqrt(3), 1], ["DTAG_X", "DTAG_Y"]).T, 120),
    ],
)
def test_angle_from_yaxis(df_hydro, angle_yaxis_DTAG_expected):
    """
    Test trial_features.get_angle_from_yaxis()
    """
    # target at the bottom, clutter at the top
    df_targets = pd.DataFrame(
        data=[[0, 0, 0, 1]],
        columns=["target_cross_pos_x", "target_cross_pos_y",
                 "clutter_cross_pos_x", "clutter_cross_pos_y"]
    )

    angle_yaxis_DTAG = tf.get_angle_from_yaxis(  # compute angle from Y-axis
        df_hydro=df_hydro,
        df_targets=df_targets,
        track_label="DTAG",
        obj_type="target",
        cal_obj="cross",
    )
    # Y-axis is inverted when computing angles due to camera view
    assert angle_yaxis_DTAG == angle_yaxis_DTAG_expected


@pytest.mark.parametrize(
    ["df_hydro", "angle_yaxis_DTAG_expected", "enso_angle_expected_list"],
    [
        # 45 deg from Y-axis
        (
            pd.DataFrame([1, -1], ["DTAG_X", "DTAG_Y"]).T,
            45,
            45 + np.array([0, 45, 90, 135]),
        ),
        # 120 deg from Y-axis
        (
            pd.DataFrame([np.sqrt(3), 1], ["DTAG_X", "DTAG_Y"]).T,
            120,
            120 + np.array([0, 45, 90, 135])
        ),
    ],
)
def test_ensonification_angle(df_hydro, angle_yaxis_DTAG_expected, enso_angle_expected_list):
    """
    Test track_features.get_ensonification_angle()
    """
    # target at the top, clutter at the bottom
    df_targets = pd.DataFrame(
        data=[[0, 1, 0, 0]],
        columns=["target_cross_pos_x", "target_cross_pos_y",
                 "clutter_cross_pos_x", "clutter_cross_pos_y"]
    )

    df_hydro["angle_yaxis_DTAG"] = tf.get_angle_from_yaxis(
        df_hydro=df_hydro,
        df_targets=df_targets,
        track_label="DTAG",
        obj_type="clutter",
        cal_obj="cross",
    )
    # Y-axis is inverted when computing angles due to camera view
    assert df_hydro["angle_yaxis_DTAG"].values == angle_yaxis_DTAG_expected

    # Compute ensonification angle
    # Ensonification angle is counted from Y-axis toward the X-axis
    for angle_index, enso_angle_expected in zip(
        [1, 2, 3, 4], enso_angle_expected_list
    ):
        enso_angle = df_hydro.apply(
            tf.get_ensonification_angle,
            axis=1,
            args=(
                ANGLE_MAP[angle_index],
                "angle_yaxis_DTAG",
            ),
        )
        assert enso_angle.values == enso_angle_expected
