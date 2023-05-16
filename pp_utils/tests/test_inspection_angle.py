import numpy as np
from pp_utils.inspection_angle import enso2spheroid_angle


def test_enso2spheroid_angle():

    assert enso2spheroid_angle(120) == 30
    assert enso2spheroid_angle(135) == 45
    assert enso2spheroid_angle(180) == 90
    assert enso2spheroid_angle(45) == -45
    assert enso2spheroid_angle(0) == -90
    assert enso2spheroid_angle(60) == -30
    assert enso2spheroid_angle(255) == 15

    assert np.array_equal(
        enso2spheroid_angle(np.arange(45, 225 + 15, 15)),
        np.hstack((np.arange(-45, 90 + 5, 15), np.array([75, 60, 45]))),
    )
