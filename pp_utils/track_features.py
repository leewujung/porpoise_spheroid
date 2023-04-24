import numpy as np


def interp_smooth_track(df_track, nan_gap=10, std=3):
    """
    Fill in small gaps and smooth the track.

    Parameters
    ----------
    df_track : pd.Dataframe
        a pandas dataframe containing track info
    nan_gap : int
        max number of points to be interpolated
    std : int
        std using in Guassian smoothing window

    Returns
    -------
    pd.DataFrame
        Input dataframe with small gaps filld and smoothed.
    """
    return (
        df_track.interpolate(limit=nan_gap)
        .rolling(nan_gap, win_type="gaussian", center=True)
        .mean(std=std)
    )


def get_dist_to_object(df_track, df_targets, obj_type, cal_obj, track_label):
    """
    Compute distance between the selected track_label to the selected object
    along the track.

    Parameters
    ----------
    df_track : pd.Dataframe
        a pandas dataframe containing track info
    df_target : pd.Dataframe
        a pandas dataframe containing target position info
    obj_type : str
        type of object to calculate distance from: 'target' or 'clutter'
    cal_obj : str
        type of calibration object: 'cross', 'hula_flip', or 'hula_noflip'
    track_label : str
        label of body part of porpoise

    Returns
    -------
        Distance between the selected track_label to the selected object
    """
    return np.linalg.norm(
        np.array(
            [
                (
                    df_track[track_label + "_X"].values
                    - df_targets["%s_%s_pos_x" % (obj_type, cal_obj)].values
                ),
                (
                    df_track[track_label + "_Y"].values
                    - df_targets["%s_%s_pos_y" % (obj_type, cal_obj)].values
                ),
            ]
        ),
        axis=0,
    )


def get_signed_angle(vec1, vec2):
    """
    Compute the signed angle from vec1 to vec2.
    """
    dot = np.diag(vec1 @ vec2.T)
    det = vec1[:, 0] * vec2[:, 1] - vec2[:, 0] * vec1[:, 1]
    return np.arctan2(det, dot)


def unwrap_angle(angle):
    """
    Unwrap angle.
    """
    idx_notnan = ~np.isnan(angle)
    angles_unwrap = np.unwrap(angle[idx_notnan])
    angle[idx_notnan] = angles_unwrap
    return angle


def get_angle_heading_to_object(df_track, df_targets, obj_type, cal_obj):
    """
    Compute the angle between heading and the vector from the DTAG marker
    to the selected object.

    Here heading is defined by the vector from the DTAG to ROSTRUM marker.

    Parameters
    ----------
    df_track : pd.Dataframe
        a pandas dataframe containing track info
    df_target : pd.Dataframe
        a pandas dataframe containing target position info
    obj_type : str
        type of object to calculate distance from: 'target' or 'clutter'
    cal_obj : str
        type of calibration object: 'cross', 'hula_flip', or 'hula_noflip'

    Returns
    -------
    angle_to_object
        Unwrapped angles to the selected object
    """
    # vec to object
    vec_obj = (
        df_targets[
            ["%s_%s_pos_x" % (obj_type, cal_obj), "%s_%s_pos_y" % (obj_type, cal_obj)]
        ].values
        - df_track[["DTAG_X", "DTAG_Y"]].values
    )
    vec_obj = vec_obj / np.expand_dims(np.linalg.norm(vec_obj, axis=1), axis=1)

    # heading vec
    vec_heading = (
        df_track[["ROSTRUM_X", "ROSTRUM_Y"]].values
        - df_track[["DTAG_X", "DTAG_Y"]].values
    )

    # angles to object
    angle_to_obj = get_signed_angle(vec_heading, vec_obj)

    return unwrap_angle(angle_to_obj) / np.pi * 180  # unwrap angle


def get_absolute_heading(df_track):
    """
    Compute absolute heading with respect to -X axis.

    Here absolute heading is defined by the signed angle spanned
    from the DTAG-to-ROSTRUM vector to the [-1, 0] direction (-X direction).

    Parameters
    ----------
    df_track : pd.Dataframe
        a pandas dataframe containing track info

    Returns
    -------
    angle_heading
        Compute absolute heading with respect to -X axis
    """
    # heading vec
    vec_heading = (
        df_track[["ROSTRUM_X", "ROSTRUM_Y"]].values
        - df_track[["DTAG_X", "DTAG_Y"]].values
    )

    # vector along -X axis
    vec_minus_x = np.array([[-1, 0]])
    vec_minus_x = np.tile(vec_minus_x, (vec_heading.shape[0], 1))

    # absolute heading wrt to the -X axis
    angle_to_origin = get_signed_angle(vec_heading, vec_minus_x)

    return unwrap_angle(angle_to_origin)  # unwrap angle


def get_speed(df_track, track_label):
    """
    Compute the speed [m/s] of a track_label.

    Parameters
    ----------
    df_track : pd.Dataframe
        a pandas dataframe containing track info
    track_label : str
        label of body part of porpoise

    Returns
    -------
        Speed [m/s] of the track_label
    """
    return (
        np.linalg.norm(
            df_track[[track_label + "_X", track_label + "_Y"]].diff(), axis=1
        )
        / df_track["time_corrected"].diff()
    )
