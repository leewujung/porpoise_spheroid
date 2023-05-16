import numpy as np
import pandas as pd
import scipy.interpolate as interpolate


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


def interpolate_track_xy(df_in, df_out, track_label):
    """
    Interpolate track X,Y values.

    >>> df_dtag["ROSTRUM_X"], df_dtag["ROSTRUM_Y"] = interpolate_track_xy(
            df_in=df_track, df_out=df_dtag, track_label="ROSTRUM"
        )
    """
    if len(df_out) == 0:  # if no entry in the df to be interpolated to
        return None, None
    else:
        fx = interpolate.interp1d(
            df_in["time_corrected"], df_in[track_label + "_X"], fill_value="extrapolate"
        )
        fy = interpolate.interp1d(
            df_in["time_corrected"], df_in[track_label + "_Y"], fill_value="extrapolate"
        )
        out_x = fx(df_out["time_corrected"])
        out_y = fy(df_out["time_corrected"])
        return out_x, out_y


def get_dist_to_object(df, df_targets, obj_type, cal_obj, track_label):
    """
    Compute distance between the positions contained in a dataframe to the selected object.

    Parameters
    ----------
    df : pd.Dataframe
        a pandas dataframe containing position info
    df_targets : pd.Dataframe
        a pandas dataframe containing target position info
    obj_type : str
        type of object to calculate distance from: 'target' or 'clutter'
    cal_obj : str
        type of calibration object: 'cross', 'hula_flip', or 'hula_noflip'
    track_label : str
        use position from which body part of porpoise

    Returns
    -------
        Distance between the selected track_label to the selected object
    """
    return np.linalg.norm(
        np.array(
            [
                (
                    df[track_label + "_X"].values
                    - df_targets["%s_%s_pos_x" % (obj_type, cal_obj)].values
                ),
                (
                    df[track_label + "_Y"].values
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


def get_angle_heading_to_object(df, df_targets, obj_type, cal_obj):
    """
    Compute the angle between heading and the vector from the DTAG marker
    to the selected object.

    Here heading is defined by the vector from the DTAG to ROSTRUM marker.

    Parameters
    ----------
    df : pd.Dataframe
        a pandas dataframe containing position info
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
        - df[["DTAG_X", "DTAG_Y"]].values
    )
    vec_obj = vec_obj / np.expand_dims(np.linalg.norm(vec_obj, axis=1), axis=1)

    # heading vec
    vec_heading = (
        df[["ROSTRUM_X", "ROSTRUM_Y"]].values
        - df[["DTAG_X", "DTAG_Y"]].values
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


def get_angle_from_yaxis(df_hydro, df_targets, obj_type, cal_obj, track_label):
    """
    Compute angle between y-axis (the line connecting target and clutter,
    pointing upward) and a vector from the object to the animal position
    while clicking.

    The animal position in df_hydro is either the DTAG or ROSTRUM position
    interpolated outside of this function based on the right cal_obj.

    Parameters
    ----------
    df_hydro : pd.Dataframe
        a pandas dataframe containing hydrophone detected clicks
    df_target : pd.Dataframe
        a pandas dataframe containing target position info
    track_label : str {"DTAG", "ROSTRUM"}
        track marker
    obj_type : str {"target", "clutter"}
        type of object to calculate distance from
    cal_obj : str {"cross", "hula_flip", "hula_noflip"}
        type of calibration object

    Returns
    -------
    Inspection angles to the target or clutter object [degrees]
    """
    # Get positions
    animal_pos = df_hydro[[f"{track_label}_X", f"{track_label}_Y"]].values
    obj_pos = df_targets[
        ["%s_%s_pos_x" % (obj_type, cal_obj), "%s_%s_pos_y" % (obj_type, cal_obj)]
    ].values

    # Compute angles
    vec = animal_pos - obj_pos
    angle = np.angle(vec[:, 0] + 1j * vec[:, 1])

    return unwrap_angle(angle) / np.pi * 180 + 90


def get_ensonification_angle(x, angle_expt, angle_yaxis_name="angle_yaxis_DTAG"):
    """
    Get ensonification angle from the target/clutter perspective.

    The ensonification angle is computed from the vector from the center
    of the target pointing to the hydrophone wire, to the vector from
    the center of the target pointing to the animal position.

    This function is meant to be used with pandas.apply
    >>> df_hydro_ch0["enso_angle"] = df_hydro_ch0.apply(
        get_ensonification_angle,  axis=1,
        args=(ANGLE_MAP[df_main.iloc[trial_idx]["ANGLE"]], "angle_yaxis_DTAG")
    )

    Parameters
    ----------
    x : pd.Series
    angle_yaxis_name : str {"angle_yaxis_DTAG", "angle_yaxis_ROSTRUM"}
        name of column containing angle from Y-axis to animal
    angle_expt : int or float
        experimental angle condition, 0-45-90-135 deg
    """
    enso_angle = x[angle_yaxis_name] + angle_expt
    return enso_angle


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


def get_track_index_based_on_dist(
    df_track, dist_name_min, dist_name_max, dist_th_min=None, dist_th_max=None
):
    """
    Retrieving the start and end indices of the desired portion of track.

    Parameters
    ----------
    df_track : pd.DataFrame
        dataframe with track and various distance measures before touch time
    dist_name_max : str
        distance measure used to select track portion based on max dist threshold
    dist_name_min : str
        distance measure used to select track portion based on min dist threshold
    dist_th_min : float
        minimum threshold of distance
        if dist_th_min=None returning the last valid frame
    dist_th_max : float
        maximum threshold of distance
        if dist_th_max=None returning the first valid frame

    Returns
    -------
    idx_dist_max, idx_dist_min
        Indices to retrieve the portion of df_track for the specified trial_name
    """
    # First index satisfying the max dist threshold
    if dist_th_max is None:
        idx_dist_max = df_track[dist_name_max].first_valid_index()
    else:
        idx_dist_max = df_track[df_track[dist_name_max] <= dist_th_max].index[0]

    # Last index not satisfying the min dist threshold
    if dist_th_min is None:
        idx_dist_min = df_track[dist_name_min].last_valid_index()
    else:
        idx_dist_min = df_track[df_track[dist_name_min] > dist_th_min].index[-1]

    return idx_dist_min, idx_dist_max


def get_feature_array(
    list_df_track: list,
    feature_name: list,
    align: str = "left",
    normalize_time: bool = True,
    normalize_timebase: np.ndarray = np.arange(0, 1.01, 0.01),
    verbose:bool = False,
):
    """
    Gather desired feature into a numpy array.

    Parameters
    ----------
    list_df_track : list
        list of all retrieved df_track portion
    feature_name : list[str]
        a list of feature names to be retrieved
    align : {"left", "right"}
        align the feature array to the left (default) or right
    normalize : boolean
        whether or not to normalize time
        if time is not normalized, the returned feature array
        is aligned to the left or right and padded with NaN
    normalized_timebase : np.array
        timebase to normalize the time to
    verbose : bool
        whether to print out reset_index for each track dataframe

    Returns
    -------
    feature_array : np.ndarray
        the compiled feature array
    feature_len : np.ndarray or None
        a 1D array containing the feature length per trial if time is not normalized
        None if time if normalized
    """

    def check_index(df_track):
        if not isinstance(
            df_track.index,
            (pd.core.indexes.range.RangeIndex, pd.core.indexes.numeric.Int64Index),
        ):
            if verbose:
                print("Reset dataframe index to be linear from 0!")
            return df_track.reset_index()
        else:
            return df_track

    if normalize_time:
        feature_array = []
        for v in list_df_track:
            df_now = check_index(v.copy())
            df_now["normalized_time"] = (df_now.index - df_now.index[0]) / (
                len(df_now) - 1
            )
            f_feature = interpolate.interp1d(
                df_now["normalized_time"],
                df_now[feature_name],
                axis=0,  # specify axis for >1 feature
            )
            feature_array.append(f_feature(normalize_timebase))
        feature_array = np.array(feature_array)
        return feature_array, None
    else:
        # gather feature into a list
        feature_list = []
        for v in list_df_track:
            feature_list.append(v[feature_name].values)
        # convert list to padded array
        feature_len = [ff.shape[0] for ff in feature_list]
        feature_array = np.nan * np.ones(
            (len(feature_list), max(feature_len), len(feature_name))
        )
        for seq, f in enumerate(feature_list):
            if align == "left":
                feature_array[seq, : feature_len[seq], :] = f  # type: ignore
            else:
                feature_array[seq, max(feature_len) - feature_len[seq] :, :] = f  # type: ignore
        return feature_array, feature_len

