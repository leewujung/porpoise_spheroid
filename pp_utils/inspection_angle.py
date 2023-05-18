import numpy as np
import pandas as pd
import xarray as xr


def get_time_corrected_bins(df_h: pd.DataFrame, bin_delta=0.05):
    """
    Get bins to pd.cut time_corrected.

    Parameters
    ----------
    df_h : pd.DataFrame
        df_hydro_ch0/1
    bin_delta : num
        bin width in seconds
    """
    time_bins = np.arange(
        0,
        (df_h["time_corrected"].values[-1] - df_h["time_corrected"].values[0])
        + bin_delta,
        bin_delta,
    )
    return pd.cut(
        x=df_h["time_corrected"],
        bins=time_bins + df_h["time_corrected"].values[0],
        include_lowest=True,  # make first interaval left-inclusive
    )


def groupby_ops(df_h: pd.DataFrame, col_name: str, time_corrected_bins: pd.cut, count_type: str):
    """
    Compute mean or median of pd.cut columns.

    Parameters
    ----------
    df_h : pd.DataFrame
        df_hydro_ch0/1
    col_name : str
        name of column to compute mean or median
    time_corrected_bins : pd.cut object
        pd.cut bins along time_corrected
    count_type : str
        {"mean", "median"}
    """
    if count_type == "mean":
        s = df_h[col_name].groupby(time_corrected_bins).mean()
        s.name = col_name + "_mean"
    else:
        s = df_h[col_name].groupby(time_corrected_bins).median()
        s.name = col_name + "_median"
    return s.to_frame().dropna()


def get_angle_dist(
    df_click,
    col_name,
    angle_bin=np.arange(0, 365, 5),
    time_binning=False,
    count_type="median",
    bin_delta=None,
    density=False,
):
    """
    Get angle

    Parameters
    ----------
    df_click : pd.DataFrame
        filtered dataframes containing hydrophoen click detections to be used
        to figure out distribution
    angle_bin : np.ndarray
        bins for ensonification angle
    col_name : str
        name of column to compute mean or median
        - "enso_angle" for in-cam-view angles
        - "theta_target" for out-of-cam angle
    count_type : str
        {"mean", "median"}
    time_binning : bool
        whether or not to count clicks based on time bins
        True: use binned median, False: use raw counts
        default is no binning (False)
    bin_delta : num
        bin width in seconds if time_binning is used
    density : bool
        whther or not distribution is normalized to be density

    Returns
    -------
    A numpy array containing the click counts along the input vector enso_bin
    """
    # Check input combination
    if time_binning is False and bin_delta is not None:
        raise ValueError("bin_delta cannot be used when time_binning is False!")

    # if input is empty
    if df_click is None or len(df_click) == 0:
        return np.zeros(len(angle_bin) - 1)
    else:
        if time_binning:  # count clicks by interval
            tc_bins = get_time_corrected_bins(df_h=df_click, bin_delta=bin_delta)
            enso_angle = groupby_ops(df_click, col_name, tc_bins, count_type).values
            out, _ = np.histogram(enso_angle, bins=angle_bin, density=density)
        else:  # count clicks as is
            out, _ = np.histogram(df_click[col_name], bins=angle_bin, density=density)
        return out


def enso2spheroid_angle(enso_angle):
    """
    Convert ensonification angle (enso_angle) to the angle convention
    used for spheroid echo measurements.

    Parameters
    ----------
    enso_angle : float
        ensonification angle

    Returns
    -------
    sph_echo_angle : float
        angles used in spheroid echo measurements
    """
    # convert to array if input a single number
    if isinstance(enso_angle, (int, np.integer, float)):
        enso_angle = np.expand_dims(np.array(enso_angle), 0)

    idx_leq_180 = enso_angle <= 180  # flipping around 180 deg
    sph_echo_angle = (360 - enso_angle) - 90
    sph_echo_angle[idx_leq_180] = enso_angle[idx_leq_180] - 90

    return sph_echo_angle.squeeze()


def get_spheroid_echo(sph_echo_angle, sph_ar, sph_path, sph_file):
    """
    Get spheroid echo measurements with angles closest to
    those in the input sph_echo_angle.

    The input angles are discretizied at 15 deg intervals
    when selecting the correpsonding echo measurements
    due to constraints from the experiment (each angle took
    too long so only measured at 15 deg intervals).

    Parameters
    ----------
    sph_echo_angle : float
        angles used in spheroid echo measurements
    sph_ar : str
        aspect ratio of spheroid
        allowed values: "1.0", "1.3", "2.9"

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing spheroid echo measurements
        at angles closest to the input angles.
    """
    da_sph = xr.open_dataarray(sph_path / sph_file[sph_ar])
    return da_sph.sel(angle=sph_echo_angle, method="nearest")
