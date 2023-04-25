import numpy as np
import pandas as pd


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
