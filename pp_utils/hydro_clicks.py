from typing import Union

import numpy as np
import pandas as pd

import soundfile as sf


def load_detected_click_dfs(clk_file):
    """
    Load detected hydro clicks from npy files.

    The loaded dataframes are not synched
    (as opposed to those in from click_sync).

    Parameters
    ----------
    clk_file : str
        filename of detected click npy file

    Returns
    -------
    df_ch0, df_ch1 : pd.DataFrame
        dataframe containing location and peak height info
        of the detected clicks
    """
    # load detected clicks
    pts = np.load(clk_file, allow_pickle=True)
    clk_locs = {ch: v for ch, v in enumerate(pts["location"])}
    clk_pks = {ch: v for ch, v in enumerate(pts["peak"])}

    # assemble channel locs-pks
    df_ch0 = pd.DataFrame(
        np.vstack((clk_locs[0], clk_pks[0])).T, columns=["locs", "pks"]
    )
    df_ch1 = pd.DataFrame(
        np.vstack((clk_locs[1], clk_pks[1])).T, columns=["locs", "pks"]
    )
    return df_ch0, df_ch1


def cut_out_click(
    clk_file,
    df_clk,
    click_seq,
    fs_hydro,
    click_len_sec=256e-6,
    perc_before_pk=50,
    ch=None,
):
    """
    Cut out 1 click from wav file.

    Parameters
    ----------
    clk_file : str
        hydrophone wav file
    df_clk : pd.DataFrame
        dataframe with click locs and pks info
    click_len_sec : int or float
        length of click to cut out [seconds]
    perc_before_pk : int or float
        percentage of click length to cut out before detected peak [%]
    click_seq : int
        sequence number of the click wanted
    fs_hydro : int or float
        hydrophone sampling rate [Hz]
    ch : int or None
        channel of the click to be cut out. None means all channels (default)
    """
    click_len_pt = click_len_sec * fs_hydro  # number of points for the specified length
    len_pt_offset = int(perc_before_pk / 100 * click_len_pt)
    click_len_pt = int(click_len_pt)  # convert to int (originally float)

    click_pk_pt = int(df_clk.iloc[click_seq, :]["locs"])  # peak location in point
    click, _ = sf.read(
        clk_file, frames=click_len_pt, start=click_pk_pt - len_pt_offset
    )  # read signal

    if ch is None:
        return click
    else:
        return click[:, ch]
    
    
def gather_clicks(
    clk_file, df_clk, fs_hydro, click_len_sec=256e-6, perc_before_pk=50, ch=0
):
    """
    Gather all clicks from a detection file.

    Parameters
    ----------
    clk_file : str
        hydrophone wav file
    df_clk : pd.DataFrame
        dataframe with click locs and pks info
    click_len_sec : int or float
        length of click to cut out [seconds]
    perc_before_pk : int or float
        percentage of click length to cut out before detected peak [%]
    fs_hydro : int or float
        hydrophone sampling rate [Hz]
    ch : int
        channel of the click to be cut out (default to 0).
    """
    clk_all = []
    for clk in df_clk.itertuples():
        click_test = cut_out_click(
            clk_file=clk_file,
            df_clk=df_clk,
            click_len_sec=click_len_sec,
            perc_before_pk=perc_before_pk,
            click_seq=clk.Index,
            fs_hydro=fs_hydro,
            ch=ch,
        )
        clk_all.append(click_test)
    return np.array(clk_all)


def get_clk_variance(clk_mtx, clk_sel_len_sec, perc_before_pk, fs_hydro):
    """
    Get variance from the click section in the extracted click matrix.

    Parameters
    ----------
    clk_mtx : np.array
        matrix containing clicks from gather_clicks
    clk_sel_len_sec : float
        length of click section around detection point [sec]
    perc_before_pk : int or float
        percentage to count before mid point (peak) to count as click
    fs_hydro : int or float
        hydrophone sampling rate [Hz]
    """
    clk_sel_len_pt = int(clk_sel_len_sec * fs_hydro)
    clk_sel_idx = (
        clk_mtx.shape[1] // 2
        + np.arange(clk_sel_len_pt)
        - int(clk_sel_len_pt * perc_before_pk / 100)
    )
    return np.var(clk_mtx[:, clk_sel_idx], axis=1)


def get_bkg_variance(clk_mtx, bkg_len_sec, fs_hydro):
    """
    Get variance from background section in the extracted click matrix.

    Parameters
    ----------
    clk_mtx : np.array
        matrix containing clicks from gather_clicks
    bkg_len_sec : float
        length of background section [sec]
    fs_hydro : int or float
        hydrophone sampling rate [Hz]
    """
    return np.var(clk_mtx[:, : int(bkg_len_sec * fs_hydro)], axis=1)


def get_clk_p2p(clk_mtx, clk_sel_len_sec, perc_before_pk, fs_hydro):
    """
    Get p2p voltage from the click section in the extracted click matrix.

    Parameters
    ----------
    clk_mtx : np.array
        matrix containing clicks from gather_clicks
    clk_sel_len_sec : float
        length of click section around detection point [sec]
    perc_before_pk : int or float
        percentage to count before mid point (peak) to count as click
    fs_hydro : int or float
        hydrophone sampling rate [Hz]
    """
    clk_sel_len_pt = int(clk_sel_len_sec * fs_hydro)
    clk_sel_idx = (
        clk_mtx.shape[1] // 2
        + np.arange(clk_sel_len_pt)
        - int(clk_sel_len_pt * perc_before_pk / 100)
    )
    clk = clk_mtx[:, clk_sel_idx]
    return clk.max(axis=1) - clk.min(axis=1)


def select_scan_clicks(df: pd.DataFrame, th_RL: Union[int, float]):
    """
    Select the clicks used to determine scans based on receive level (RL).
    """
    return df[df["before_touch"] & (df["RL"] > th_RL)].copy()


def match_clicks(df_h0: pd.DataFrame, df_h1: pd.DataFrame, time_name="time_corrected", max_TDOA=1 / 1500):
    """
    Match clicks across hydrophone channels.

    Parameters
    ----------
    df_h0 : pd.DataFrame
        dataframe holding click info from ch0
    df_h1 : pd.DataFrame
        dataframe holding click info from ch1
    time_name : str
        default column name for the common timebase for both channels
    max_TDOA : num
        maximum time difference of arrival, default to
        1/1500 since target-clutter distance is 1 m

    Returns
    -------
    df_h_all : pd.DataFrame
        dataframe holding all click info from BOTH channels and the matching info
        the matched click index within this dataframe is in column "the_other_ch_index"
    """
    # Add ch source
    df_h0["CH"] = 0
    df_h1["CH"] = 1

    # Concat and sort dataframe with all click info from both channels
    df_h_all = pd.concat((df_h0, df_h1))
    df_h_all = (
        df_h_all.sort_values(by=time_name)
        .reset_index()
        .rename(columns={"index": "index_in_ch"})
    )

    # Match clicks from Ch0
    fill_index_col = []
    for index_all, clk in df_h_all.iterrows():
        fill_index = None  # if no match fill None
        time_diff = clk[time_name] - df_h_all.drop(index_all)[time_name]
        min_idx = np.abs(time_diff).idxmin()
        min_val = np.abs(time_diff).min()
        if min_val < max_TDOA:
            fill_index = min_idx
        fill_index_col.append(fill_index)
    df_h_all["the_other_ch_index"] = fill_index_col

    return df_h_all


def assign_scan(df_h_all: pd.DataFrame, RL_tolerance: Union[int, float], print_opt=False):
    """
    Assign the channel (target) scanned by animal.

    Parameters
    ----------
    df_h_all : pd.DataFrame
        dataframe holding all click info from BOTH channels
        this will be output from ``match_clicks``
    RL_tolerance : num
        RL tolerance, below which do not switch the assigned scan channel
        from the previous click

    Returns
    -------
    df_h_all : pd.DataFrame
        the same dataframe as input but with column "scan_ch" added
    """

    def if_to_print(text):
        if print_opt:
            print(text)

    df_h_all_out = df_h_all.copy()

    # Loop to identify scans
    scan_ch = None  # current scan channel
    scan_ch_prev = None  # previous scan channel
    match_index_prev = None  # previous click index
    scan_ch_all = []  # list to hold all scan channel

    for seq, clk in df_h_all_out.iterrows():
        # if click matched in another ch
        # the first (is not None): data type is None if all entries are empty
        # the second: data type is NaN is some entries are not empty
        if clk["the_other_ch_index"] is not None and ~np.isnan(
            clk["the_other_ch_index"]
        ):
            # if current click is the match of the previous
            if match_index_prev is not None and match_index_prev == seq:
                scan_ch = scan_ch_prev
                if_to_print(f"{seq:03d}: scan_ch={scan_ch}: matched click")
            else:
                # do not switch channel from previous click if RL difference < tolerance
                if (
                    np.abs(
                        clk["RL"] - df_h_all_out.loc[clk["the_other_ch_index"], "RL"]
                    )
                    < RL_tolerance
                ):
                    scan_ch = scan_ch_prev if scan_ch_prev is not None else clk["CH"]
                    if_to_print(
                        f"{seq:03d}: scan_ch={scan_ch}: RL diff < tol, use previous scan_ch"
                    )
                else:
                    # switch ch if RL of the other ch is higher and diff > RL tolerance
                    if df_h_all_out.loc[clk["the_other_ch_index"], "RL"] > clk["RL"]:
                        scan_ch = df_h_all_out.loc[clk["the_other_ch_index"], "CH"]
                        if_to_print(
                            f"{seq:03d}: scan_ch={scan_ch}: RL diff > tol and the other ch higher"
                            "use the matched ch"
                        )
                    else:
                        scan_ch = clk["CH"]
                        if_to_print(
                            f"{seq:03d}: scan_ch={scan_ch}: RL diff > tol and this ch higher"
                        )
        else:
            scan_ch = clk["CH"]
            if_to_print(f"{seq:03d}: scan_ch={scan_ch}: no match")

        # Save before update
        scan_ch_prev = scan_ch
        match_index_prev = clk["the_other_ch_index"]
        scan_ch_all.append(scan_ch)

    df_h_all_out["scan_ch"] = scan_ch_all

    return df_h_all_out


def get_streaks(df: pd.DataFrame, col_name="scan_ch"):
    """
    Set streak (stride of clicks) info for each clicks.
    """
    df_out = df.copy()
    df_out["start_of_streak"] = df_out[col_name].ne(df_out[col_name].shift())
    df_out["streak_id"] = df_out["start_of_streak"].cumsum()
    df_out["streak_counter"] = df_out.groupby("streak_id").cumcount() + 1
    return df_out


def has_enough_consecutive_clicks(x: pd.DataFrame, th_num_clk: int, type="scan"):
    """
    Whether a particular streak has enough number of clicks to be a scan.

    Parameters
    ----------
    x
        Dataframe from groupby operation of streaks dataframe, see example below
    th_num_clk
        Minimum number of clicks to be a scan
    type
        "scan": use this function for determining scans ("RL_diff" must exist in x),
        "ICI": use this function for determining buzzes

    Examples
    --------
    >>> streak_valid = streaks.groupby("streak_id").apply(has_enough_consecutive_clicks)
    >>> streaks = streaks[
            streaks["streak_id"].isin(streak_valid[streak_valid].index.tolist())
        ].reset_index()
    """
    if type == "scan":
        # Number of non-matching clicks
        len_nomatch = len(x[x["RL_diff"].isna()])

        # Number of matching clicks
        xx = x.drop_duplicates(subset=["RL_diff"])
        len_nodup = len(xx[~xx["RL_diff"].isna()])

        return (len_nomatch + len_nodup) >= th_num_clk
    elif type == "ICI":
        # just count the number of clicks in streak
        return len(x) >= th_num_clk
    else:
        raise ValueError("Provided type not supported!")


def clean_streaks(df: pd.DataFrame, th_num_clk: int, type="scan"):
    """
    Clean up the first streak count by removing streaks with too few clicks.
    """
    df_out = df.copy()
    valid_idx = df_out.groupby("streak_id").apply(
        has_enough_consecutive_clicks, th_num_clk, type=type
    )
    df_out = df_out[
        df_out["streak_id"].isin(valid_idx[valid_idx].index.tolist())
    ].reset_index(drop=True)
    return df_out


def is_true_scan(x: pd.DataFrame, th_RL_diff: Union[int, float], max_num_click_has_RL_diff: int):
    """
    Whether a particular scan is a true scan
    or both targets were actually ensonified roughly equally.

    Parameters
    ----------
    x
        Dataframe from groupby operation of streaks dataframe, see example below
    th_RL_diff
        RL difference to accept as true scan (beam alternating between targets)
    th_num_click_has_RL_diff
        Max number of overlapping to be consider a true scan

    Examples
    --------
    >>> streaks.groupby("streak_id").apply(is_true_scan)
    """

    # make sure to have the right field
    if "RL_diff" not in x or "the_other_ch_index" not in x:
        raise ValueError("Input dataframe does not contain the required field!")

    x_has_RL_diff = x.dropna(subset=["RL_diff"])
    num_click_has_RL_diff = x_has_RL_diff["RL_diff"].unique().size
    num_click_no_RL_diff = len(x[x["RL_diff"].isna()])

    # # has to have >=2 consecutive clicks to form a scan
    # if len(x) < 2:
    #     return False  # False: <2 consecutive clicks

    # true scan if not many overlapping clicks
    if (
        num_click_has_RL_diff <= max_num_click_has_RL_diff
        or num_click_has_RL_diff / (num_click_has_RL_diff + num_click_no_RL_diff) < 0.5
    ):
        return True  # True: not many overlapping clicks

    else:  # if many overlapping clicks
        # if percentage of RL_diff > threshold is large
        RL_diff_val_unique = x_has_RL_diff.drop_duplicates(subset=["RL_diff"])["RL_diff"].values
        if (np.abs(RL_diff_val_unique) > th_RL_diff).sum() / len(RL_diff_val_unique) >= 0.5:
            return True  # True: RL_diff is large
        else:
            return False  # False: many overlapping clicks but RL_diff is small
        

def add_buzz_streak(df: pd.DataFrame, th_ICI: Union[int, float]):
    """
    Add sequential number for consecutive buzz-like streaks.
    """
    df_out = df.copy()
    df_out["click_spacing"] = df_out["time_corrected"].diff()

    stride_cnt = 0
    stride_all = []
    for _, r in df_out.iterrows():
        if np.isnan(r["click_spacing"]) or r["click_spacing"] >= th_ICI:
            stride_cnt += 1
        stride_all.append(stride_cnt)

    df_out["streak"] = stride_all

    return df_out
