import numpy as np


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