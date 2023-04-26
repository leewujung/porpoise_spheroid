"""
A collection of convenient functions for pulling out files with different types of data.
"""

from typing import Tuple, Union, Dict, List
from pathlib import Path, PosixPath
import re

import numpy as np
import pandas as pd
import cv2

from .core import DATA_PATH


CAL_OBJ_STR = (
    "\d{8}_s\d{1}_t\d{1,2}_\w{8}_xypressure_(?P<cal_obj>\w+)_transformed.csv"  # noqa
)
CAL_OBJ_MATCHER = re.compile(CAL_OBJ_STR)


TRIAL_FILE_FLAGS = (
    "has_LED_or_chirp_sync",
    "can_identify_touch_frame",
    "has_target_file",
    "has_track_file",
    "has_dtag_clicks",
    "has_hydro_clicks_ch0",
    "has_hydro_clicks_ch1",
    "has_extracted_clicks_ch0",
    "has_extracted_clicks_ch1",
)
TRIAL_PARAMS = (
    "fname_prefix",
    "sync_source",  # {None, "LED", "chirp"}
    "idx_touch",  # index of which the animal touches target
    "chose_target",  # whether choice is correct, {True, False}
    "cal_obj",  # {"hula_flip" "hula_noflip", "cross"}
)

TRIAL_FILE_PATHS = {
    "target",  # path to target position file(s)
    "track",  # path to track file(s)
    "dtag_DTAG",  # path to dtag file, depending on sync source
    "dtag_ROSTRUM",
    "hydro_ch0_DTAG",  # path to file(s) with hydro click detections, depending on sync source
    "hydro_ch1_DTAG",
    "hydro_ch0_ROSTRUM",
    "hydro_ch1_ROSTRUM",
    "extracted_click_ch0",
    "extracted_click_ch1",
}


def df_master_loader(folder=DATA_PATH["main"], filename="analysis_master_info_append_09.csv"):
    """
    Read and add/modify info into the master dataframe.
    """
    # master dataframe with all info
    df_master = pd.read_csv(folder / filename, index_col=0)
    df_master["fname_prefix"] = df_master.apply(
        lambda x: "%s_s%s_t%s" % ((x["DATE"]), x["SESSION"], x["TRIAL"]), axis=1
    )
    df_master["TARGET_ANGLE"] = df_master.apply(
        lambda x: x["LEFT"] + x["RIGHT"] + str(x["ANGLE"]), axis=1
    )
    # # add a column for trial index
    # df_master["trial_index"] = np.arange(len(df_master))

    # Revise `TOUCH_FRAME` for trial 114 and 116
    # See 2022/04/04 notes.
    df_master.loc[
        (df_master["DATE"] == 20190628)
        & (df_master["SESSION"] == 3)
        & (df_master["TRIAL"] == 4),
        "TOUCH_FRAME",
    ] = 18959
    df_master.loc[
        (df_master["DATE"] == 20190628)
        & (df_master["SESSION"] == 3)
        & (df_master["TRIAL"] == 6),
        "TOUCH_FRAME",
    ] = 22510

    return df_master


def get_priority_cal_obj(track_files: List) -> Tuple[str, int]:
    """
    Select the priority calibration object if multiple exist.

    The "cross" cal object has the highest priority over "hula_flip" and "hula_noflip".
    """
    cal_obj_list = []
    for tr in track_files:
        cal_obj_list.append(CAL_OBJ_MATCHER.match(tr.name)["cal_obj"])

    # Use cross cal_obj if exist
    if "cross" in cal_obj_list:
        cal_obj_idx = int(np.argwhere(np.array(cal_obj_list) == "cross"))
    else:
        cal_obj_idx = 0

    return cal_obj_list[cal_obj_idx], cal_obj_idx


def get_cal_obj_and_track_filepath(fname_prefix: str, track_path: str) -> Tuple[Union[str, None], Union[PosixPath, None]]:
    """
    get file path(s) with tracks.

    Parameters
    ----------
    fname_prefix : str
        file name prefix
    track_path : str
        path to transformed xypressure files

    Returns
    -------
    cal_obj : str
        priority cal object
    track_file_wanted : str
        track file path corresponding tyo the priority cal object
    """
    # the "_" added is to prevent matching t10_*.csv when it should be t1_*csv
    track_file_sel = sorted(list(Path(track_path).glob(fname_prefix + "_*.csv")))

    if track_file_sel:
        cal_obj, cal_obj_idx = get_priority_cal_obj(track_file_sel)
        track_file_wanted = track_file_sel[cal_obj_idx]
    else:
        cal_obj = None
        track_file_wanted = None

    return cal_obj, track_file_wanted


def get_target_filepath(fname_prefix, cal_obj, target_path) -> Union[PosixPath, None]:
    """
    Get file path(s) containing target positions.
    """
    target_file = Path(target_path).joinpath(
        "%s_targets_%s_transformed.csv" % (fname_prefix, cal_obj)
    )
    return target_file if target_file.exists() else None


def get_hydro_filepath(fname_prefix: str, sync_path: str, flags: Dict) -> Tuple[str, str]:
    """
    Get file path(s) with detected hydro clicks and modify flags related to hydro clicks in place.
    """
    hydro_ch0_file = sync_path / f"{fname_prefix}_hydro_ch0.csv"
    hydro_ch1_file = sync_path / f"{fname_prefix}_hydro_ch1.csv"
    df_hydro_ch0 = pd.read_csv(hydro_ch0_file, index_col=0)
    df_hydro_ch1 = pd.read_csv(hydro_ch1_file, index_col=0)

    if len(df_hydro_ch0) > 0:
        flags["has_hydro_clicks_ch0"] = True  # modify flag in place
    else:
        flags["has_hydro_clicks_ch0"] = False
        hydro_ch0_file = None

    if len(df_hydro_ch1) > 0:
        flags["has_hydro_clicks_ch1"] = True  # modify flag in place
    else:
        flags["has_hydro_clicks_ch1"] = False
        hydro_ch1_file = None
    return hydro_ch0_file, hydro_ch1_file


def get_dtag_filepath(fname_prefix: str, sync_path: Path, flags: Dict):
    """
    Get file path with detected dtag clicks and modify flags related to dtag clicks in place.
    """
    dtag_file = sync_path / ("%s_dtag.csv" % fname_prefix)
    flags["has_dtag_clicks"] = True if dtag_file.exists() else False  # modify flag in place
    return dtag_file


def get_extracted_clk_filepath(fname_prefix: str, clk_path: Path, flags: Dict):
    """
    Get file path to extracted hydro clicks.
    """
    clk_file_ch0 = clk_path / f"{fname_prefix}_extracted_clicks_ch0.npy"
    clk_file_ch1 = clk_path / f"{fname_prefix}_extracted_clicks_ch1.npy"

    if clk_file_ch0.exists():
        flags[f"has_extracted_clicks_ch0"] = True  # modify flag in place
    else:
        flags[f"has_extracted_clicks_ch0"] = False
        clk_file_ch0 = None

    if clk_file_ch1.exists():
        flags[f"has_extracted_clicks_ch1"] = True  # modify flag in place
    else:
        flags[f"has_extracted_clicks_ch1"] = False
        clk_file_ch0 = None

    return clk_file_ch0, clk_file_ch1



def get_trial_info(df_master : pd.DataFrame, data_path: Dict, trial_idx: int) -> Tuple[Dict, Dict, Dict]:
    """
    Obtain file existence flags, file paths, and other params for a trial.

    Parameters
    ----------
    df_master : pd.DataFrame
        the master dataframe
    data_path : dict
        dictionary containing path for each file category, with the following entries:
    trial_idx : int
        sequential index of the trial to be selected in df_master

    Returns
    -------
    flags : dict
        whether or not some files are found
    params : dict
        params specific for this trial
    paths : dict
        paths to combination of files and metadata
    """
    # Inititate dicts
    flags = dict.fromkeys(TRIAL_FILE_FLAGS)
    params = dict.fromkeys(TRIAL_PARAMS)
    paths = dict.fromkeys(TRIAL_FILE_PATHS)

    # Get series for the trial
    ts = df_master.iloc[trial_idx]
    params["fname_prefix"] = ts["fname_prefix"]

    # Select click sync set: use LED sync if that exists
    if ts["trials_sync_at_least_chirp_or_LED"]:
        flags["has_LED_or_chirp_sync"] = True
        if ts["LED_trial_clicks_synced"]:
            params["sync_source"] = "LED"
            sync_path = data_path["LED"]
        else:  # use chirp sync results
            params["sync_source"] = "chirp"
            sync_path = data_path["chirp"]
    else:
        flags["has_LED_or_chirp_sync"] = False
        sync_path = None
        print("Clicks not synced!")

    # Get frame index when the animal touches the chosen object
    idx_touch = ts["TOUCH_FRAME"] - ts["FRAME_NUM_START"]
    if not np.isnan(idx_touch):
        flags["can_identify_touch_frame"] = True
        params["idx_touch"] = idx_touch.astype(int)  # convert to int since it is an index
    else:
        flags["can_identify_touch_frame"] = False
        print("Cannot identify touch frame in this video!")

    # Record animal choice
    params["chose_target"] = ts["CHOICE"] == 1

    # Get paths track files
    params["cal_obj"], paths["track"] = get_cal_obj_and_track_filepath(ts["fname_prefix"], data_path["track"])
    if paths["track"]:
        flags["has_track_file"] = True
    else:
        flags["has_track_file"] = False
        print("No track file!")

    # Get paths to target positions
    paths["target"] = get_target_filepath(
        ts["fname_prefix"], params["cal_obj"], data_path["target"]
    )
    if paths["target"]:
        flags["has_target_file"] = True
    else:
        flags["has_target_file"] = False
        print("No target file!")

    # Note for below:
    #  Within the functions flags["has_hydro_clicks_ch0/1"] and flags["has_dtag_clicks"]
    #  will be modified in place.
    #  Since the function is called twice, the flags will be overwritten the 2nd time.
    #  This is ok because the files come from the same source so file existence is consistent

    if sync_path is None:
        for flag_name in ["has_hydro_clicks_ch0", "has_hydro_clicks_ch1", "has_dtag_clicks"]:
            flags[flag_name] = False
    else:
        # Get paths to hydro detected clicks
        paths["hydro_ch0_DTAG"], paths["hydro_ch1_DTAG"] = get_hydro_filepath(
            ts["fname_prefix"], sync_path / "DTAG", flags
        )
        paths["hydro_ch0_ROSTRUM"], paths["hydro_ch1_ROSTRUM"] = get_hydro_filepath(
            ts["fname_prefix"], sync_path / "ROSTRUM", flags
        )

        # Get paths to dtag detected clicks
        paths["dtag_DTAG"] = get_dtag_filepath(ts["fname_prefix"], sync_path / "DTAG", flags)
        paths["dtag_ROSTRUM"] = get_dtag_filepath(ts["fname_prefix"], sync_path / "ROSTRUM", flags)

        # Set paths to hydrophone extracted clicks
        paths["extracted_click_ch0"], paths["extracted_click_ch1"] = get_extracted_clk_filepath(
            ts["fname_prefix"], data_path["extracted_clicks"], flags
        )

    return flags, params, paths


def assemble_target_df(target_file: str, cal_obj: str, target_pos_comb: str) -> pd.DataFrame:
    """
    Assemble target dataframe based on one cal_obj selection.

    Parameters
    ----------
    target_file : str
        path to target csv
    cal_obj : str
        the selected cal object, one of {"hula_flip" "hula_noflip", "cross"}
    target_pos_comb : str
        target position combination, "TC" or "CT"
    """    
    df_t = pd.read_csv(target_file, index_col=0)

    if target_pos_comb == "TC":
        df_t.rename(
            columns={
                "TOP_OBJECT_X": "target_" + cal_obj + "_pos_x",
                "TOP_OBJECT_Y": "target_" + cal_obj + "_pos_y",
                "TOP_OBJECT_Z": "target_" + cal_obj + "_pos_z",
                "BOTTOM_OBJECT_X": "clutter_" + cal_obj + "_pos_x",
                "BOTTOM_OBJECT_Y": "clutter_" + cal_obj + "_pos_y",
                "BOTTOM_OBJECT_Z": "clutter_" + cal_obj + "_pos_z",
            },
            inplace=True,
        )
    elif target_pos_comb == "CT":
        df_t.rename(
            columns={
                "TOP_OBJECT_X": "clutter_" + cal_obj + "_pos_x",
                "TOP_OBJECT_Y": "clutter_" + cal_obj + "_pos_y",
                "TOP_OBJECT_Z": "clutter_" + cal_obj + "_pos_z",
                "BOTTOM_OBJECT_X": "target_" + cal_obj + "_pos_x",
                "BOTTOM_OBJECT_Y": "target_" + cal_obj + "_pos_y",
                "BOTTOM_OBJECT_Z": "target_" + cal_obj + "_pos_z",
            },
            inplace=True,
        )
    else:
        df_t = None
    return df_t


def get_fs_video(vid_file):
    cap = cv2.VideoCapture(vid_file)
    fs_video = cap.get(cv2.CAP_PROP_FPS)
    return fs_video
