import pandas as pd


def get_before_touch_column(df, idx_touch=None, time_touch=None):
    """
    Get a column noting whether row is before animal touches the chosen object.

    Parameters
    ----------
    df : pd.Dataframe
        a dataframe containing info to determine the "before_touch" flag
    idx_touch : int (optional)
        index at which the animal touches the chosen object
    time_touch : float (optional)
        time at which the animal touches the chosen object [seconds]
    """
    # Check inputs
    if idx_touch is None and time_touch is None:
        raise ValueError("idx_touch and time_touch cannot both be None!")
    elif idx_touch is not None and time_touch is not None:
        print(
            "Default to using idx_touch (frame index-based) "
            "when both idx_touch and time_touch are provided."
        )
        idx_flag = True
    elif idx_touch is None:  # only time_touch provided
        idx_flag = False
    else:  # only idx_touch provided
        idx_flag = True

    if idx_flag:  # based on index
        if not isinstance(
            df.index,
            (pd.core.indexes.range.RangeIndex, pd.core.indexes.numeric.Int64Index),
        ):
            print("Reset dataframe index to be linear from 0!")
            df_tmp = df.reset_index()
        else:
            df_tmp = df
        df_tmp["before_touch"] = False
        df_tmp.loc[
            df_tmp.index <= idx_touch, "before_touch"
        ] = True  # include the TOUCH_FRAME
    else:
        df_tmp = df.copy()  # avoid modify in place
        df_tmp["before_touch"] = df_tmp["time_corrected"] <= time_touch
    return df_tmp["before_touch"]
