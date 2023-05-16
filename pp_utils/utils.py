import pandas as pd


def get_time_range_threshold(df_tr: pd.DataFrame, th_range: int):
    """
    Get time_corrected corresponding to the last track point satisfying range_th.
    """
    df_tr = df_tr[df_tr["before_touch"]]
    return df_tr[df_tr["DTAG_dist_elliptical"] > th_range]["time_corrected"].values[-1]

def filter_clicks_far(df_h: pd.DataFrame, th_RL: float, time_range_th: float, time_decision: float):
    """
    Select only clicks before the animal reached a certain range.
    """
    if time_decision < time_range_th:
        # decision before reaching threshold range
        return df_h[(df_h["time_corrected"] < time_decision) & (df_h["RL"] > th_RL)]
    else:
        return df_h[(df_h["time_corrected"] < time_range_th) & (df_h["RL"] > th_RL)]

def filter_clicks_close(
    df_h: pd.DataFrame, th_RL: float, time_range_th: float, time_decision: float, **kwargs
):
    """
    Select only clicks AFTER the animal reached a certain range.
    """
    df_h = df_h[(df_h["RL"] > th_RL) & (df_h["before_touch"])]
    
    return df_h[
        (df_h["time_corrected"] > time_range_th)  # within range threshold
        & (df_h["time_corrected"] < time_decision)  # before decision
    ]    

def filter_clicks_close_before_last_scan(
    df_h: pd.DataFrame, th_RL: float, time_range_th: float, time_last_scan_start: float, **kwargs
):
    """
    Select only clicks AFTER the animal reached a certain range but BEFORE the last scan.
    """
    df_h = df_h[(df_h["RL"] > th_RL) & (df_h["before_touch"])]
    
    return df_h[
        (df_h["time_corrected"] > time_range_th)  # within range threshold
        & (df_h["time_corrected"] < time_last_scan_start)  # before last scan
    ]

def sort_df_in_cluster(cluster_fp, df_all):
    """
    Consolidate entries in 1 cluster and
    sort according to target presentation angle
    """
    df_cluster = pd.concat(
        [df_all.loc[df_all["fname_prefix"]==fp] for fp in cluster_fp]
    )
    df_cluster.sort_values(by="TARGET_ANGLE", inplace=True)
    return df_cluster
