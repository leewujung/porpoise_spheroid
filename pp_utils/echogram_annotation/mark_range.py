"""
A simple GUI to manually select points representing target echoes along range.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pp_utils.file_handling import df_main_loader
from pp_utils.trial_processor import TrialProcessor
from pp_utils.echogram_annotation.plot_func import plot_echogram_with_detection

import pickle


# Set paths and load main info df
analysis_path = Path("./data_processed")
raw_path = Path("./data_raw")

# Set paths and load main info df
egram_path = analysis_path / "echogram/batch_gen"  # pre-built echogram
tp_path = analysis_path / "data_summary/trial_processor_object"  # Trial processor object

output_path = analysis_path / "echogram/echogram_mark"
if not output_path.exists():
    output_path.mkdir()

# Load main dataframe
df_main = df_main_loader(
    folder=analysis_path / "all_info_csv", filename="main_info_append_09.csv"
)


def tellme(s, ax):
    print(s)
    ax.set_title(s, fontsize=16)
    plt.draw()


# Loop through all trials
for trial_idx in df_main.index:

    # Load TrialProcessor object
    tp_obj_fname = tp_path / f"trial_{trial_idx:03d}.pickle"
    with open(tp_obj_fname, "rb") as filein:
        tp: TrialProcessor = pickle.load(filein)

    egram_fname = egram_path / (
        f"t{trial_idx:03d}"
        f"_{tp.trial_series['DATE']}"
        f"_s{tp.trial_series['SESSION']}"
        f"_t{tp.trial_series['TRIAL']:02d}"
        "_dtag_filtered_bpf_echogram.npz"
    )

    with np.load(egram_fname) as egram:
        echogram = egram["echogram"]
        animal_range = egram["animal_range"]
        time_corrected = egram["time_corrected"]

    # Plot echogram
    with plt.ion():
        print("start plotting echogram")
        fig, ax = plot_echogram_with_detection(
            df_dtag=tp.df_dtag,
            df_hydro_ch0=tp.df_hydro_ch0,
            df_hydro_ch1=tp.df_hydro_ch1,
            df_track=tp.df_track,
            echogram_mtx=echogram,
            time_vec=time_corrected,
            range_vec=animal_range,
            figsize=(10, 8),
        )
        print("finish plotting echogram")

        tellme("Click to begin selecting animal ranges", ax=ax[0])

        plt.waitforbuttonpress()

        while True:
            pts = []
            while len(pts) < 3:
                tellme(
                    "Click the animal range you want to record, "
                    "hit 'Enter' to finish",
                    ax=ax[0],
                )
                pts = np.asarray(plt.ginput(-1, timeout=-1))
                print(pts)
                ax[0].plot(pts[:, 0], pts[:, 1], "mo")

            tellme("Happy? Key click for yes, mouse click for no", ax=ax[0])

            if plt.waitforbuttonpress():
                np.save(
                    # [:-13] to conform with previous filename
                    output_path / (egram_fname.stem[:-13] + "_animal_range.npy"),
                    pts,
                    allow_pickle=False,
                )
                plt.close()
                break
