"""
Print all echogram and annotated position for later reference.
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

# Trial processor object
tp_path = analysis_path / "data_summary/trial_processor_object"

# Set paths and load master info df
egram_path = analysis_path / "echogram/batch_gen"
mark_path = analysis_path / "echogram/echogram_mark"

save_path = analysis_path / "echogram/echogram_prints"
if not save_path.exists():
    save_path.mkdir()

# Load main dataframe
df_main = df_main_loader(
    folder=analysis_path / "all_info_csv", filename="main_info_append_09.csv"
)


# Loop through all trials
for trial_idx in df_main.index:

    # Load TrialProcessor object
    tp_obj_fname = tp_path / f"trial_{trial_idx:03d}.pickle"
    with open(tp_obj_fname, "rb") as filein:
        tp: TrialProcessor = pickle.load(filein)

    if not tp.trial_usable:
        print(f"Trial {trial_idx} is not usable!")
        continue

    # Load pre-built echogram
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
    print(f"Print echogram for trial {trial_idx:03d}")

    with plt.ion():
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
        ax[0].set_xticklabels("")
        ax[1].set_xticklabels("")
        plt.savefig(
            save_path / (egram_fname.stem + "_no_marks.png"),
            dpi=150,
            pad_inches=0.1,
            facecolor="w",
        )

        # Plot annotation marks
        marks = np.load(
            # [:-13] to conform with previous filename
            mark_path / (egram_fname.stem[:-13] + "_animal_range.npy")
        )
        ax[0].plot(marks[:, 0], marks[:, 1], "wo", lw=3, markerfacecolor="none")

        plt.savefig(
            save_path / (egram_fname.stem + "_w_marks.png"),
            dpi=150,
            pad_inches=0.1,
            facecolor="w",
        )

        plt.close()
