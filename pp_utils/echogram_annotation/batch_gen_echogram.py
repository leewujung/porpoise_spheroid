"""
Batch build echograms from all trials.
"""

from pathlib import Path

import numpy as np

from pp_utils.file_handling import df_main_loader
from pp_utils.echogram import Echogram
from pp_utils.trial_processor import TrialProcessor

import pickle


# Set paths and load main info df
analysis_path = Path("./data_processed")
raw_path = Path("./data_raw")

# Trial processor object
tp_path = analysis_path / "data_summary/trial_processor_object"

# Output path to save generated echogram
output_path = analysis_path / "echogram/batch_gen"
if not output_path.exists():
    output_path.mkdir()

# Path to dtag filterd bpf
dtag_bpf_path = analysis_path / "dtag/dtag_split_wav_filtered_bpf"

# Load main dataframe
df_main = df_main_loader(
    folder=analysis_path / "all_info_csv", filename="main_info_append_09.csv"
)

# Loop through all trials to build echogram and save
for trial_idx in df_main.index:

    # Load TrialProcessor object
    tp_obj_fname = tp_path / f"trial_{trial_idx:03d}.pickle"
    with open(tp_obj_fname, "rb") as filein:
        tp: TrialProcessor = pickle.load(filein)

    if not tp.trial_usable:
        print(f"Trial {trial_idx} is not usable!")
        continue

    # Filtered bpf dtag data
    dtag_bpf_wav_path = dtag_bpf_path / (
        f"t{trial_idx:03d}"
        f"_{tp.trial_series['DATE']}"
        f"_s{tp.trial_series['SESSION']}"
        f"_t{tp.trial_series['TRIAL']:02d}"
        "_dtag_filtered_bpf.wav"
    )

    print(f"Building echogram for trial {trial_idx:03d}")
    egram = Echogram(
        df_dtag=tp.df_dtag,
        dtag_wav_path=dtag_bpf_wav_path,
    )
    egram.build_echogram(time_step=50e-3)

    np.savez(
        output_path / (f"{dtag_bpf_wav_path.stem}_echogram.npz"),
        echogram=egram.echo_mtx_fine,
        animal_range=egram.range_vec,
        time_corrected=egram.time_fine,
    )
