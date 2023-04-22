from .core import DATA_PATH
from .file_handling import get_trial_info, assemble_target_df

import pandas as pd


class TrialProcessor:

    def __init__(self, df_master, trial_idx, data_path=DATA_PATH, track_label="DTAG"):
        
        self.trial_idx = trial_idx
        self.track_label = track_label
        self.data_paths = data_path
        self.trial_series = df_master.iloc[trial_idx, :]

        # Get all trial flags, params, and paths
        self.flags, self.params, self.paths = get_trial_info(
            df_master, self.data_paths, self.trial_idx
        )

        # Get all trial dataframes
        self.get_all_trial_dfs

        # Print trial info
        self._print_scenario()
        self._print_choice()



    def get_all_trial_dfs(self, track_label="DTAG"):
        """
        Obtain all dataframes for this trial.
        """
        # Get dataframes from stored paths and track label selection
        self.df_track = pd.read_csv(self.paths["track"][self.params["cal_obj_idx"]], index_col=0)
        self.df_dtag = pd.read_csv(self.paths[f"dtag_{track_label}"], index_col=0)
        self.df_hydro_ch0 = pd.read_csv(self.paths[f"hydro_ch0_{track_label}"], index_col=0)
        self.df_hydro_ch1 = pd.read_csv(self.paths[f"hydro_ch1_{track_label}"], index_col=0)

        # Assemble target dataframe
        target_pos_comb = self.trial_series["TARGET_ANGLE"][:2]  # "TC" or "CT"
        self.df_targets = assemble_target_df(
            self.paths["target"], self.params["cal_obj"], target_pos_comb
        )

    def _print_file_paths(self):
        """
        Print file paths
        """        
        print("* track file(s):")
        for ff in self.paths["track"]:
            print(f"  - {ff.name}")
        print(f"* cal object selected: {self.params['cal_obj']}")
        print(f"* dtag file:      %s" % self.paths[f"dtag_{self.track_label}"].name)
        print(f"* hydro ch0 file: %s" % self.paths[f"hydro_ch0_{self.track_label}"].name)
        print(f"* hydro ch1 file: %s" % self.paths[f"hydro_ch1_{self.track_label}"].name)

    def _print_scenario(self):
        print("* scenario: %s" % self.trial_series["TARGET_ANGLE"])

    def _print_choice(self):
        print("* choice: %d" % self.trial_paths["chose_target"])

