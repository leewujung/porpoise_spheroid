# Porpoise spheroid discrimination

This repository contains code for processing data from an behavioral echolocation target discrimination experiment, in which a harbor porpoise was trained to select a sphere against spheroids in a two-alterative forced-choice (2AFC) paradigm.


## Setup

The data processing steps requires both raw experimental data and processed data. They are set up using symbolic link to the actual data folders on external harddrives.

To do this, use:
```
$ ln -s PATH_TO_RAW_DATA THIS_REPO/data_raw
$ ln -s PATH_TO_PROCESSED_DATA THIS_REPO/data_processed
```
This will create 2 symlinks within the repo folder called `data_raw` and `data_processed` that contain the raw and processed data, respectively. The data pre-processing notebooks in the `notebooks_procs` folder use these data to create more processed data in `data_processed`.

The `notebooks_procs/run_all_proc_steps.ipynb` notebook uses [papermill](https://papermill.readthedocs.io/en/latest/index.html) to run through all the data pre-processing notebooks in sequence. If you run into trouble that papermill does not recognize your conda environment, check [this page](https://papermill.readthedocs.io/en/latest/troubleshooting.html) and set up a jupyter kernel using:

```shell
$ python -m ipykernel install --user --name porpoise_spheroid
```