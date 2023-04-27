import glob

import numpy as np


def get_images(path, encoding=".png"):
    return sorted(glob.glob(path + "*" + encoding))


def load_K_d(data_dir):
    f = open(data_dir + "K.csv", "r")
    K = []
    for line in f:
        for i in range(3):
            K.append(float(line.rstrip().split(",")[i]))

    K = np.array(K).reshape(3, 3)
    f = open(data_dir + "d.csv", "r")
    d = f.readlines()[0].split(",")
    d = np.array([float(val) for val in d])

    return K, d
