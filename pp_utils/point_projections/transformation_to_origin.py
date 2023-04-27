#!/usr/bin/env python3

import argparse
import glob
import os

import cv2
import numpy as np
from calculate_point_locations import PointIdentifier3D
from scipy.spatial.transform import Rotation

import utils


class TransformationCalculator:
    def __init__(self, extrinsic_calibration_path):
        if extrinsic_calibration_path[-1] != "/":
            extrinsic_calibration_path += "/"
        self.extrinsic_calibration_path = extrinsic_calibration_path
        self.extrinsic_files = glob.glob(extrinsic_calibration_path + "*")

        self.PI3D = PointIdentifier3D()

    def main(self, args, K=None):
        targets_file = open(args.targets_file, "r")
        fname_missed_set = set()
        transformations = {}
        heights = {}
        norm_avg = []
        for i, line in enumerate(targets_file):
            if i == 0:
                continue
            line_split = line.rstrip().split(",")
            fname = line_split[4].split("_")
            if len(fname) < 2:
                continue
            fname = "_".join(fname[1:3])
            # print(fname)
            extrinsic_file = self._calibration_exists(fname)
            if len(extrinsic_file) < 1:
                fname_missed_set.add(fname)
                continue
            extrinsics = self._get_extrinsic_file_vals(extrinsic_file)

            xpx_top = float(line_split[5])
            ypx_top = float(line_split[6])
            depth = abs(float(line_split[7]))
            if line_split[8] == "":  # ???
                continue
            xpx_bottom = float(line_split[8])
            ypx_bottom = float(line_split[9])
            height = extrinsics[2]
            roll = extrinsics[3]
            pitch = extrinsics[4]
            ext_yaw = extrinsics[5]

            R_, _ = cv2.Rodrigues(np.array([roll, pitch, ext_yaw]))
            r = Rotation.from_matrix(R_)
            eulers = r.as_euler("xyz")  # 'xyz' for extrinsic rotation

            self.PI3D.depth = depth
            self.PI3D.height = height

            xw_top, yw_top = self.PI3D.calc_point_refraction(
                xpx_top,
                ypx_top,
                K=K,
                disable_refraction_correction=args.disable_refraction_correction,
            )
            xw_bottom, yw_bottom = self.PI3D.calc_point_refraction(
                xpx_bottom,
                ypx_bottom,
                K=K,
                disable_refraction_correction=args.disable_refraction_correction,
            )
            axis = np.array([xw_top - xw_bottom, yw_top - yw_bottom])

            norm_avg.append(np.linalg.norm(axis))
            axis = np.divide(axis, np.linalg.norm(axis))
            norm = np.array([0, -1])
            yaw = np.arccos(np.dot(axis, norm))

            trans = [xw_bottom, yw_bottom, height + depth]

            G = np.eye(4)
            # R = Rotation.from_euler('xyz', eulers)

            # NOTE: New eulers assumes orthoginal
            # Create matrix which has no roll or pitch
            eulers[0] = 0
            eulers[1] = 0
            eulers[2] = yaw
            R2 = Rotation.from_euler("xyz", eulers)

            G[0:3, 0:3] = R2.as_matrix()
            G[0:3, 3] = trans
            transformations[line_split[4]] = G
            heights[line_split[4]] = height
            # print(line, G)
        print(np.mean(norm_avg))
        print(np.std(norm_avg))
        if args.save:
            self._dump(args.output_dir, transformations, heights)

    def _calibration_exists(self, fname):
        for _file in self.extrinsic_files:
            # extrinsic_fname = '_'.join(_file.split('/')[-1].split('_')[0:2])
            extrinsic_fname = "_".join(os.path.basename(_file).split("_")[:2])
            if fname == extrinsic_fname:
                return _file

        return ""

    def _get_extrinsic_file_vals(self, extrinsic_file):
        # print("extrinsic_file", extrinsic_file)
        vals = open(extrinsic_file, "r").readlines()
        vals = [float(val.rstrip()) for val in vals]
        return vals

    def _dump(self, save_path, transformations, heights):

        if save_path[-1] != "/":
            save_path += "/"
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        out_name = (
            save_path
            + "transformation_to_origin_"
            + self.extrinsic_calibration_path.split("/")[-2]
            + ".txt"
        )
        if os.path.isfile(out_name):
            message = "FILE EXISTS %s type 'y' to override: " % (out_name)
            # if sys.version_info.major == 2:  # these are for py2 compatibility [WJL 20220721]
            #     val = raw_input(message)
            # else:
            val = input(message)
            if val != "y":
                print("Will not override, exiting")
                exit()
        f = open(out_name, "w+")
        for key in transformations.keys():
            G = transformations[key]
            height = heights[key]
            line = self._get_out_str(key, G, height)
            # print(line, G)
            f.write(line + "\n")
        f.close()

    def _get_out_str(self, name, G, height):
        s = name
        s += ","
        G = G.flatten()
        for i, val in enumerate(G):
            if i == 0:
                s += "" + str(val)
            else:
                s += " " + str(val)
        s += "," + str(height)
        return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser("transformation_to_origin")
    parser.add_argument(
        "-t", "--targets_file", default="data/extrinsic_calibration/targets.csv"
    )
    parser.add_argument(
        "-ec",
        "--extrinsic_calibration_path",
        default="data/extrinsic_calibration/hula_flip",
    )
    parser.add_argument(
        "-sid",
        "--saved_intrinsic_dirs",
        default="data/intrinsic_calibration",
        help="Saved K and d directory \
                        (expected names K and d).",
    )
    parser.add_argument(
        "-s", "--save", action="store_true", help="Store extrinsic estimation"
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        default="data/transformation_to_origin",
        help="saved output dir",
    )
    parser.add_argument(
        "-dr",
        "--disable_refraction_correction",
        action="store_true",
        help="Whether or not to disable refraction correction",
    )

    args = parser.parse_args()

    saved_intrinsic_dirs = args.saved_intrinsic_dirs
    if saved_intrinsic_dirs[-1] != "/":
        saved_intrinsic_dirs += "/"

    K, d = utils.load_K_d(saved_intrinsic_dirs)

    TC = TransformationCalculator(args.extrinsic_calibration_path)

    TC.main(args, K=K)
