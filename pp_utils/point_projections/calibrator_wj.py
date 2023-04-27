#!/usr/bin/env python3
import argparse
import glob
import logging

import constants
import cv2
import numpy as np

import utils


class Calibrator:
    def __init__(
        self,
        dimensions_file,
        display_images=None,
        K=None,
        d=None,
        R=None,
        t=None,
        inches=False,
    ):
        # Parse dimesions as: np.array((x1,y1, 0),(x2,y2, 0),..,(xn,yn, 0))
        self.dimensions = self._load_dimensions(dimensions_file, inches=inches)
        self.display_images = display_images
        self.K = K  # Intrinsic matrix
        self.d = d  # distortion matrix
        self.R = R  # 3X3 Rotation matrix
        self.t = t  # 3X1 Translation matrix

        # self.points is nested list of points:
        # np.array([(img1_x1, img1_y1), (img1_x2, img1_y2), ...,
        # (img1_xn, img1_yn)],
        # [(img2_x1, img2_y1)],.., [(imgn_x1, imgn_y1)])
        self.points = []

        # self.images is list of strings pointing to img,
        # only used if display_images is True and images are found
        self.images = []

    def parse_images(self, path):
        self.images = glob.glob(path + "/*")

    def calibrate(self, path):
        if self.display_images is None:
            logging.warn("Display images not specified")
            return False
        self.points, skip = self._parse_points_dlt(path)
        if self.display_images:
            cv2.named_window(constants.img_name, cv2.WINDOW_NORMAL)
        for i, img_points in enumerate(self.points):
            img = None
            if self.display_images and i < len(self.images):
                img = cv2.imread(self.images[i])
                if img is None:
                    logging.warn("Specified image does not exist")

            elif self.display_images:
                logging.warn(
                    "Display images is specified, \
                              but no image found. Not displaying"
                )

        if self.display_images and img is not None:
            for point in img_points:
                cv2.circle(img, (int(point[0]), int(point[1])), (255, 0, 0))
            cv2.imshow(constants.img_name, img)
            cv2.waitKey(constants.wait_key)

        objpoints = [np.float32(self.dimensions) for i in range(len(self.points))]

        ret, K, d, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, np.array(self.points), constants.img_size, None, None
        )

        self.K = K
        self.d = d

        return True

    def extrinsic_calibration(self, args, post_fix="*.txt"):
        if self.K is None or self.d is None:
            logging.warning(
                "Have not yet gotten an intrinsic calibration." " Returning "
            )
            return
        if args.extrinsic_dir[-1] != "/":
            extrinsic_dir = args.extrinsic_dir + "/"
        else:
            extrinsic_dir = args.extrinsic_dir

        extrinsic_points_file = glob.glob(extrinsic_dir + post_fix)

        tvecs = []
        tvecs_rot = []
        rvecs = []
        for extrinsic_file in extrinsic_points_file:
            if args.parse_dlt:
                extrinsic_points, valid = self._parse_points_dlt(extrinsic_file)
            else:
                extrinsic_points = self._parse_points(extrinsic_file)
                valid = None

            if valid is not None:
                objpoints = []

                for i, pnt in enumerate(self.dimensions):
                    if i * 2 in valid:
                        objpoints.append(pnt)
                objpoints = np.float32(objpoints)
            else:
                objpoints = np.float32(self.dimensions)

            if extrinsic_points.shape[0] != objpoints.shape[0]:
                # print(extrinsic_file + " FALIED")
                continue

            ret, rvec, tvec = cv2.solvePnP(objpoints, extrinsic_points, self.K, self.d)

            self.R = cv2.Rodrigues(np.array([[float(rvec[0]), float(rvec[1]), 0]]))[0]

            self.t = tvec
            # print(self.t)
            tvecs.append(tvec)
            rvecs.append(rvec)
            tvecs_rot.append(np.dot(np.linalg.inv(self.R), tvec))

            if args.save:
                if args.name[-1] != "/":
                    save_name = args.name + "/"
                else:
                    save_name = args.name
                extrinsic_file = extrinsic_file.replace(
                    extrinsic_dir, args.outfile + save_name
                )

                name = ".".join(extrinsic_file.split(".")[:-1]) + "EXTRINSIC.txt"
                # Saves as [trans, rvec]
                out = self.t.flatten().tolist() + rvec.flatten().tolist()
                #
                np.savetxt(name, np.array(out), delimiter=",")

    def dump(self, path):
        if path[-1] != "/":
            path += "/"
        if self.K is not None:
            np.savetxt(path + "K.csv", self.K, delimiter=",")
        else:
            logging.warning("Not saving K, not yet calculated")
        if self.d is not None:
            np.savetxt(path + "d.csv", self.d, delimiter=",")
        else:
            logging.warning("Not saving distortion, not yet calculated")
        if self.R is not None:
            np.savetxt(path + "R.csv", self.R, delimiter=",")
        else:
            logging.warning("Not saving R, not yet calculated")
        if self.t is not None:
            np.savetxt(path + "t.csv", self.t, delimiter=",")
        else:
            logging.warning("Not saving translation, not yet calculated")

    def _calc_mean_std(self, arr):
        x = []
        y = []
        z = []
        for val in arr:
            x.append(val[0])
            y.append(val[1])
            z.append(val[2])
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        mean = [np.mean(x), np.mean(y), np.mean(z)]
        std = [np.std(x), np.std(y), np.std(z)]

        return mean, std

    def _load_dimensions(self, dimensions_file, inches=False):
        calibration_dimensions = open(dimensions_file, "r")
        points = calibration_dimensions.readlines()[0].split(";")
        # print(points)
        if inches:
            dimensions = [
                (
                    float(val.split(",")[0]) * 2.54 / 100.0,
                    float(val.split(",")[1]) * 2.54 / 100.0,
                    0,
                )
                for val in points
                if val != "\n"
            ]
        else:
            dimensions = [
                (float(val.split(",")[0]), float(val.split(",")[1]), 0)
                for val in points
                if val != "\n"
            ]

        return np.array(dimensions, dtype=np.float64)

    def _parse_points(self, path):
        f = open(path, "r")
        points = []
        for line in f:
            line = line.rstrip().split(",")
            x = float(line[0])
            y = float(line[1])
            points.append((x, y))
        points = np.array(points, dtype=np.float64)
        return points

    def _parse_points_dlt(self, path):
        f = open(path, "r")
        points = []
        full_valid = []
        for i, line in enumerate(f):
            img_points = []
            valid = []
            if i == 0:
                continue
            line = line.rstrip()

            # Randomly like 50 lines of Nan?
            if np.isnan(float(line.split(",")[0])):
                continue
            for i in range(0, len(line.split(",")), 2):
                x = float(line.split(",")[i])
                y = float(line.split(",")[i + 1])

                if not np.isnan(x) and not np.isnan(y):
                    img_points.append((x, y))
                    valid.append(i)
                    valid.append(i + 1)

            points.append(np.float32(img_points))
            full_valid.append(valid)
        return points[0], full_valid[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("calibrator")
    parser.add_argument("-n", "--name", default="cross", help="Type name")
    parser.add_argument(
        "-d",
        "--dimensions",
        default="data/dimensions/cross.csv",
        help="Path to dimensions file",
    )
    parser.add_argument(
        "--inches", action="store_true", help="If dimensions were inches"
    )
    parser.add_argument(
        "-p",
        "--points",
        default="data/points/input.csv",
        help="Path to points file for intrinsic calibration",
    )
    parser.add_argument(
        "-ed",
        "--extrinsic_dir",
        default="data/calibration_points/cross/",
        help="Path to points file for extrinsic calibration",
    )
    parser.add_argument(
        "-di", "--display_images", action="store_true", help="Display images"
    )
    parser.add_argument(
        "-i",
        "--images",
        default="data/images/",
        help="Images path, only is display images is true",
    )
    parser.add_argument(
        "-s",
        "--save",
        default="True",
        # action='store_true',
        help="Flag to save data",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="data/extrinsic_calibration/",
        help="Path to save output data, if save is true",
    )
    parser.add_argument(
        "-eo",
        "--extrinsic_only",
        default="True",
        # action='store_true',
        help="Only run extrinsic calibration. K and d must be \
                        set",
    )
    parser.add_argument(
        "-sid",
        "--saved_intrinsic_dirs",
        default="data/intrinsic_calibration/",
        help="Saved K/d directory (expected names K and d). \
                        Only needed when extrinsic_only is TRUE",
    )
    parser.add_argument(
        "-dlt",
        "--parse_dlt",
        default="True",
        # action='store_true',
        help="Parse from saved dlt method",
    )
    parser.add_argument(
        "-pf",
        "--post_fix",
        default="*_xypts.csv",
        help="String post fix for loading extrinsic points",
    )

    args = parser.parse_args()

    if args.extrinsic_only:
        saved_intrinsic_dirs = args.saved_intrinsic_dirs
        if saved_intrinsic_dirs[-1] != "/":
            saved_intrinsic_dirs += "/"
        K, d = utils.load_K_d(saved_intrinsic_dirs)
        calibrator = Calibrator(args.dimensions, K=K, d=d, inches=args.inches)
        calibrator.extrinsic_calibration(args, post_fix=args.post_fix)

    else:
        calibrator = Calibrator(
            args.dimensions, display_images=args.display_images, inches=args.inches
        )

        if args.display_images:
            calibrator.parse_images(args.images)

        calibrator.calibrate(args.points)

        calibrator.extrinsic_calibration(args.extrinsic_points)

        if args.save:
            calibrator.dump(args.outfile)
