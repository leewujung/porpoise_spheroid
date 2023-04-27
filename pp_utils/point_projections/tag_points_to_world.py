#!/usr/bin/env python3

import argparse
import glob
import os

import cv2
import numpy as np
from calculate_point_locations import PointIdentifier3D

import utils


class Transformer:
    def __init__(self, args, K):
        self.transformation_information = self._load_transformation(
            args.transformation_file
        )
        self.args = args
        self.K = K
        self.d = None
        self.PI3D = PointIdentifier3D()
        self.PI3D.refraction_convergence_threshold = 0.001
        self.PI3D.max_numeric_count = args.max_numeric_count
        self.target_depth = 1
        self.missing = []

        self.depth_init = None
        self.depth_origin_init = None

    def main(self):
        points_files = sorted(glob.glob(self.args.points_files + args.post_fix))
        for point_file in points_files:
            # name_reduced = '_'.join(name.split('/')[-1].split('_')[:3])
            name_reduced = "_".join(os.path.basename(point_file).split("_")[:3])
            if name_reduced not in self.transformation_information:
                self.missing.append(name_reduced)
                print("Skipping ", point_file)
                continue
            print("Transforming ", point_file)
            points_file = open(point_file, "r").readlines()
            line_names = []
            for i, line in enumerate(points_file):
                camera_points = {}
                line_split = line.rstrip().split(",")
                if i == 0:
                    for j in range(0, len(line_split), 2):
                        if j == len(line_split) - 1:
                            # last tag is depth...
                            continue
                        line_name = "_".join(line_split[j].split("_")[:-1])
                        line_names.append(line_name)
                    if self.args.save:
                        name = os.path.splitext(os.path.basename((point_file)))[0] + "_"

                        if self.args.output_path[-1] == "/":
                            output_path = self.args.output_path
                        else:
                            output_path = self.args.output_path + "/"

                        camera_fname, world_fname = self._create_file(
                            name, output_path, line_names
                        )
                    continue
                camera_points[line_names[0]] = (line_split[0], line_split[1])
                camera_points[line_names[1]] = (line_split[2], line_split[3])
                camera_points[line_names[2]] = (line_split[4], line_split[5])
                camera_points[line_names[3]] = (line_split[6], line_split[7])
                camera_points[line_names[4]] = (line_split[8], line_split[9])
                depth = abs(float(line_split[-1]))

                points = self.calculate_world3D(
                    point_file,
                    camera_points,
                    depth,
                    self.args.disable_refraction_correction,
                )  # disable_refraction correction, defaults to False
                if args.save and camera_fname != " " and world_fname != " ":
                    if args.output_path[-1] == "/":
                        output_path = args.output_path
                    else:
                        output_path = args.output_path + "/"

                    self._dump(
                        point_file.split("/")[-1],
                        line_names,
                        camera_fname,
                        world_fname,
                        points,
                        output_path,
                        i,
                    )

    def calculate_world3D(
        self, name, camera_points_px, depth, disable_refraction_correction=True
    ):
        # Name in stored format
        name_reduced = "_".join(os.path.basename(name).split("_")[:3])
        se3, height = self.transformation_information[name_reduced]

        se3 = self._parse_se3(se3)

        points_3d = {}
        for key in camera_points_px:
            point_px = camera_points_px[key]
            if point_px[0] == "" or point_px[1] == "":
                camera = []
                origin = []
                points_3d[key] = [camera, origin]
            else:
                xw, yw = self._calculate_camera_loc_3d(
                    float(point_px[0]),
                    float(point_px[1]),
                    height,
                    depth,
                    disable_refraction_correction,
                )
                camera = [xw, yw, height + depth]

                homogeneous = camera + [1]

                transformed = np.dot(se3, homogeneous)

                xw_origin = transformed[0] / transformed[3]
                yw_origin = transformed[1] / transformed[3]
                depth_origin = -1 * transformed[2]

                if self.depth_init is None:
                    self.depth_init = depth
                    self.depth_origin_init = depth_origin
                #
                origin = [xw_origin, yw_origin, depth_origin]
                points_3d[key] = [camera, origin]  # [camera points, world points]

        return points_3d

    def _calculate_camera_loc_3d(
        self, xpx, ypx, height, depth, disable_refraction_correction=False
    ):
        self.PI3D.height = float(height)
        self.PI3D.depth = depth

        if self.d is not None:
            dat = np.array([[xpx, ypx]], dtype=np.float64).reshape(-1, 1, 2)

            dst = cv2.undistortPoints(dat, self.K, self.d, P=self.K)

            x_un = dst[0][0][0]
            y_un = dst[0][0][1]

            xpx, ypx = x_un, y_un

        xw, yw = self.PI3D.calc_point_refraction(
            xpx,
            ypx,
            K=self.K,
            disable_refraction_correction=disable_refraction_correction,
        )

        return xw, yw

    def _dump(
        self,
        point_fname,
        line_names,
        camera_fname,
        world_fname,
        points_dict,
        output_path,
        idx,
    ):

        camera_file = open(camera_fname, "a+")
        world_file = open(world_fname, "a+")

        full_camera_line = ""
        full_world_line = ""
        for key in line_names:
            if key not in points_dict:
                camera_save_line = ""
                world_save_line = ""
            else:
                camera_points, world_points = points_dict[key]
                camera_save_line = self._get_save_line(camera_points)
                world_save_line = self._get_save_line(world_points)

            full_camera_line += camera_save_line + ","
            full_world_line += world_save_line + ","

        full_camera_line = point_fname + "," + str(idx) + "," + full_camera_line[:-1]
        full_world_line = point_fname + "," + str(idx) + "," + full_world_line[:-1]

        camera_file.write(full_camera_line + "\n")
        world_file.write(full_world_line + "\n")

        camera_file.close()
        world_file.close()

    def _get_save_line(self, points):
        line = ""
        if len(points) == 0:
            return ",,"
        for i, pnt in enumerate(points):
            if i == len(points) - 1:
                line += str(pnt)
            else:
                line += str(pnt) + ","

        return line

    def _load_transformation(self, transformation_file):
        transformation_information = {}
        # print(transformation_file)
        f = open(transformation_file, "r").readlines()
        for i, line in enumerate(f):
            name = line.split(",")[0]
            vals = line.split(",")[1].split(" ")
            vals = [float(val) for val in vals]
            height = float(line.split(",")[2])
            # Names seem to deviate, need to get them in same format
            name_reduced = "_".join(name.split("_")[1:3])
            post = name.split("_")[3]
            post_num = int(post[1:])
            post = post[0] + str(post_num)
            name_reduced += "_" + post
            # print(name_reduced, height)
            transformation_information[name_reduced] = [vals, height]

        return transformation_information

    def _parse_se3(self, se3_flat):
        arr = np.array(se3_flat).reshape(4, 4)

        return np.linalg.inv(arr)

    def _create_file(self, name, output_path, line_names):
        camera_file_name = output_path + name + "camera.csv"
        world_file_name = output_path + name + "world.csv"
        if not self._check_file(camera_file_name):
            print("Not replacing camera file, exiting")
            # exit()
            return " ", " "
        if not self._check_file(world_file_name):
            print("Not replacing world file, exiting")
            return " ", " "
        if not os.path.exists(args.output_path.split("/")[0]):
            os.mkdir(args.output_path.split("/")[0])
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)

        camera_file = open(camera_file_name, "w+")
        world_file = open(world_file_name, "w+")
        locations = ["X", "Y", "Z"]
        first_line = "fname,idx"
        for i, loc in enumerate(line_names):
            for j in range(3):
                val = loc + "_" + locations[j]
                # if i == len(line_names) - 1 and j == 2:
                first_line += "," + val

        camera_file.write(first_line + "\n")
        world_file.write(first_line + "\n")
        camera_file.close()
        world_file.close()

        return camera_file_name, world_file_name

    def _check_file(self, name):
        if os.path.isfile(name):
            message = "FILE EXISTS %s type 'y' to overide: " % (name)
            # if sys.version_info.major == 2:  # these are for py2 compatibility [WJL 20220721]
            #     val = raw_input(message)
            # else:
            val = input(message)
            if val != "y":
                print("Will not override, exiting")
                return False
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Project tag pixels to world points")
    parser.add_argument(
        "-tf",
        "--transformation_file",
        default="data/transformation_to_origin/transformation_to_origin_hula_flip.txt",
    )
    parser.add_argument(
        "-p",
        "--points_files",
        default="/Volumes/SSD_2TB/clutter_infotaxis_analysis/0_from_kavin/xypressure_wj/",
        type=str,
    )
    parser.add_argument(
        "-mnc",
        "--max_numeric_count",
        type=int,  # no longer needed for refraction correction
        default=100000,
    )
    parser.add_argument("-pf", "--post_fix", default="*.csv")
    parser.add_argument(
        "-sid",
        "--saved_intrinsic_dirs",
        default="data/intrinsic_calibration",
        help="Saved K and d directory \
                        (expected names K and d).",
    )
    parser.add_argument(
        "-s",
        "--save",
        default=True,
        # action='store_true',
        help="Save output",
    )
    parser.add_argument(
        "-op", "--output_path", default="data/tag_points_output/", help="Save output"
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

    transformer = Transformer(args, K)
    transformer.d = d
    transformer.main()
