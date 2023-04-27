"""
Functions for calibration and transformation of video data.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

intrisinc_path = "./point_projections/data/intrinsic_calibration/"
extrinsic_path = "./point_projections/data/extrinsic_calibration/"


class video_projection:
    def __init__(self, intrisinc_path=intrisinc_path, extrinsic_path=extrinsic_path):
        self.intrisinc_path = intrisinc_path
        self.extrinsic_path = extrinsic_path

        self.K, self.d = self.load_K_d()

    def load_K_d(self):
        f = open(self.intrisinc_path + "/K.csv", "r")
        K = []
        for line in f:
            for i in range(3):
                K.append(float(line.rstrip().split(",")[i]))

        K = np.array(K).reshape(3, 3)
        f = open(self.intrisinc_path + "/d.csv", "r")
        d = f.readlines()[0].split(",")
        d = np.array([float(val) for val in d])
        return K, d

    def get_extrinsic_filename(self, ex_prefix, cal_obj):
        ex_file = list(
            Path(self.extrinsic_path).joinpath(cal_obj).glob(ex_prefix + "*.txt")
        )
        if ex_file:
            ex_file = str(ex_file[0])
            return ex_file
        else:
            print("extrinsic file does not exist")
            return ""

    @staticmethod
    def get_extrinsic_vals(extrinsic_file):
        vals = open(extrinsic_file, "r").readlines()
        vals = [float(val.rstrip()) for val in vals]
        return vals

    def project_to_depth(self, xy, z, ex, nw=1.3375):
        """Project x, y position to depth with refraction correction.

        Parameters
        ----------
        xy : array-like
            x and y pos
        z : float
            depth
        ex : array-like
            extrinsics
        nw : float
            refraction index

        Returns
        -------
        xy_cam : array-like
            undistorted camera xy positions
        xy_surf : array-like
            calibrated xy positions on the water surface
        xy_depth : array-like
            calibrated xy positions at depth z with refraction correction
        """
        # undistort points and project to camera frame
        xy_cam = cv2.undistortPoints(xy, self.K, self.d).squeeze()
        if len(xy_cam.shape) == 1:
            xy_cam = np.expand_dims(xy_cam, axis=0)  # expand dim to work with next step

        # xy point on surface
        xy_world = np.dot(
            np.eye(4), np.hstack((xy_cam, np.ones((xy_cam.shape[0], 2)))).T
        ).T  # no extrinsic rotation

        # scale to surface
        xy_surf = xy_world[:, :3] * np.expand_dims(ex[2] / xy_world[:, 2], axis=1)
        xy_surf = xy_surf[:, :2]

        # compute added distance accounting for refraction
        H = ex[2]  # height of camera above water
        W = np.linalg.norm(
            xy_surf[:, :2], axis=1
        )  # distance from camera center at water surface

        # horizontal distance extended due to refraction
        W_add = np.sqrt((z**2 * W**2) / (nw**2 * H**2 + (nw**2 - 1) * W**2))

        xy_depth = xy_surf * np.expand_dims((W_add + W) / W, axis=1)

        return xy_cam, xy_surf, xy_depth

    def proc_xyz_targets(self, extrinsics, target_series, label_names):

        xy_depth_all = []
        for ln in label_names:
            xy = np.array([target_series[ln + "_X"], target_series[ln + "_Y"]]).T
            if len(xy.shape) == 1:
                xy = np.expand_dims(
                    xy, axis=0
                )  # expand dim to work with function expectation
            xy_cam, xy_surf, xy_depth = self.project_to_depth(
                xy, z=target_series[ln + "_Z"], ex=extrinsics, nw=1.3375
            )
            xy_depth = np.hstack((xy_depth.squeeze(), -1))  # add z (depth)
            xy_depth_all.append(xy_depth)

        # assemble dataframe
        df_xyz_new = target_series.to_frame().T
        df_xyz_new = df_xyz_new.reset_index()
        # in below: target_series.keys()[4:]] = [
        #   'TOP_OBJECT_X', 'TOP_OBJECT_Y', 'TOP_OBJECT_Z',
        #   'BOTTOM_OBJECT_X', 'BOTTOM_OBJECT_Y', 'BOTTOM_OBJECT_Z'
        # ]
        df_xyz_new[target_series.keys()[4:]] = np.array(xy_depth_all).flatten()

        return df_xyz_new

    @staticmethod
    def _rename_NOSTRIL_to_ROSTRUM(df_in):
        return df_in.rename(
            columns={"NOSTRIL_X": "ROSTRUM_X", "NOSTRIL_Y": "ROSTRUM_Y"}
        )

    def proc_xyz_track(self, extrinsics, df_xyz, label_names):
        # process xypressure data: all labeled points
        xy_depth_all = []
        for ln in label_names:
            xy = np.array([df_xyz[ln + "_X"], df_xyz[ln + "_Y"]]).T
            xy_cam, xy_surf, xy_depth = self.project_to_depth(
                xy, z=df_xyz["DTAG_PRESSURE"].values, ex=extrinsics, nw=1.3375
            )
            xy_depth_all.append(xy_depth.T)
        xy_depth_all = (
            np.array(xy_depth_all).reshape(-1, xy_depth.shape[0]).T
        )  # [num of frames x 5 pairs of X-Y]

        # assemble dataframe
        df_xyz_new = pd.DataFrame(data=xy_depth_all, columns=df_xyz.columns[:-1])
        df_xyz_new = self._rename_NOSTRIL_to_ROSTRUM(df_xyz_new)  # rename columns
        df_xyz_new["DTAG_PRESSURE"] = df_xyz["DTAG_PRESSURE"].values

        return df_xyz_new
