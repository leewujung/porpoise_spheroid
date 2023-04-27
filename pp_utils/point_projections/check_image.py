from argparse import ArgumentParser

import cv2
import numpy as np
from calculate_point_locations import PointIdentifier3D

import utils


class ImagePoints:
    def __init__(self):
        self.pnts = []

    def click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pnts.append((x, y))
        # cv2.setMouseCallback("image", self.click)


def main(args, K):
    PI3D = PointIdentifier3D()
    PI3D.height = 3.4
    PI3D.depth = 1.0
    imgpnts = ImagePoints()
    fname = args.image
    img = cv2.imread(fname)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", imgpnts.click_callback)
    # cv2.waitKey(0)
    while True:
        cv2.imshow("image", img)
        k = cv2.waitKey(1)
        if k == ord("c") or k == ord("\n") or k == ord("\r"):
            break
    pnts = imgpnts.pnts

    xpx_top, ypx_top = pnts[0]
    xpx_bottom, ypx_bottom = pnts[1]

    xw_top, yw_top, _ = PI3D.calc_point_refraction(
        xpx_top, ypx_top, K=K, disable_refraction_correction=False
    )
    xw_bottom, yw_bottom, _ = PI3D.calc_point_refraction(
        xpx_bottom, ypx_bottom, K=K, disable_refraction_correction=False
    )

    axis = np.array([xw_top - xw_bottom, abs(yw_top - yw_bottom)])
    print("axis")
    print(axis)
    print(np.linalg.norm(axis))


if __name__ == "__main__":
    parser = ArgumentParser("Check image refraction")
    parser.add_argument(
        "-sid",
        "--saved_intrinsic_dirs",
        default="data/intrinsic_calibration",
        help="Saved K and d directory \
                        (expected names K and d).",
    )
    parser.add_argument(
        "-i", "--image", default="data/images/20190702_s1_GOPR1481.20234.jpg"
    )
    args = parser.parse_args()

    saved_intrinsic_dirs = args.saved_intrinsic_dirs
    if saved_intrinsic_dirs[-1] != "/":
        saved_intrinsic_dirs += "/"

    K, d = utils.load_K_d(saved_intrinsic_dirs)

    main(args, K)
