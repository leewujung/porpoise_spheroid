import numpy as np


class PointIdentifier3D:
    def __init__(
        self,
        height=-1,
        K=None,
        d=None,
        R=None,
        t=None,
        depth=None,
        refraction_index=1.3375,
    ):
        self.height = height
        self.K = K  # Intrinsic matrix
        self.d = d  # distortion matrix
        self.R = R  # 3X3 Rotation matrix
        self.t = t  # 3X1 Translation matrix

        self.depth = depth
        self.refraction_index = refraction_index
        self.refraction_convergence_threshold = 0.001
        self.max_numeric_count = 1000000

    def calc_point_refraction(
        self, x, y, height=None, K=None, disable_refraction_correction=False
    ):
        x_lst = [x]
        y_lst = [y]

        x_out, y_out = self._refract(
            x_lst,
            y_lst,
            height=height,
            K=K,
            disable_refraction_correction=disable_refraction_correction,
        )

        return x_out[0], y_out[0]

    def _record(self, x_points, y_points):
        for i in range(len(x_points)):
            self.world_points.append((x_points[i], y_points[i]))

    def _refract(
        self,
        x_points,
        y_points,
        height=None,
        K=None,
        disable_refraction_correction=False,
    ):
        x_refract = []
        y_refract = []
        for x, y in zip(x_points, y_points):
            homogeneous = [x, y, 1]
            if K is None:
                K = self.K
            K_inv = np.linalg.inv(K)

            if height is None:
                height = self.height
            # print("height", height)
            # print("depth", self.depth)
            # print(height + abs(self.depth))
            world = np.multiply(
                (np.dot(K_inv, np.array(homogeneous))), (height + abs(self.depth))
            )

            if disable_refraction_correction:
                # world = np.multiply((np.dot(K_inv, np.array(homogeneous))),
                #                     (height + abs(self.depth)))
                x = world[0]
                y = world[1]
            else:
                world_d = np.linalg.norm([world[0], world[1]])
                d = self._refractSingleVal(world_d, H=height)

                # Scale x and y based on adjusted diagonal distance (d)
                x = d / world_d * world[0]
                y = d / world_d * world[1]

            x_refract.append(x)
            y_refract.append(y)

        return x_refract, y_refract

    def _refractSingleVal(self, W, H=None):

        # Known variables: H, W, D
        D = self.depth

        # x: horizontal distance between ray intersection point with water surface to target
        x = D / (H + D) * W

        # C: const coming from sin(theta_a) where theta_a is incidence angle in the air
        C = (W - x) / np.sqrt(H**2 + (W - x) ** 2)

        # delta_x: backward correction of horizontal distance
        delta_x = x - np.sqrt(C**2 * D**2 / (self.refraction_index**2 - C**2))

        return W - delta_x
