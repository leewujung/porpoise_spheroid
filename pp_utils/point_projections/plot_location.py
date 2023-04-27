#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser("Plot tag locations")
parser.add_argument(
    "-f",
    "--file",
    default="output/world_locs/20190625_s1_t6_GP011466_xypressure_world.csv",
)

parser.add_argument(
    "-if",
    "--image_file",
    default="data/tag_locations/20190625_s1_t6_GP011466_xypressure.csv",
)

parser.add_argument("-t", "--tag", default="RIGHT_EYE")

args = parser.parse_args()

tag_x = args.tag + "_X"
tag_y = args.tag + "_Y"

world = pd.read_csv(args.file)

world_image = pd.read_csv(args.image_file)


fig = plt.figure(figsize=[12, 6])
ax = fig.add_subplot(111, label="1")
ax2 = fig.add_subplot(111, label="2", frame_on=False)

ax.scatter(world[[tag_x]], world[[tag_y]], color="tab:blue")
ax.set_xlabel("Blue: Original", color="tab:blue")
ax.tick_params(axis="x", colors="tab:blue")
ax.tick_params(axis="y", colors="tab:blue")


ax2.scatter(world_image[[tag_x]], world_image[[tag_y]], color="tab:orange")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel("Orange: Rectified world", color="tab:orange")
ax2.xaxis.set_label_position("top")
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis="x", colors="tab:orange")
ax2.tick_params(axis="y", colors="tab:orange")

plt.show()
# plt.plot(world[[tag_x]], world[[tag_y]])
# plt.plot(world_image[[tag_x]], world_image[[tag_y]])
# plt.xlabel("X", fontsize=32)
# plt.ylabel("Y", fontsize=32)
# plt.show()
