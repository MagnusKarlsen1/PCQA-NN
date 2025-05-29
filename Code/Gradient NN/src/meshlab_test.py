import math
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import pythoncom
import argparse
import pickle
import jax 
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
import torch
from scipy.spatial import cKDTree
from scipy.spatial import KDTree

import random
import numpy.linalg as LA
import os
from tqdm import tqdm
from IPython.display import clear_output

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAWINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../Drawings"))
sys.path.append(DRAWINGS_PATH)
GRADIENT_NN_SRC = os.path.abspath(os.path.join(BASE_DIR, "../../Code/Gradient NN/src"))
sys.path.append(GRADIENT_NN_SRC)

store_path_stl = os.path.join(BASE_DIR, "../../../Drawings", "temp_cloud.stl")
store_path = os.path.join(BASE_DIR, "../../../Drawings", "temp_cloud.xyz")

sd_path = os.path.join(BASE_DIR, "../../../Drawings", "MeshTest_SufaceDensity.txt")
std_path = os.path.join(BASE_DIR, "../../../Drawings", "MeshTest_std.txt")


import meshlab_functions as mf
import solidworks_functions as sf
import geometric_functions as gf

import Datagenerator_pyntcloud_copy as Generator
sd = np.array([2.69120073, 1.83468036, 1.30338184, 1.01694597, 0.80130886, 0.68447244,
      0.56438817, 0.4512232, 0.39104326, 0.33837338, 0.30846906, 0.2459274,
      0.23189741, 0.20983105, 0.17414554, 0.16162432])
meshsize = np.arange(0.5, 2.01, 0.1)
singlePC_std = []

point_data = []
for n in range(160, 170, 1):
    area = n/sd
    for i in range(0, 16):
        print(area)
        shape_configs = {"ball" : {"radius": math.sqrt(area[i]/(4*math.pi))}, 
                    "box" : {"width": math.sqrt(area[i]/6), "height": math.sqrt(area[i]/6), "length": math.sqrt(area[i]/6)}, 
                    "donut" : {"diameter": 2*math.sqrt(area[i])/math.pi, "thickmess": math.sqrt(area[i])/math.pi/4}
                    }
        for shape, params in shape_configs.items():
            print(n)
            print(meshsize[i])
            model = sf.Create_geometry(shape, store_path_stl, params)
            mf.sample_stl_by_point_distance(store_path_stl, store_path, meshsize[i])
            xyz = np.loadtxt(store_path)[:,0:3]
            if shape == "ball":
                thresh_x = params["radius"]
                thresh_y = params["radius"]
                thresh_z = params["radius"]

            if shape == "box":
                thresh_x = math.sqrt(area[i]/6)/2
                thresh_y = math.sqrt(area[i]/6)/2
                thresh_z = math.sqrt(area[i]/6)/2

            if shape == "donut":
                thresh_x = params["diameter"] + params["thickmess"]/2
                thresh_y = params["thickmess"]/2
                thresh_z = params["diameter"] + params["thickmess"]/2

            count = np.array([np.sum((xyz[:, 0] > thresh_x) & (xyz[:, 1] > thresh_y) & (xyz[:, 2] > thresh_z)),
                              np.sum((xyz[:, 0] > thresh_x) & (xyz[:, 1] > thresh_y) & (xyz[:, 2] < thresh_z)),
                              np.sum((xyz[:, 0] > thresh_x) & (xyz[:, 1] < thresh_y) & (xyz[:, 2] > thresh_z)),
                              np.sum((xyz[:, 0] > thresh_x) & (xyz[:, 1] < thresh_y) & (xyz[:, 2] < thresh_z)),
                              np.sum((xyz[:, 0] < thresh_x) & (xyz[:, 1] > thresh_y) & (xyz[:, 2] > thresh_z)),
                              np.sum((xyz[:, 0] < thresh_x) & (xyz[:, 1] > thresh_y) & (xyz[:, 2] < thresh_z)),
                              np.sum((xyz[:, 0] < thresh_x) & (xyz[:, 1] < thresh_y) & (xyz[:, 2] > thresh_z)),
                              np.sum((xyz[:, 0] < thresh_x) & (xyz[:, 1] < thresh_y) & (xyz[:, 2] < thresh_z))])
            surface_density = 8*count/area[i]
            singlePC_std.append(np.std(surface_density))
            #np.mean(surface_density)
            print(count)
            point_data.append(np.hstack((surface_density, count, np.full(len(count), meshsize[i]))))

point_data = np.vstack(point_data)
sd_all = point_data[:,0:8].flatten()
meshsize_all = point_data[:,16:].flatten()
print(point_data)

unique_mesh = np.unique(meshsize_all)
means_ball = []
stds_ball = []

means_box = []
stds_box = []

means_donut = []
stds_donut = []

sd_ball = sd_all[0::3]
meshsize_ball = meshsize_all[0::3]

sd_box = sd_all[1::3]
meshsize_box = meshsize_all[1::3]

sd_donut = sd_all[2::3]
meshsize_donut = meshsize_all[2::3]

for val in unique_mesh:
    ball_vals = sd_ball[meshsize_ball == val]
    means_ball.append(np.mean(ball_vals))
    stds_ball.append(np.std(ball_vals))

    box_vals = sd_box[meshsize_box == val]
    means_box.append(np.mean(box_vals))
    stds_box.append(np.std(box_vals))

    donut_vals = sd_donut[meshsize_donut == val]
    means_donut.append(np.mean(donut_vals))
    stds_donut.append(np.std(donut_vals))

#print("Mean standard deviation for a single pointcloud:", np.mean(singlePC_std))
#print("Mean standard deviation for all meshsizes:", np.mean(stds))

#plt.scatter(meshsize_all, sd_all, marker='o', color='cornflowerblue', label='All surface densities')

# plt.errorbar(meshsize, means_ball, yerr=stds_ball, fmt='o', color='darkolivegreen', capsize=5, label='Mean ± Std - ball')
# plt.plot(meshsize, means_ball, color='darkolivegreen', linestyle='--', linewidth=1)

# plt.errorbar(meshsize, means_box, yerr=stds_box, fmt='o', color='darkmagenta', capsize=5, label='Mean ± Std - box')
# plt.plot(meshsize, means_box, color='darkmagenta', linestyle='--', linewidth=1)

# plt.errorbar(meshsize, means_donut, yerr=stds_donut, fmt='o', color='peru', capsize=5, label='Mean ± Std - donut')
# plt.plot(meshsize, means_donut, color='peru', linestyle='--', linewidth=1)

# plt.xlabel("Mesh Size")
# plt.ylabel("Surface density")
# plt.title("Surface density variation for meshsize")
# plt.legend()
# plt.grid(True)
# plt.show()
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)  # 1 row, 3 columns

# Ball
axes[0].scatter(meshsize_all[0::3], sd_all[0::3], marker='o', color='cornflowerblue', label='All surface densities')
axes[0].errorbar(meshsize, means_ball, yerr=stds_ball, fmt='o', color='darkolivegreen', capsize=5, label='Mean ± Std')
axes[0].plot(meshsize, means_ball, color='darkolivegreen', linestyle='--', linewidth=1)
axes[0].set_title('Ball')
axes[0].set_xlabel('Mesh Size')
axes[0].set_ylabel('Surface Density')
axes[0].grid(True)
axes[0].legend()

# Box
axes[1].scatter(meshsize_all[1::3], sd_all[1::3], marker='o', color='cornflowerblue', label='All surface densities')
axes[1].errorbar(meshsize, means_box, yerr=stds_box, fmt='o', color='darkmagenta', capsize=5, label='Mean ± Std')
axes[1].plot(meshsize, means_box, color='darkmagenta', linestyle='--', linewidth=1)
axes[1].set_title('Box')
axes[1].set_xlabel('Mesh Size')
axes[1].grid(True)
axes[1].legend()

# Donut
axes[2].scatter(meshsize_all[2::3], sd_all[2::3], marker='o', color='cornflowerblue', label='All surface densities')
axes[2].errorbar(meshsize, means_donut, yerr=stds_donut, fmt='o', color='peru', capsize=5, label='Mean ± Std')
axes[2].plot(meshsize, means_donut, color='peru', linestyle='--', linewidth=1)
axes[2].set_title('Donut')
axes[2].set_xlabel('Mesh Size')
axes[2].grid(True)
axes[2].legend()

fig.suptitle('Surface Density Variation for Mesh Size', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Adjust space for the suptitle
plt.savefig("sd_vs_mesh.png", dpi=2000, bbox_inches='tight')
plt.show()

header_sd = ["Meshsize", "Surface Density"]
header_std =["Label"]

np.savetxt(sd_path, np.hstack((meshsize_all.reshape(-1,1), sd_all.reshape(-1,1))), delimiter=" ", fmt="%.6f", header=" ".join(header_sd))
np.savetxt(std_path, [np.mean(singlePC_std), np.mean(stds)], delimiter=" ", fmt="%.6f", header=" ".join(header_std))