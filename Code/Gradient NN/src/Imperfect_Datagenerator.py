import pandas as pd
import sys
import numpy as np
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

# Import our own scripts:

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAWINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../Drawings"))
sys.path.append(DRAWINGS_PATH)
GRADIENT_NN_SRC = os.path.abspath(os.path.join(BASE_DIR, "../../Code/Gradient NN/src"))
sys.path.append(GRADIENT_NN_SRC)

import meshlab_functions as mf
import solidworks_functions as sf
import geometric_functions as gf

############################

def main(neighborhood_size, params, delete_points = 10, shape = "angle_curve", mesh_size = 0.5, noise = False, holes = False):
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path_STL = os.path.join(BASE_DIR, "..", "Data", "Training_data", "perpoint.stl")
    output_path_xyz = os.path.join(BASE_DIR, "..", "Data", "Training_data", "PerPoint.xyz")

    # Create shape
    model = sf.Create_geometry(shape, output_path_STL, params)

    
    # Create pointcloud
    point_dist = mesh_size
    raw_pointcloud = mf.sample_stl_by_point_distance(output_path_STL, output_path_xyz, point_dist)
    pointcloud = np.loadtxt(output_path_xyz, usecols=(0, 1, 2)) # NOTE: Load the saved pointcloud
    
    # Calculate point density
    area = sf.get_surface_area(model, "mm2")
    surface_density = gf.calculate_point_density(area, pointcloud) * neighborhood_size/(neighborhood_size+delete_points)

    # Get features and labels
    all_features = gf.feature_function(pointcloud, k=20, delete_points=delete_points)
    labels = np.hstack((np.full(len(pointcloud), surface_density).reshape(-1,1), np.full(len(pointcloud), mesh_size).reshape(-1,1)))
    
    return all_features, pointcloud, labels

if __name__ == "__main__":
    
    params_ball = {"radius": 50}

    params = {"angle": 150,
              "thicknes": 20,
              "diameter": 10}
    features, _, labels = main(20, params=params_ball, shape="ball", holes=True)

    # np.savetxt("../Data/Training_data/ball_w_holes", pc, fmt='%.3f', delimiter=' ')

    header_label =["Label"]
    header = ["radius", "curvature", "linearity", "planarity", "omnivaraiance", "eigensum", "grad_dist", "average_radius", "points_inside"]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    feature_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/testcloud_feat2.txt"))
    label_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/testcloud_lab2.txt"))
    np.savetxt(feature_PATH, features, delimiter=" ", fmt="%.6f", header=" ".join(header))
    np.savetxt(label_PATH, labels, delimiter=" ", fmt="%.6f", header=" ".join(header_label))