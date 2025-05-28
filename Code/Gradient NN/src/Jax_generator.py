
# Imports

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
import importlib
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

importlib.reload(mf)
importlib.reload(gf)
importlib.reload(sf)
############################


def main(neighborhood_size, params, shape = "angle_curve", mesh_size = 1, noise = False, holes = False):
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    output_path_STL = os.path.join(BASE_DIR, "..", "Data", "Training_data", "perpoint.stl")
    output_path_xyz = os.path.join(BASE_DIR, "..", "Data", "Training_data", "PerPoint.xyz")
    
    features_list = []
    
    # part_path, params = sf.get_part_and_params(shape)
    
    
    # Create shape
    model = sf.Create_geometry(shape, output_path_STL, params)
    
    
    # Create pointcloud
    point_dist = mesh_size
    raw_pointcloud = mf.sample_stl_by_point_distance(output_path_STL, output_path_xyz, point_dist)
    
    # Optionals: 
        #  Add noise

    
    pointcloud = np.loadtxt(output_path_xyz, usecols=(0, 1, 2)) # NOTE: Load the saved pointcloud
    
    if noise:
        noise_std = 0.7
        noise_mean = 0.5
        
        pointcloud = mf.create_noise(pointcloud, noise_std, noise_mean)
        print("Creating noise in pointcloud")
        
        
        # Create holes
    if holes:
        num_holes = 1
        hole_size = 1000
        
        pointcloud = mf.create_mesh_holes(pointcloud, num_holes, hole_size)
        print("Creating holes in pointcloud")
        
    # mf.save_neighborhood_to_txt(pointcloud, "./Pre_neighborhood_cloud.txt")    
    
    # Calculate point density
    area = sf.get_surface_area(model, "mm2")
    surface_density = gf.calculate_point_density(area, pointcloud)
    surface_density_array = np.full((len(pointcloud), 1), surface_density)
    
 
    
    radius_list = []
    
    # Calculate gradients
    _, gradients, curvature_ours = gf.get_nn_data(pointcloud, neighborhood_size, neighborhood_size)
    
    gradients = np.mean(gradients, axis=1, keepdims=True)
    
    

    pointcloud_jnp = jnp.array(pointcloud)
    # Step 1: Build KDTree
    tree = KDTree(pointcloud)
    _, neighbor_indices = tree.query(pointcloud, k=neighborhood_size + 1)

    # Step 2: Prepare indices
    neighbor_indices_jax = jnp.array(neighbor_indices)

    point_indices = jnp.arange(len(pointcloud_jnp))

    
    
    features, radii = vmap(lambda idx: gf.get_features(idx, pointcloud_jnp, neighbor_indices_jax))(point_indices)
    
    features = jax.device_get(features)
    radii = jax.device_get(radii)   
    
    
    features_np = np.array(features)
    radii_np = np.array(radii)

 
    
    new_radius = np.mean(radii_np)
    label = []
    

    # Build KDTree
    tree2 = cKDTree(pointcloud)

    # Query all neighbors inside radius at once
    all_neighbors = tree2.query_ball_point(pointcloud, r=new_radius)

    pointsIN = np.array([len(nbh) for nbh in all_neighbors]).reshape(-1, 1)

    
    radius_array = np.full((len(pointcloud), 1), new_radius)

    #     quality = num_points_inside_sphere/neighborhood_size
        
    #     # Quality_score.append(quality)
    #     if quality > 0.76:
    #         Quality_score.append(1)
    #     else:
    #         Quality_score.append(0)
    
    features_array = np.hstack((features_np, gradients, radius_array, pointsIN))    
        
    #     # if num_points_inside_sphere >= neighborhood_size:
    #     #     Quality_score.append(1)
            
    #     # else:
    #     #     Quality_score.append(0)


    
    
    # test = np.array(pointcloud[:, :3])  # shape: (N, 2)
    # test = np.hstack([test, Quality_score.reshape(-1,1)])  # shape: (N, 3)

    # # 5. Save to file
    # np.savetxt("./quality_test.txt", test, fmt="%.6f", delimiter=" ")
    # np.savetxt("./labels.txt", labels, , fmt="%.6f", delimiter=" ")
    
    
    
    
    
    
    # Calculate edge number based on neighbors
    
    # Save all data in a np.array
    # PC_numpy = 
    
    # output_path_features = os.path.join(BASE_DIR, "..", "Data", "Training_data", "Feature_list.txt")
    # Save to txt file
    # mf.save_neighborhood_to_txt(features_array, output_path_features)

    # np.savetxt(output_path_features, features_array, fmt="%.6f", delimiter=" ", header=" ".join(header), comments="")
    
    return features_array, pointcloud, radii_np






if __name__ == "__main__":
    
    params_ball = {"radius": 10}

    params = {"angle": 150,
              "thicknes": 100,
              "diameter": 20}
    features, _, _ = main(20, params=params, mesh_size=0.5, shape="angle_curve", holes=False)
    print(features)
    








