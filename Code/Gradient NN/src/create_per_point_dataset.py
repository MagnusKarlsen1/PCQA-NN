
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


def main(neighborhood_size, params, shape = "angle_curve", noise = False, holes = False):
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    output_path_STL = os.path.join(BASE_DIR, "..", "Data", "Training_data", "perpoint.stl")
    output_path_xyz = os.path.join(BASE_DIR, "..", "Data", "Training_data", "PerPoint.xyz")
    
    features_list = []
    eigenvector_list = []
    eigenvalues_list = []
    # part_path, params = sf.get_part_and_params(shape)
    
    
    # Create shape
    model = sf.Create_geometry(shape, output_path_STL, params)
    
    
    # Create pointcloud
    point_dist = 0.2
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
        num_holes = 10
        hole_size = 20
        
        pointcloud = mf.create_mesh_holes(pointcloud, num_holes, hole_size)
        print("Creating holes in pointcloud")
        
    mf.save_neighborhood_to_txt(pointcloud, "./Pre_neighborhood_cloud.txt")    
    
    # Calculate point density
    area = sf.get_surface_area(model, "mm2")
    surface_density = gf.calculate_point_density(area, pointcloud)
    
    
    # Calculate gradients
    _, gradients, curvature_ours = gf.get_nn_data(pointcloud, neighborhood_size, 2)
    
    # Find edges 
    
        
    # Find neighbors
    
    kdtree = KDTree(pointcloud)
    
    for index in tqdm(range(len(pointcloud)), desc="Processing points"):
    
        neighborhood, raw_neighborhood, indices, distances = gf.find_neighbors(kdtree, index, pointcloud, neighborhood_size)


        points_inside_ball, radius = gf.points_inside_ball(pointcloud, kdtree, index, distances)
        volume_density = gf.calculate_ball_density(radius, points_inside_ball)
        
        # Calculate curvature
        features_jax, eigenvectors, eigenvalues = gf.compute_geometric_properties(neighborhood)
        features = np.array(features_jax)  # Convert to regular NumPy array
        features = features.tolist()      # Now it's a list of floats 
        
        
        eigenvector_list.append(eigenvectors.tolist())
        eigenvalues_list.append(eigenvalues.tolist()) 
        
        # Get gradient vector(s) for the current point
        grad_vectors = gradients[index]           # shape: (k, 2)
        grad_flat = grad_vectors.flatten().tolist()

        # Get curvature value for the current point
        curv_value = curvature_ours[index]

        # Combine everything
        feature_row = features + grad_flat + [curv_value] + [surface_density] + [volume_density]
        
        features_list.append(feature_row)
    
    eigenvectors_list = np.array(eigenvector_list)
    eigenvalues_list = np.array(eigenvalues_list)
    
    # np.savetxt("./Eigenvectors.txt", eigenvectors_list, fmt="%.6f", delimiter=" ")
    # np.savetxt("./Eigenvalues.txt", eigenvalues_list, fmt="%.6f", delimiter=" ")
    
    ############# TEST med curvature ##############
    
    # 1. Take the first two columns of the pointcloud
    test = np.array(pointcloud[:, :3])  # shape: (N, 2)
    features_array_test = np.array(features_list)

    # 2. Get the curvature column (assumes 1D array of shape (N,))
    curvature_test = features_array_test[:,0]     # or whatever column index represents curvature

    # 3. Normalize curvature
    # curvature_tester = gf.min_max_scale(curvature_test)
    # curvature_tester = curvature_tester.reshape(-1, 1)  # reshape to (N, 1) for hstack

    # 4. Stack coordinates and normalized curvature
    test = np.hstack([test, curvature_test.reshape(-1,1)])  # shape: (N, 3)

    # 5. Save to file
    np.savetxt("./denHER.txt", test, fmt="%.6f", delimiter=" ")
    
    ###############################################    
    
    
    features_array = np.array(features_list)
    header = ["Eigenvalue_curvature", "anisotropy", "linearity", "planarity", "sphericity", "variation"]
    header += ["grad_x", "grad_y", "mean_curvature", "surface_density", "Volume_Density"]
        
    # Calculate edge number based on neighbors
    
    # Save all data in a np.array
    # PC_numpy = 
    
    output_path_features = os.path.join(BASE_DIR, "..", "Data", "Training_data", "Feature_list.txt")
    # Save to txt file
    mf.save_neighborhood_to_txt(features_array, output_path_features)
    np.savetxt(output_path_features, features_array, fmt="%.6f", delimiter=" ", header=" ".join(header), comments="")
    return features_array, pointcloud


if __name__ == "__main__":
    
    params_ball = {"radius": 10}

    params = {"angle": 150,
              "thicknes": 20,
              "diameter": 10}
    main(40, params=params, shape="angle_curve")










