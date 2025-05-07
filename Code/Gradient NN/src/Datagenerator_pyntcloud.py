
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

def main(neighborhood_size, params, shape = "angle_curve", mesh_size = 1, noise = False, holes = False):
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    output_path_STL = os.path.join(BASE_DIR, "..", "Data", "Training_data", "perpoint.stl")
    output_path_xyz = os.path.join(BASE_DIR, "..", "Data", "Training_data", "PerPoint.xyz")
    
    
    
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
    radius_list = []
    
   
    
    features_list = []
    
    kdtree = KDTree(pointcloud)
    
    for index in tqdm(range(len(pointcloud)), desc="Processing points"):
    
        neighborhood, raw_neighborhood, indices, distances = gf.find_neighbors(kdtree, index, pointcloud, neighborhood_size)


        points_inside_ball, radius = gf.points_inside_ball(pointcloud, kdtree, index, distances)
        volume_density = gf.calculate_ball_density(radius, points_inside_ball)
        
        radius_list.append(radius)
    
    all_radius = np.array(radius_list)
    
    new_radius = np.average(all_radius)
    
    radius_array = np.full((len(pointcloud), 1), new_radius)
    label = []
    
    
    tree = cKDTree(pointcloud)
    
    
    for index2 in tqdm(range(len(pointcloud)), desc="Second process"):
        
        indices = tree.query_ball_point(pointcloud[index2], new_radius)

        # The number of points inside the sphere
        num_points_inside_sphere = len(indices)
        
        label_row = [num_points_inside_sphere]
        
        label.append(label_row)
    

    
    labels = np.array(label)
    
    feature_array, grad_dist = gf.Get_variables(output_path_xyz, neighborhood_size, save="No")
    
    
    
    all_features = np.hstack((feature_array, grad_dist.reshape(-1,1), radius_array))
    # print(feature_array.shape)
    
    print(f"her {all_features.shape}")
    
    return all_features, pointcloud, labels





if __name__ == "__main__":
    
    params_ball = {"radius": 10}

    params = {"angle": 150,
              "thicknes": 20,
              "diameter": 10}
    main(20, params=params, shape="angle_curve", holes=False)










