
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
from sklearn.neighbors import NearestNeighbors
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

def main(neighborhood_size, params, shape = "angle_curve", mesh_size = 0.5, noise = False, holes = False):
    
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
   # After creating holes
    if holes:
        num_holes = 2
        hole_size = 500
        pointcloud = mf.create_mesh_holes(pointcloud, num_holes, hole_size)
        print("Creating holes in pointcloud")

        # Find which points were removed by hole creation
        pc_full = np.loadtxt(output_path_xyz, usecols=(0, 1, 2))  # Load the original, full cloud
        full_points_set = set(map(tuple, pc_full))
        holed_points_set = set(map(tuple, pointcloud))
        missing_points = np.array(list(full_points_set - holed_points_set))

        # Identify edge points near missing points
        if len(missing_points) > 0:
            
            nn = NearestNeighbors(n_neighbors=10, radius=2.0)
            nn.fit(pointcloud)
            neighbor_indices = nn.radius_neighbors(missing_points, radius=2.0, return_distance=False)
            edge_indices = np.unique(np.concatenate(neighbor_indices))
            # Save the final point cloud (with holes) to a new file
    output_path_xyz_holed = os.path.join(BASE_DIR, "..", "Data", "Training_data", "PerPoint_w_holes.xyz")
    np.savetxt(output_path_xyz_holed, pointcloud, fmt="%.6f")
    # mf.save_neighborhood_to_txt(pointcloud, "./Pre_neighborhood_cloud.txt")    
    
    # Calculate point density
    area = sf.get_surface_area(model, "mm2")
    surface_density = gf.calculate_point_density(area, pointcloud)
    # radius_list = []
    
   
    
    # features_list = []
    feature_array, grad_dist, radius = gf.Get_variables(output_path_xyz_holed, neighborhood_size, save="No")
    #feature_array, grad_dist, radius = gf.Get_variables(output_path_xyz, neighborhood_size, save="No")
    
    average_radius = np.mean(radius)
    average_radius_array = np.full((len(pointcloud), 1), average_radius)
    
    tree2 = cKDTree(pointcloud)

    # Query all neighbors inside radius at once
    all_neighbors = tree2.query_ball_point(pointcloud, r=average_radius)

    pointsIN = np.array([len(nbh) for nbh in all_neighbors]).reshape(-1, 1)
    
    
    
    all_features = np.hstack((feature_array, grad_dist.reshape(-1,1)))
    # kdtree = KDTree(pointcloud)
    
    # for index in tqdm(range(len(pointcloud)), desc="Processing points"):
    
    #     neighborhood, raw_neighborhood, indices, distances = gf.find_neighbors(kdtree, index, pointcloud, neighborhood_size)


    #     points_inside_ball, radius = gf.points_inside_ball(pointcloud, kdtree, index, distances)
    #     volume_density = gf.calculate_ball_density(radius, points_inside_ball)
        
    #     radius_list.append(radius)
    
    # all_radius = np.array(radius_list)
    
    # new_radius = np.average(all_radius)
    
    # radius_array = np.full((len(pointcloud), 1), new_radius)
    labels = np.hstack((np.full(len(pointcloud), surface_density).reshape(-1,1), np.full(len(pointcloud), mesh_size).reshape(-1,1)))
    # label = []
    # new_radius = np.average(radius)
    # Reduce label for edge points
    if holes and len(missing_points) > 0:
        labels[edge_indices, 0] *= 0.5  # Halve the surface_density for edge points
    # tree = cKDTree(pointcloud)
    
    
    # for index2 in tqdm(range(len(pointcloud)), desc="Second process"):
        
    #     indices = tree.query_ball_point(pointcloud[index2], new_radius)

    #     # The number of points inside the sphere
    #     num_points_inside_sphere = len(indices)
        
    #     label_row = [num_points_inside_sphere]
        
    #     label.append(label_row)
    

    
    # labels = np.array(label)
    

    # print(feature_array.shape)
    
    
    return all_features, pointcloud, labels





if __name__ == "__main__":
    
    params_ball = {"radius": 50}

    params = {"angle": 150,
              "thicknes": 20,
              "diameter": 10}
    features, _, labels = main(20, params=params_ball, shape="ball", holes=True)

    # np.savetxt("../Data/Training_data/ball_w_holes", pc, fmt='%.3f', delimiter=' ')

    header_label =["Label"]
    header = ["edge_mean", "plane_mean", "curvature", "linearity", "planarity", "omnivaraiance", "eigensum", "grad_dist", "radius"]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    feature_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/holetest_feat.txt"))
    label_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/holetest_label.txt"))
    np.savetxt(feature_PATH, features, delimiter=" ", fmt="%.6f", header=" ".join(header))
    np.savetxt(label_PATH, labels, delimiter=" ", fmt="%.6f", header=" ".join(header_label))








