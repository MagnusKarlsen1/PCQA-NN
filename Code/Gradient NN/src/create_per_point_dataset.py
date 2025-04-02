
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


def main(neighborhood_size, shape = "ball", noise = False, holes = False):
    output_path = "../Data/Training_data"
    features_list = []
    
    part_path, params = sf.get_part_and_params(shape)
    
    
    # Create shape
    model = sf.Create_geometry(shape, output_path, params)
    
    
    # Create pointcloud
    point_dist = 1
    raw_pointcloud = mf.sample_stl_by_point_distance(output_path, output_path, point_dist)
    
    # Optionals: 
        #  Add noise
    if noise:
        noise_std = 0.2
        noise_mean = 0.1
        
        raw_pointcloud = mf.create_noise(raw_pointcloud, noise_std, noise_mean)
    
        # Create holes
    if holes:
        num_holes = 10
        hole_size = 5
        
        raw_pointcloud = mf.create_mesh_holes(raw_pointcloud, num_holes, hole_size)
    
    pointcloud = np.loadtxt(output_path, usecols=(0, 1, 2)) # NOTE: Load the saved pointcloud
    
 
    # Calculate point density
    area = sf.get_surface_area(model, "mm2")
    surface_density = gf.calculate_point_density(area, pointcloud)
    
    
    # Calculate gradients
    _, gradients, curvature = gf.get_nn_data(pointcloud, 5, 5)
    
    # Find edges 
    
        
    # Find neighbors
    
    kdtree = KDTree(pointcloud)
    
    for index in tqdm(range(len(pointcloud)), desc="Processing points"):
    
        neighborhood, raw_neighborhood, indices = gf.find_neighbors(kdtree, index, pointcloud, neighborhood_size)

        
        # Calculate curvature
        features_jax = gf.compute_geometric_properties(neighborhood)
        features = np.array(features_jax)  # Convert to regular NumPy array
        features = features.tolist()      # Now it's a list of floats
        
        
        # Get gradient vector(s) for the current point
        grad_vectors = gradients[index]           # shape: (k, 2)
        grad_flat = grad_vectors.flatten().tolist()
        
        
        # Get curvature value for the current point
        curv_value = curvature[index].item() if isinstance(curvature[index], np.generic) else curvature[index]

        # Combine everything
        feature_row = features + grad_flat + [curv_value] + [surface_density]
        
        features_list.append(feature_row)
        
    features_array = np.array(features_list)
        
        
        # Calculate edge number based on neighbors
    
    # Save all data in a np.array
    # PC_numpy = 
    
    
    # Save to txt file
    mf.save_neighborhood_to_txt(features_array, "TESTERERE.txt")
    



if __name__ == "__main__":
    main(10)










