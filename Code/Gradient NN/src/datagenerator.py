import importlib
import pandas as pd
import sys
import numpy as np
import pythoncom
from win32com.client import VARIANT
import pickle
import solidworks_functions as sf
import os
from tqdm import tqdm
from IPython.display import clear_output
importlib.reload(sf)

sys.path.append('../../../Drawings')
import meshlab_functions as mf

sys.path.append('../Code/Gradient NN/src')
import geometric_functions as gf


#####################################################


def create_data(shape = "mix", noise = False, holes = False, ):
    output_path = "./"
    
    
    part_path, params = sf.get_part_and_params(shape)
    
        
    # Create shape
    model = sf.Create_geometry(shape, output_path, params)
    
    
    
    # Create pointcloud
    point_dist = 1
    messy_PC = mf.sample_stl_by_point_distance(output_path, output_path, point_dist)
    
    # Standardize pointcloud
    
    
    
    # Optionals: 
        #  Add noise
    if noise:
        noise_std = 0.2
        noise_mean = 0.1
        
        mf.create_noise(pointcloud, noise_std, noise_mean)
    
        # Create holes
    if holes:
        num_holes = 10
        hole_size = 5
        
        mf.create_mesh_holes(pointcloud, num_holes, hole_size)
    
    
    
    # Calculate point density
    area = sf.get_surface_area(model, "mm2")
    surface_density = gf.calculate_point_density(area, pointcloud)
    
    
    
    # Find edges 
    
        
    # Find neighbors

        
        
        # Calculate curvature
        
        
        
        # Calculate edge number based on neighbors
    
    # Save all data in a np.array
    PC_numpy = 
    
    
    # Save to txt file
    
    return PC_numpy


















