import numpy as np
import pandas as pd
import sys
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRADIENT_NN_SRC = os.path.abspath(os.path.join(BASE_DIR, "../../Code/Gradient NN/src"))
sys.path.append(GRADIENT_NN_SRC)
output_path_point = os.path.join(BASE_DIR, "..", "Leihui Code", "dataset", "scanning_repository", "bunny.xyz")




import jax_functions as jf



# def pointcloud_evaluator(pointcloud, minimum_point_density):
    
    
#     # Load XGB1 as local_estimator.
#     # Load XGB2 as overall_estimator.
    
    
    
    
    
#     # Calculate features for the pointcloud.
#         # Get average radius.
    
    
    
    
    
#     # Use average radius to get the actual points inside ball.
    
#     # Get a (n, 1) array prediction of local quality from local_estimator.
    
#     # Divide the estimated and predicted number of points inside circle and save as Local_quality array of shape (n,1).
    
#     # Output a (n,4) array [x, y, z, Local_quality]
#         # Save it as txt
    
    
    
    
    
#     return

pointcloud = np.loadtxt(output_path_point)
pointloud = jnp.asarray(pointcloud)

print(type(pointcloud))

neighborhood_size = 20

print("From here :)")

# Vectorize over all indices
point_indices = jnp.arange(len(pointcloud))

features, radii = jax.vmap(lambda idx: jf.get_features(idx, pointcloud, neighborhood_size))(point_indices)




print(f"Radius: {radii.shape}, Features: {features.shape}")




















