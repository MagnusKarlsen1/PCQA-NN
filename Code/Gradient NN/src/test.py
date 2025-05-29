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

import meshlab_functions as mf
import solidworks_functions as sf
import geometric_functions as gf

stl_path = "C:/Users/aagaa/OneDrive - Aarhus universitet/Dokumenter/GitHub/R-D/Drawings/Chain_whee.STL"
cloud_path = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/Gear_w_holes.txt"))
feature_path = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/Gear_w_holes_features.txt"))

mf.sample_stl_by_point_distance(stl_path, cloud_path, 0.5)

xyz = np.loadtxt(cloud_path)[:,0:3]

xyz = mf.create_mesh_holes(xyz, 20, 500)
np.savetxt(cloud_path, xyz, fmt="%.6f", delimiter=" ")

features = gf.feature_function(xyz, k=20, delete_points=0)

header = ["radius", "curvature", "linearity", "planarity", "omnivaraiance", "eigensum", "grad_dist", "AverageRadius", "PointsInside"]


np.savetxt(feature_path, features, delimiter=" ", fmt="%.6f", header=" ".join(header))