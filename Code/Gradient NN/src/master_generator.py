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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAWINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../Drawings"))
sys.path.append(DRAWINGS_PATH)
GRADIENT_NN_SRC = os.path.abspath(os.path.join(BASE_DIR, "../../Code/Gradient NN/src"))
sys.path.append(GRADIENT_NN_SRC)


import meshlab_functions as mf
import solidworks_functions as sf
import geometric_functions as gf

import New_idea as Generator

shape_configs = {
    "ball" : [{"radius": 5},
              {"radius": 10},
              {"radius": 15},
              {"radius": 20}],
    "box" : [{"width": 20, "height": 20, "length": 40},
             {"width": 40, "height": 30, "length": 80},
             {"width": 60, "height": 40, "length": 120},
             {"width": 80, "height": 50, "length": 160}],

    "angle_curve" : [{"angle": 120, "thicknes": 10,"diameter": 15},
                     {"angle": 130, "thicknes": 20,"diameter": 20},
                     {"angle": 140, "thicknes": 30,"diameter": 25},
                     {"angle": 150, "thicknes": 40,"diameter": 30}],
    
    "donut" : [{"diameter": 20, "thickmess": 5},
               {"diameter": 30, "thickmess": 10},
               {"diameter": 40, "thickmess": 15},
               {"diameter": 50, "thickmess": 20}],
    
    "donut_center" : [{"dia_ring": 30, "center_length": 60},
                      {"dia_ring": 35, "center_length": 70},
                      {"dia_ring": 40, "center_length": 80},
                      {"dia_ring": 45, "center_length": 90}],
    
    "wedge": [{"height": 100, "width": 20},
              {"height": 120, "width": 30},
              {"height": 140, "width": 40},
              {"height": 160, "width": 50}],
    
    "nut": [{"width": 50, "height": 50, "hole_dia": 20},
            {"width": 60, "height": 60, "hole_dia": 25},
            {"width": 70, "height": 70, "hole_dia": 30},
            {"width": 80, "height": 80, "hole_dia": 35}],
    
    "heart": [{"top_height": 100, "width": 130, "bottom_height": 75, "thick": 50},
              {"top_height": 120, "width": 140, "bottom_height": 85, "thick": 60},
              {"top_height": 140, "width": 150, "bottom_height": 95, "thick": 70},
              {"top_height": 160, "width": 160, "bottom_height": 105, "thick": 100}]
}

mesh_size = [0.5, 1, 2]


def run_batch():
    features_total = []
    labels_total = []
    for shape, param_list in shape_configs.items():
        for param_set in param_list:
            for mesh in mesh_size:
                print(f"🚀 Running: {shape} | params: {param_set} | mesh size: {mesh}")
                try:
                    features, _, labels = Generator.main(
                        params=param_set,
                        shape=shape,
                        mesh_size=mesh
                    )
                    
                    features_list = features.tolist()
                    labels_list = labels.tolist()
                    
                    features_total.extend(features_list) 
                    labels_total.extend(labels_list) 
                    
                    
                except Exception as e:
                    print(f"❌ Error processing {shape} with {param_set}: {e}")
    features_total_np = np.array(features_total) 
    labels_total_np = np.array(labels_total) 



    header_label =["Mesh size", "Num_points"]

    header = ["curvature", "anisotropy", "linearity", "planarity", "sphericity", "variation", "curv_value", "grad"]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    feature_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/Features_NY.txt"))
    label_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/Labels_NY.txt"))
    np.savetxt(feature_PATH, features_total_np, delimiter=" ", fmt="%.6f", header=" ".join(header))
    np.savetxt(label_PATH, labels_total_np, delimiter=" ", fmt="%.6f", header=" ".join(header_label))
        
        
        
        
if __name__ == "__main__":
    # for n_size in [5,10,15,20,25,30,35,40,45,50]:
    run_batch()



