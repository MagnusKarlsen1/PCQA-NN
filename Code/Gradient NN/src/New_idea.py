
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


def main(params, shape="angle_curve", mesh_size=1):
    import os
    import numpy as np
    from tqdm import tqdm
    from scipy.spatial import cKDTree
    import meshlab_functions as mf
    import solidworks_functions as sf
    import geometric_functions as gf

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    output_path_STL = os.path.join(BASE_DIR, "..", "Data", "Training_data", "perpoint.stl")
    output_path_xyz = os.path.join(BASE_DIR, "..", "Data", "Training_data", "PerPoint.xyz")

    # Step 1: Create geometry and sample point cloud
    model = sf.Create_geometry(shape, output_path_STL, params)
    point_dist = mesh_size
    raw_pointcloud = mf.sample_stl_by_point_distance(output_path_STL, output_path_xyz, point_dist)
    pointcloud = np.loadtxt(output_path_xyz, usecols=(0, 1, 2))

    # Step 2: Get gradients and curvature
    _, gradients, curvature_ours = gf.get_nn_data(pointcloud, 5, 5)
    gradients = np.mean(gradients, axis=1, keepdims=True)

    # Step 3: Build KD-tree and estimate radius
    tree = cKDTree(pointcloud)
    distances, _ = tree.query(pointcloud, k=2)  # 2nd nearest neighbor to avoid zero
    radius = np.mean(distances[:, 1]) * 4.5
    print(f"Radius: {radius:.4f}")

    # Step 4: Precompute all neighborhoods
    all_neighbors = tree.query_ball_point(pointcloud, r=radius)


    # Step 5: Compute features
    features_list = []
    num_points_label = []
    for index, indices in enumerate(tqdm(all_neighbors, desc="Computing features")):
        neighborhood = pointcloud[indices]
        num_points = len(indices)
        features_jax, _, _ = gf.compute_geometric_properties(neighborhood)
        features = np.array(features_jax).tolist()
        grad_flat = gradients[index].flatten().tolist()
        curv_value = curvature_ours[index]
        feature_row = features + [curv_value] + grad_flat
        num_points_label.append(num_points)
        features_list.append(feature_row)

    # Step 6: Format output
    features_array = np.array(features_list)
    num_points_label = np.array(num_points_label).reshape(-1,1)
    
    labels = np.full((len(pointcloud), 1), mesh_size)
    labels = np.hstack((labels, num_points_label))

    print(labels)
    print(labels.shape)
    return features_array, pointcloud, labels








if __name__ == "__main__":
    
    params_ball = {"radius": 10}

    params = {"angle": 150,
              "thicknes": 100,
              "diameter": 20}
    main(params=params, shape="angle_curve")

    








