import jax.numpy as jnp
from jax import jit
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from pcdiff import knn_graph, estimate_basis, build_grad_div, laplacian, coords_projected, gaussian_weights, weighted_least_squares, batch_dot
import torch
import jax

from pyntcloud import PyntCloud 
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import pandas as pd
import os
import sys
import pdb
from pcdiff import knn_graph



# @jit
# def get_radius(neighborhood_points, index_point):   
#     distances = neighborhood_points - index_point
#     diff = jnp.linalg.norm(distances, axis = -1)
#     radius = jnp.max(diff)
#     return radius



def get_features(index, pointcloud, neighborhood_size):

    # Find the neighbors to index
    pointcloud_center = pointcloud[index]
    diffs = pointcloud - pointcloud_center
    dists = jnp.linalg.norm(diffs, axis=-1)

    # Sort neighbors by distance
    sorted_indices = jnp.argsort(dists)
    neighborhood_indices = sorted_indices[1:neighborhood_size+1]

    neighborhood = pointcloud[neighborhood_indices]
    
    # Get distances to k neighbors
    neighborhood_distances = dists[neighborhood_indices]
    radius = jnp.max(neighborhood_distances)

    features = compute_geometric_properties(neighborhood)

    return features, radius


@jit
def compute_geometric_properties(neighborhood):
    # Compute the covariance matrix using JAX
    cov_matrix = jnp.cov(neighborhood.T)

    # Perform PCA (eigenvalue decomposition) using JAX
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues in descending order
    sorted_indices = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Calculate geometric properties based on sorted eigenvalues
    curvature = eigenvalues[2] / jnp.sum(eigenvalues)
    anisotropy = (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0]
    linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
    planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
    sphericity = eigenvalues[2] / eigenvalues[0]
    variation = eigenvalues[0] / (eigenvalues[0] + eigenvalues[2])
    
    
    # Return the properties as a JAX array for faster computations later
    return jnp.array([curvature, linearity, planarity]), eigenvectors, eigenvalues
















