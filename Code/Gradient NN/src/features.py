import jax.numpy as jnp
from jax import jit
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from pcdiff import knn_graph, estimate_basis, build_grad_div, laplacian, coords_projected, gaussian_weights, weighted_least_squares, batch_dot
import torch
import jax
from pyntcloud import PyntCloud 
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import pandas as pd
import os
import sys
import pdb



def get_eigenfeatures(neighborhood, normalize = False):
    cov_matrix = jnp.cov(neighborhood.T)
    # Perform PCA (eigenvalue decomposition) using JAX
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues in descending order
    sorted_indices = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    if normalize:
        eigen_sum = jnp.sum(eigenvalues)
        eigenvalues = eigenvalues / eigen_sum
    
    
    # Calculate geometric properties based on sorted eigenvalues
    curvature = eigenvalues[2] if normalize else eigenvalues[2] / (jnp.sum(eigenvalues) + 1e-10)
    linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
    planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]    
    scattering = eigenvalues[2]/eigenvalues[0]
    anisotropy = (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0]
    omnivariance = jnp.cbrt(eigenvalues[0]*eigenvalues[1]*eigenvalues[2])
    sphericity = eigenvalues[2] / eigenvalues[0]
    variation = eigenvalues[0] / (eigenvalues[0] + eigenvalues[2])
    eigensum = eigenvalues[0]+eigenvalues[1]+eigenvalues[2]
    eigentropy = -jnp.sum(eigenvalues * jnp.log(eigenvalues + 1e-10))
    
    
    
    
    return eigenfeatures















