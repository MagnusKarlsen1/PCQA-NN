import jax.numpy as jnp
from jax import jit
import numpy as np
from sklearn.decomposition import PCA

def standardize_pointcloud(pointcloud):
    pca = PCA(n_components=3)
    aligned_pointcloud = pca.fit_transform(pointcloud)  # Centering + rotation
    return aligned_pointcloud


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
    return np.array([curvature, anisotropy, linearity, planarity, sphericity, variation])


@jit
def preprocess(points):
    mean_p = points.mean(axis=0)
    min_p, max_p = jnp.min(points, axis=0), jnp.max(points, axis=0)
    bbdiag = jnp.linalg.norm(max_p - min_p, ord=2) # Bounding box diagonal L2 norm (Euclidean distance)
    return (points - mean_p) / (0.5 * bbdiag)


# def find_neighbors(kdtree, index, points, k):
#     point_distances, patch_point_inds = kdtree.query(points[index, :], k=searchK)
    

    
#     aligne_neighborhood = standardize_pointcloud(points)
    
#     return neighborhood

def pca_points(patch_points):
    """
    Aligns a local patch using PCA

    Args:
        patch_points (np.ndarray): (k, 3) array of XYZ coordinates

    Returns:
        aligned_patch (np.ndarray): (k, 3) PCA-aligned patch
    """
    # 1. Center the patch around the mean
    mean = patch_points.mean(axis=0)
    centered = patch_points - mean

    # 2. Fit PCA
    pca = PCA(n_components=3)
    aligned = pca.fit_transform(centered)  # Applies rotation

    return aligned

def calculate_point_density(surface_area, pointcloud):
    point_density = len(pointcloud) / surface_area    
    return point_density



