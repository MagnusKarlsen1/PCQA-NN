import jax.numpy as jnp
from jax import jit
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from pcdiff import knn_graph, estimate_basis, build_grad_div, laplacian, coords_projected, gaussian_weights, weighted_least_squares, batch_dot
import torch
import jax

def standardize_patch(pointcloud):
    mean = pointcloud.mean(axis=0)
    centered = pointcloud - mean

    pca = PCA(n_components=3)
    aligned = pca.fit_transform(centered)

    # Optional: scale to unit sphere
    max_norm = np.max(np.linalg.norm(aligned, axis=1))
    standardized = aligned / max_norm if max_norm > 0 else aligned

    return standardized


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
    return jnp.array([curvature, anisotropy, linearity, planarity, sphericity, variation])


@jit
def preprocess(points):
    mean_p = points.mean(axis=0)
    min_p, max_p = jnp.min(points, axis=0), jnp.max(points, axis=0)
    bbdiag = jnp.linalg.norm(max_p - min_p, ord=2) # Bounding box diagonal L2 norm (Euclidean distance)
    return (points - mean_p) / (0.5 * bbdiag)


def find_neighbors(kdtree, index, points, searchK):
    _, indices = kdtree.query([points[index]], k=searchK)
    neighborhood = points[indices[0]]
    aligned_neighborhood = standardize_patch(neighborhood)
    
    return aligned_neighborhood, neighborhood, indices[0]

def pca_points(patch_points):
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



def xyz_projected(pos, normal, x_basis, y_basis, edge_index, k=None):
    """Projects neighboring points to the tangent basis
    and returns the local coordinates.

    Args:
        pos (Tensor): an [N, 3] tensor with the point positions.
        normal (Tensor): an [N, 3] tensor with normals per point.
        x_basis (Tensor): an [N, 3] tensor with x basis per point.
        y_basis (Tensor): an [N, 3] tensor with y basis per point.
        edge_index (Tensor): indices of the adjacency matrix of the k-nn graph [2, N * k].
        k (int): the number of neighbors per point.
    """
    row, col = edge_index
    k = (row == 0).sum() if k is None else k

    # Compute coords
    normal = np.tile(normal[:, None], (1, k, 1)).reshape(-1, 3)
    x_basis = np.tile(x_basis[:, None], (1, k, 1)).reshape(-1, 3)
    y_basis = np.tile(y_basis[:, None], (1, k, 1)).reshape(-1, 3)
    local_pos = pos[col] - pos[row]
    z_pos = batch_dot(local_pos, normal)
    local_pos = local_pos - normal * z_pos
    x_pos = batch_dot(local_pos, x_basis).flatten()
    y_pos = batch_dot(local_pos, y_basis).flatten()
    coords = np.stack([x_pos, y_pos], axis=1)

    return coords, z_pos



def grad_curvature(pos, k, kernel_width=1, regularizer=1e-8, shape_regularizer=None):

    edge_index = knn_graph(pos, k)

    normal, x_basis, y_basis = estimate_basis(pos, edge_index)

    row, col = edge_index

    coords, z_pos = xyz_projected(pos, normal, x_basis, y_basis, edge_index, k)

    dist = LA.norm(pos[col] - pos[row], axis=1)
    weights = gaussian_weights(dist, k, kernel_width)

    if shape_regularizer is None:
        wls = weighted_least_squares(coords, weights, k, regularizer)
    else:
        wls, wls_shape = weighted_least_squares(coords, weights, k, regularizer, shape_regularizer)

    C = (wls * z_pos).reshape(-1, k, 6).sum(axis=1)

    # df/dx^2 = 2*c3
    grad_xx = 2 * C[:,3]
    
    # df/dxdy = c4
    grad_xy = C[:,4]
    
    # df/dy^2 = 2*c5
    grad_yy = 2 * C[:,5]

    C = C[row]

    # df/dx = c1 + 2*c3*x + c4*y
    grad_x = C[:,1] + 2 * C[:,3] * coords[:,0] + C[:,4] * coords[:,1]

    # df/dy = c2 + 2*c5*y + c4*x
    grad_y = C[:,2] + 2 * C[:,5] * coords[:,1] + C[:,4] * coords[:,0]
    
    grad = np.column_stack((grad_x, grad_y))
    curvature = grad_xx + 2 * grad_xy + grad_yy

    return grad.reshape(-1, k, 2), curvature, edge_index



def get_nn_data(pos, k, comparison_size):

    grad, curv, edge_index = grad_curvature(pos, k, kernel_width=1, regularizer=1e-8, shape_regularizer=None)
    row, col = edge_index

    grad_dist = L2norm_nbh(grad, comparison_size)
    
    pos_dist = L2norm_nbh(pos[col].reshape(-1, k, 3), comparison_size)

    print(grad_dist.shape)
    print(pos_dist.shape)
    print(curv.shape)

    return pos_dist, grad_dist, curv

def L2norm_nbh(data, comparison_size, origin_index=0):
    # Convert PyTorch tensor to JAX array if needed
    if isinstance(data, torch.Tensor):
        data = jax.device_put(data.detach().cpu().numpy())  # Convert to JAX arrayee

    data_jax = jnp.array(data)[:, 1:comparison_size+1, :]  # Ensure JAX array

    # Select the origin point from each batch (shape: (batch_size, num_features))
    origin = jnp.array(data)[:, origin_index, :]

    # Compute L2 norm for each row in the neighborhood (shape: (batch_size, num_neighbors))
    dist = jnp.linalg.norm(data_jax - origin[:, None, :], axis=2)

    return dist  # Shape: (batch_size, num_neighbors)