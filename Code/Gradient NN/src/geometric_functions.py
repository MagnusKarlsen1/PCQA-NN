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
    return jnp.array([curvature, anisotropy, linearity, planarity, sphericity, variation]), eigenvectors, eigenvalues


@jit
def preprocess(points):
    mean_p = points.mean(axis=0)
    min_p, max_p = jnp.min(points, axis=0), jnp.max(points, axis=0)
    bbdiag = jnp.linalg.norm(max_p - min_p, ord=2) # Bounding box diagonal L2 norm (Euclidean distance)
    return (points - mean_p) / (0.5 * bbdiag)


def find_neighbors(kdtree, index, points, searchK):
    distances, indices = kdtree.query([points[index]], k=searchK)
    neighborhood = points[indices[0]]
    aligned_neighborhood = standardize_patch(neighborhood)
    
    return aligned_neighborhood, neighborhood, indices[0], distances

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


def points_inside_ball(pointcloud, kdtree, index, distances):
    """
    Computes number of points inside a ball defined by k nearest neighbors.

    Returns:
        count_inside_ball (int): Number of points inside the ball
        radius (float): Radius of the ball
    """
    # Step 1: Get radius as max distance to neighbor
    radius = np.max(distances)

    # Step 2: Query all points within radius
    indices_in_ball = kdtree.query_ball_point(pointcloud[index], r=radius)

    return int(len(indices_in_ball)), radius

def calculate_ball_density(radius: float, points_inside_ball):
    volume = (4/3) * np.pi*radius**3 
    
    volume_density = points_inside_ball/volume

    return volume_density

def min_max_scale(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def Edge_and_Plane(path, edge_k = 10, plane_overlap = 6, edge_thresh = 0.06, plane_thresh = 0.001, plane_deviation = 0.0001, min_planesize = 20):
    # Load the XYZ file as a DataFrame
    df = pd.read_csv(path, sep=" ", usecols=[0, 1, 2], names=["x", "y", "z"])
    df["row_index"] = df.index
    # Convert DataFrame to a PyntCloud object
    pc = PyntCloud(df)
    x = pc.points['x'].values.reshape(-1,1)
    y = pc.points['y'].values.reshape(-1,1)
    z = pc.points['z'].values.reshape(-1,1)
    pc_array = np.hstack((x,y,z))

    tree = pc.add_structure("kdtree")
    nbh_curv = pc.get_neighbors(k=edge_k, kdtree=tree) 
    eigen_curv = pc.add_scalar_field("eigen_values", k_neighbors=nbh_curv)
    curvfield = pc.add_scalar_field("curvature", ev = eigen_curv)
    curvature = pc.points['curvature('+str(edge_k+1)+')'].values

    tree = pc.add_structure("kdtree")
    nbh_omni = pc.get_neighbors(k=min_planesize, kdtree=tree) 
    eigen_omni = pc.add_scalar_field("eigen_values", k_neighbors=nbh_omni)
    omnivarfield = pc.add_scalar_field("omnivariance", ev = eigen_omni)
    omnivaraiance = pc.points['omnivariance('+str(min_planesize+1)+')'].values

    nbh_origin = np.hstack((pc.points['row_index'].values.reshape(-1,1), nbh_curv))
    
    plane_deviation = np.mean(L2norm_nbh(pc_array[nbh_origin,:],5)) * plane_deviation

    plane_size = min_planesize

    while plane_size >= min_planesize:
        plane_field = pc.add_scalar_field("plane_fit", max_dist=plane_deviation, max_iterations=500)
        plane = pc.points['is_plane'].values.reshape(-1,1)
        plane_size = np.sum(plane[:,0], axis=0)
        
        if plane_size >= min_planesize:
            row_indicies = pc.points['row_index'].values.reshape(-1,1)
            plane_col = np.zeros_like(x)
            plane_col[row_indicies[np.where(plane == 1)[0]]] = 1
            pc_array = np.hstack((pc_array,plane_col))
            pc.points = pc.points[pc.points['is_plane'] != 1]

    if pc_array.shape[1] == 3:
        pc_array = np.hstack((pc_array, np.zeros_like(x).reshape(-1,1)))

    num_planes = len(pc_array[0,3:])
    plane_count = np.count_nonzero(np.sum(pc_array[nbh_origin[:,:plane_overlap],3:], axis=1), axis=1)
    edge_index = np.where((plane_count > 1) | (curvature >= edge_thresh))[0]
    edges = np.zeros_like(x)
    edges[edge_index,0] = 1

    plane_index = np.where(((plane_count == 1) | (omnivaraiance <= plane_thresh)) & (edges[:,0] != 1))[0]
    planes = np.zeros_like(x)
    planes[plane_index,0] = 1
    return edges, planes


def Scalar_fields(path, k = 50):
    df = pd.read_csv(path, sep=" ", usecols=[0, 1, 2], names=["x", "y", "z"])
    df["row_index"] = df.index
    pc = PyntCloud(df)
    x = pc.points['x'].values.reshape(-1,1)
    y = pc.points['y'].values.reshape(-1,1)
    z = pc.points['z'].values.reshape(-1,1)
    pc_array = np.hstack((x,y,z))

    tree = pc.add_structure("kdtree")
    nbh_curv = pc.get_neighbors(k=k, kdtree=tree)

    eigenField = pc.add_scalar_field("eigen_values", k_neighbors=nbh_curv)

    curvfield = pc.add_scalar_field("curvature", ev = eigenField)
    curvature = pc.points['curvature('+str(k+1)+')'].values.reshape(-1,1)

    linfield = pc.add_scalar_field("linearity", ev = eigenField)
    linearity = pc.points['linearity('+str(k+1)+')'].values.reshape(-1,1)

    planfield = pc.add_scalar_field("planarity", ev = eigenField)
    planarity = pc.points['planarity('+str(k+1)+')'].values.reshape(-1,1)

    spherefield = pc.add_scalar_field("sphericity", ev = eigenField)
    sphericity = pc.points['sphericity('+str(k+1)+')'].values.reshape(-1,1)

    omnivarfield = pc.add_scalar_field("omnivariance", ev = eigenField)
    omnivaraiance = pc.points['omnivariance('+str(k+1)+')'].values.reshape(-1,1)

    eigentropyfield = pc.add_scalar_field("eigenentropy", ev = eigenField)
    eigentropy = pc.points['eigenentropy('+str(k+1)+')'].values.reshape(-1,1)

    anisofield = pc.add_scalar_field("anisotropy", ev = eigenField)
    anisotropy = pc.points['anisotropy('+str(k+1)+')'].values.reshape(-1,1)

    eigensum_field = pc.add_scalar_field("eigen_sum", ev = eigenField)
    eigensum = pc.points['eigen_sum('+str(k+1)+')'].values.reshape(-1,1)

    return curvature, linearity, planarity, sphericity, omnivaraiance, eigentropy, anisotropy, eigensum

def Get_variables(path, k=50, edge_k=10, edge_thresh=0.06, plane_thresh=0.001, plane_overlap=6, min_planesize=20, plot="No", save="yes"):
    curvature, linearity, planarity, sphericity, omnivaraiance, eigentropy, anisotropy, eigensum = Scalar_fields(path, k=k)
    edge, plane = Edge_and_Plane(path, edge_k=edge_k, plane_overlap=plane_overlap, edge_thresh=edge_thresh, plane_thresh=plane_thresh, min_planesize=min_planesize)
    xyz = np.loadtxt(path)[:,0:3]

    if save == "yes":
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(BASE_DIR, "Pointcloud_Data", "PC_variables.xyz")
        PC_variables = np.hstack((xyz, edge, plane, curvature, linearity, planarity, sphericity, omnivaraiance, eigentropy, anisotropy, eigensum))
        np.savetxt(save_path, PC_variables, fmt="%.6f", delimiter=" ")

    if plot == "yes":
        # edge
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            c=edge, cmap='viridis')

        fig.colorbar(scatter, ax=ax, label='edge')

        plt.show()

        # plane
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            c=plane, cmap='viridis')

        fig.colorbar(scatter, ax=ax, label='Plane')

        plt.show()

        # curvature
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            c=curvature, cmap='viridis')

        fig.colorbar(scatter, ax=ax, label='Curvature')

        plt.show()

        #Linearity
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            c=linearity, cmap='viridis')

        fig.colorbar(scatter, ax=ax, label='Linearity')

        plt.show()

        #Planarity
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xyz = np.loadtxt(path)

        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            c=planarity, cmap='viridis')

        fig.colorbar(scatter, ax=ax, label='Planarity')

        plt.show()

        #Sphericity
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xyz = np.loadtxt(path)

        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            c=sphericity, cmap='viridis')

        fig.colorbar(scatter, ax=ax, label='Sphericity')

        plt.show()

        #Omnivariance
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xyz = np.loadtxt(path)

        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            c=omnivaraiance, cmap='viridis')

        fig.colorbar(scatter, ax=ax, label='Omnivariance')

        plt.show()

        #Eigentropy
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xyz = np.loadtxt(path)

        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            c=eigentropy, cmap='viridis')

        fig.colorbar(scatter, ax=ax, label='Eigentropy')

        plt.show()

        #Anisotropy
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xyz = np.loadtxt(path)

        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            c=anisotropy, cmap='viridis')

        fig.colorbar(scatter, ax=ax, label='Anisotropy')

        plt.show()

        #Eigensum
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xyz = np.loadtxt(path)

        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            c=eigensum, cmap='viridis')

        fig.colorbar(scatter, ax=ax, label='Eigensum')

        plt.show()
    
    return [edge, plane, curvature, linearity, planarity, sphericity, omnivaraiance, eigentropy, anisotropy, eigensum]