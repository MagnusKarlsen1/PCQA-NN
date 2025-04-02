import pymeshlab
import numpy as np
from sklearn.neighbors import NearestNeighbors



def sample_stl_by_point_distance(input_stl, output_xyz, sampling_distance):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_stl)

    # Poisson disk sampling
    ms.generate_sampling_poisson_disk(
        radius=pymeshlab.PureValue(sampling_distance)
    )

    # Get the resampled point cloud
    sampled_mesh = ms.current_mesh()
    points = sampled_mesh.vertex_matrix()  # shape (N, 3)
    
    # Save only XYZ (first 3 columns) to txt or xyz format
    np.savetxt(output_xyz, points[:, :3], fmt="%.6f")

    print(f"✅ Resampled point cloud saved (XYZ only): {output_xyz}")



def save_neighborhood_to_txt(patch_points, filename="neighborhood.txt"):
    np.savetxt(filename, patch_points, fmt="%.6f", delimiter=" ")
    print(f"Saved neighborhood to {filename}")




# def create_mixed_distances(pointcloud, min_dist, max_dist, num_iterations):
    
    
#     return 


def create_noise(pointcloud, noise_std, noise_mean):
    
    noise = np.random.normal(loc=noise_mean, scale=noise_std, size=pointcloud.shape)
    noisy_points = pointcloud + noise
    return noisy_points
    


def create_mesh_holes(pointcloud, num_holes: int, hole_size: int):
    points_remaining = pointcloud.copy()
    nbrs = NearestNeighbors(n_neighbors=hole_size+1, algorithm='auto').fit(points_remaining)

    for _ in range(num_holes):
        if len(points_remaining) <= hole_size:
            break  # Prevent removing too many

        # Randomly choose a seed point
        seed_idx = np.random.choice(len(points_remaining))
        seed_point = points_remaining[seed_idx].reshape(1, -1)

        # Find the seed's k neighbors (including itself)
        _, indices = nbrs.kneighbors(seed_point)

        # Flatten and remove these points
        indices_to_remove = indices[0]
        mask = np.ones(len(points_remaining), dtype=bool)
        mask[indices_to_remove] = False
        points_remaining = points_remaining[mask]

        # Refit KNN model on remaining points
        nbrs.fit(points_remaining)

    return points_remaining




















