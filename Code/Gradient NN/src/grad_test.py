import sys
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
from pcdiff import knn_graph
from numpy import linalg as LA
from scipy.spatial import cKDTree
from scipy.spatial import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAWINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../Drawings"))
sys.path.append(DRAWINGS_PATH)
GRADIENT_NN_SRC = os.path.abspath(os.path.join(BASE_DIR, "../../Code/Gradient NN/src"))
sys.path.append(GRADIENT_NN_SRC)
DRAWINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../Drawings"))
sys.path.append(DRAWINGS_PATH)

import geometric_functions as gf
import meshlab_functions as mf

path = "C:/Users/aagaa/OneDrive - Aarhus universitet/Dokumenter/GitHub/R-D/Code/Leihui Code/dataset/SelfGeneratedClouds/Chain_wheel05mm.xyz"
path_in = "C:/Users/aagaa/OneDrive - Aarhus universitet/Dokumenter/GitHub/R-D/Drawings/Chain_whee.STL"
#gf.Get_variables(path, k=50,plot="no", save="no", edge_k=10,edge_thresh=0.06)


mf.sample_stl_by_point_distance(path_in, path, 0.5)

xyz = np.loadtxt(path)[:,0:3]

feature_array, grad_dist, radius = gf.Get_variables(path, 20, save="No", plot="no")
features = np.hstack((feature_array, grad_dist.reshape(-1,1), radius))

SfA = 9796.870
sd = gf.calculate_point_density(SfA, xyz)
labels = np.full((len(xyz),2), sd)

header_label =["Label"]
header = ["edge_mean", "plane_mean", "curvature", "linearity", "planarity", "omnivaraiance", "eigensum", "grad_dist", "radius"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
feature_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/testcloud_feat2.txt"))
label_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/testcloud_lab2.txt"))
np.savetxt(feature_PATH, features, delimiter=" ", fmt="%.6f", header=" ".join(header))
np.savetxt(label_PATH, labels, delimiter=" ", fmt="%.6f", header=" ".join(header_label))
k=20

upper = xyz[np.where(xyz[:,2] <= np.mean(xyz[:,2])),:][0]
lower = xyz[np.where(xyz[:,2] > np.mean(xyz[:,2])),:][0]

indices = np.arange(len(lower))
n_delete = int(np.round(lower.shape[0] * 0.5))
indices_to_delete = np.random.choice(indices, n_delete, replace=False)
lower = np.delete(lower, indices_to_delete, axis=0)

xyz = np.vstack((upper, lower))

tree = cKDTree(xyz)
distances, _ = tree.query(xyz, k=4)
radius_upper = np.mean(distances[:len(upper), 3]) * 4
radius_lower = np.mean(distances[len(upper):, 3]) * 4
upper_neighbors = tree.query_ball_point(xyz, r=radius_upper)
lower_neighbors = tree.query_ball_point(xyz, r=radius_lower)
upper_sizes = np.array([len(row) for row in upper_neighbors])
lower_sizes = np.array([len(row) for row in lower_neighbors])
nbh_sizes = np.hstack((upper_sizes[:len(upper)], lower_sizes[len(upper):]))

print(radius_upper)
print(radius_lower)

row, col = knn_graph(xyz, k)
#print(len(col.reshape(-1,k)))
R = np.linalg.norm(xyz - xyz[col.reshape(-1,k)[:,k-1]], axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    c=nbh_sizes, cmap='viridis')

fig.colorbar(scatter, ax=ax, label='R')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    c=R, cmap='viridis')

fig.colorbar(scatter, ax=ax, label='R')

plt.show()