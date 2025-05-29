import sys
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
from pcdiff import knn_graph
from numpy import linalg as LA
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
import pymeshlab
import win32com.client
import pythoncom
from win32com.client import VARIANT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAWINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../Drawings"))
sys.path.append(DRAWINGS_PATH)
GRADIENT_NN_SRC = os.path.abspath(os.path.join(BASE_DIR, "../../Code/Gradient NN/src"))
sys.path.append(GRADIENT_NN_SRC)

import geometric_functions as gf
import meshlab_functions as mf
import solidworks_functions as sw

path = os.path.join(BASE_DIR, "../../../code/Leihui Code/dataset/SelfGeneratedClouds/", "Chain_wheel_05.xyz")
# path_stl = os.path.join(BASE_DIR, "../../../Drawings", "Chain_whee.stl")
# store_path = os.path.join(BASE_DIR, "../../../Drawings", "temp_cloud.xyz")
# feature_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/Chain_whee_features.txt"))
# label_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/Chain_whee_labels.txt"))
# cloud_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/Chain_whee_cloud.txt"))

PC_array, grad, radius, xyz = gf.Get_variables(path,k=20,save="No")

features = {
    'Curvature': PC_array[:,2],
    'Linearity': PC_array[:,3],
    'Planarity': PC_array[:,4],
    'Omnivariance': PC_array[:,5],
    'Eigensum': PC_array[:,6],
    'Gradient difference': grad
}

fig = plt.figure(figsize=(18, 10))
cols = 3
rows = (len(features) + cols - 1) // cols

for i, (title, values) in enumerate(features.items(), start=1):
    ax = fig.add_subplot(rows, cols, i, projection='3d')
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=values, cmap='viridis')
    ax.set_title(title, pad=5)  # Adds space between title and plot
    # Slightly shrink and move colorbar to the side
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label(title)

plt.subplots_adjust(hspace=0.2, wspace=0.1)
plt.savefig("feature_plots.png", dpi=2000, bbox_inches='tight')
plt.show()