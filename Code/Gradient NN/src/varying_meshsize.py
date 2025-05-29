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

path_part = os.path.join(BASE_DIR, "../../../Drawings", "Chain_whee.SLDPRT")
path_stl = os.path.join(BASE_DIR, "../../../Drawings", "Chain_whee.stl")
store_path = os.path.join(BASE_DIR, "../../../Drawings", "temp_cloud.xyz")
feature_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/varying_mesh_features.txt"))
cloud_PATH = os.path.abspath(os.path.join(BASE_DIR, f"../Data/Training_data/varying_mesh_cloud.txt"))

swApp = sw.start_SW()
model = sw.open_part(swApp, path_part)
sw.save_as_stl(model, path_stl)
SfA = sw.get_surface_area(model, unit = "mm2")

features_total = []
labels_total = []
pointcloud = []
i=0
for meshsize in [0.5,0.6,0.7]:
    i=i+1
    mf.sample_stl_by_point_distance(path_stl, store_path, meshsize)
    xyz = np.loadtxt(store_path)[:,0:3]
    sd = gf.calculate_point_density(SfA, xyz)
    print(sd)
    feature_array, grad_dist, radius = gf.Get_variables(store_path, 20, save="No", plot="no")
    features = np.hstack((feature_array[:,2:], grad_dist.reshape(-1,1)))

    indicies = np.where((xyz[:,0]>=(37/3)*(i-1)) & (xyz[:,0]<(37/3)*i))

    pointcloud.append(xyz[indicies])
    features_total.append(features[indicies])

features_total = np.vstack(features_total)
pointcloud = np.vstack(pointcloud)

header = ["edge_mean", "plane_mean", "curvature", "linearity", "planarity", "omnivaraiance", "eigensum", "Average radius", "pointsIN","grad_dist"]
header_pc = ["x", "y", "z"]

np.savetxt(feature_PATH, features_total, delimiter=" ", fmt="%.6f", header=" ".join(header))
np.savetxt(cloud_PATH, pointcloud, delimiter=" ", fmt="%.6f", header=" ".join(header_pc))