import sys
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
from pcdiff import knn_graph
from numpy import linalg as LA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAWINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../Drawings"))
sys.path.append(DRAWINGS_PATH)
GRADIENT_NN_SRC = os.path.abspath(os.path.join(BASE_DIR, "../../Code/Gradient NN/src"))
sys.path.append(GRADIENT_NN_SRC)

import geometric_functions as gf

path = "C:/Users/aagaa/OneDrive - Aarhus universitet/Dokumenter/GitHub/R-D/Code/Leihui Code/dataset/SelfGeneratedClouds/gear_shaft.xyz"
#path = "C:/Users/aagaa/OneDrive - Aarhus universitet/Dokumenter/GitHub/R-D/Drawings/STL/test.xyz"
#gf.Get_variables(path, k=50,plot="no", save="no", edge_k=10,edge_thresh=0.06)

k=50

xyz = np.loadtxt(path)[:,0:3]
upper = xyz[np.where(xyz[:,2] <= np.mean(xyz[:,2])),:][0]
lower = xyz[np.where(xyz[:,2] > np.mean(xyz[:,2])),:][0]

indices = np.arange(len(lower))
n_delete = int(np.round(lower.shape[0] * 0.5))
indices_to_delete = np.random.choice(indices, n_delete, replace=False)
lower = np.delete(lower, indices_to_delete, axis=0)

xyz = np.vstack((upper, lower))

row, col = knn_graph(xyz, k)

R = np.linalg.norm(xyz - xyz[col.reshape(-1,k)[:,k-1]], axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    c=R, cmap='viridis')

fig.colorbar(scatter, ax=ax, label='R')

plt.show()