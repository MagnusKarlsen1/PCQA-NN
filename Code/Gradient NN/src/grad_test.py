import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAWINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../Drawings"))
sys.path.append(DRAWINGS_PATH)
GRADIENT_NN_SRC = os.path.abspath(os.path.join(BASE_DIR, "../../Code/Gradient NN/src"))
sys.path.append(GRADIENT_NN_SRC)

import geometric_functions as gf

path = "C:/Users/aagaa/OneDrive - Aarhus universitet/Dokumenter/GitHub/R-D/Code/Leihui Code/dataset/SelfGeneratedClouds/bevel_gear.xyz"
#path = "C:/Users/aagaa/Documents/GitHub/R-D/Code/Leihui Code/dataset/scanning_repository/bunny.xyz"
Get_variables(path, k=20,plot="yes", edge_k=10,edge_thresh=0.06)