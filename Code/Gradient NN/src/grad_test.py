import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAWINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../Drawings"))
sys.path.append(DRAWINGS_PATH)
GRADIENT_NN_SRC = os.path.abspath(os.path.join(BASE_DIR, "../../Code/Gradient NN/src"))
sys.path.append(GRADIENT_NN_SRC)

import geometric_functions as gf

#path = "C:/Users/aagaa/OneDrive - Aarhus universitet/Dokumenter/GitHub/R-D/Code/Leihui Code/dataset/SelfGeneratedClouds/worm_shaft.xyz"
path = "C:/Users/aagaa/OneDrive - Aarhus universitet/Dokumenter/GitHub/R-D/Drawings/STL/test.xyz"
gf.Get_variables(path, k=50,plot="yes", save="no", edge_k=10,edge_thresh=0.06)