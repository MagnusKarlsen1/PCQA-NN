import numpy as np
import pandas as pd
import sys
import argparse
import pickle
import jax 
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
import torch
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
import importlib
import random
import numpy.linalg as LA
import os
from tqdm import tqdm
from IPython.display import clear_output
import time



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRADIENT_NN_SRC = os.path.abspath(os.path.join(BASE_DIR, "../../Code/Gradient NN/src"))
sys.path.append(GRADIENT_NN_SRC)
import geometric_functions as gf

output_path_point = os.path.join(BASE_DIR, "..", "Leihui Code", "dataset", "scanning_repository", "bunny.xyz")


def evaluator_local(pointcloud):
    
    Features = gf.Get_variables(Pointcloud,20,save="no")
    
    
    
    
    
    
    return 












