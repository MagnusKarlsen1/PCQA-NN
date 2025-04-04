import pandas as pd
import sys
import numpy as np
import pythoncom
import argparse
import pickle
import jax 
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
import torch
from scipy.spatial import cKDTree
from scipy.spatial import KDTree

import random
import numpy.linalg as LA
import os
from tqdm import tqdm
from IPython.display import clear_output


shape_configs = {
    "ball" : [{"radius": 5},
              {"radius": 10},
              {"radius": 15},
              {"radius": 20}],
    "box" : [],
    
    "angle_curve" : [],
    
    
    
}



def run_batch(neighborhood_size=20):
    for shape, param_list in shape_configs.items():
        for param_set in param_list:
            for variant in scan_variants:
                print(f"🚀 Running: {shape} | params: {param_set} | variant: {variant}")
                try:
                    main(
                        neighborhood_size,
                        params=param_set,
                        shape=shape,
                        noise=variant["noise"],
                        holes=variant["holes"]
                    )
                except Exception as e:
                    print(f"❌ Error processing {shape} with {param_set}: {e}")
































































