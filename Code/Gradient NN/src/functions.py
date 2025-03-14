
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA





class Dataexplorer:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.features = data.drop(columns=[target_column])
        self.target = data[target_column]
        
        
    def 