# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:44:32 2022

@author: Don Hesha
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv("Data.csv")

# First take all the rows and then take all the columns except the last
# Aka independent variable vector
X = dataset.iloc[:, :-1].values

# Dependent variable vector
Y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
# taking indexs 1 upto 2 (1:3)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])