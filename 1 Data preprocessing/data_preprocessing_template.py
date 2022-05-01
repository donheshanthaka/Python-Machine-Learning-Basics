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