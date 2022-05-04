# -*- coding: utf-8 -*-
"""
Created on Mon May  2 04:25:20 2022

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
y = dataset.iloc[:, 3].values


# Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train)
"""
















