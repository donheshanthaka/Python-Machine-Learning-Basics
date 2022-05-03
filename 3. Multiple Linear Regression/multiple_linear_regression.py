# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:06:59 2022

@author: Don Hesha
"""

# Multiple Linear Regression

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv("50_Startups.csv")

# First take all the rows and then take all the columns except the last
# Aka independent variable vector
X = dataset.iloc[:, :-1].values

# Dependent variable vector
y = dataset.iloc[:, 4].values


# Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)




"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""