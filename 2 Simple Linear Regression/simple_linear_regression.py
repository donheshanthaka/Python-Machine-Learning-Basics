# -*- coding: utf-8 -*-
"""
Created on Mon May  2 22:13:23 2022

@author: Don Hesha
"""

# Simple Linear Regression


# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv("Salary_Data.csv")

# First take all the rows and then take all the columns except the last
# aka independent variable matrix
X = dataset.iloc[:, :-1].values

# Dependent variable vector
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)



"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)






















