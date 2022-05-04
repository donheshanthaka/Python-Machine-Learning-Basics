# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:03:38 2022

@author: Don Hesha
"""

# Regression Template

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv("Position_Salaries.csv")

# First take all the rows and then take all the columns except the last
# Aka independent variable vector
# By adding the :2 after 1 will make the vector a matrix which is much suitable for this kind of computation
X = dataset.iloc[:, 1:2].values

# Dependent variable vector
y = dataset.iloc[:, 2].values


# No dataset splitting since a small number of samples are used
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting the regression model to the dataset
# Create your regressor here


# Predicting a new result
y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))
y_pred = regressor.predict([[6.5]])


# Visualizing the polynomial regression results

plt.scatter(X, y, color = "red")
plt.plot(X, regressor.predict(X), color = "blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()



# Visualizing the polynomial regression results (For higher resolution and smoother curve)

# Create a vector of X with more resolution
X_grid = np.arange(min(X), max(X), 0.1)

# Reshaping the newly created vector to a matirx
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()









