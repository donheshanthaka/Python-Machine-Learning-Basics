# -*- coding: utf-8 -*-
"""
Created on Wed May  4 03:12:17 2022

@author: Don Hesha
"""

# Polynomial Regression

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


# Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Fitting polynomial regression to the dataset

# Converted the original matrix of X to new maxtrix of features containing the original independent position
# ... levels and its associated polynomial terms
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# Creation of a new linear regression model with new X_poly matrix
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)



# Visualizing the linear regression results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()


# Visualizing the polynomial regression results

# Create a vector of X with more resolution
X_grid = np.arange(min(X), max(X), 0.1)

# Reshaping the newly created vector to a matirx
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()



# Predicting a new result with linear regression
lin_reg.predict(np.array([6.5]).reshape(1, 1))


# Predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1, 1)))
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))















