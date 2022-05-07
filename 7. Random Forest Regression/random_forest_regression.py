# -*- coding: utf-8 -*-
"""
Created on Sat May  7 23:02:14 2022

@author: Don Hesha
"""

# Random Forest Regression


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

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train)
"""


# Fitting the Random Forest Regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)


# Predicting a new result
y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))
y_pred = regressor.predict([[6.5]])



# Visualizing the regression results (For higher resolution and smoother curve)

# Create a vector of X with more resolution
X_grid = np.arange(min(X), max(X), 0.1)

# Reshaping the newly created vector to a matirx
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Truth or Bluff (Random Forest Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()









