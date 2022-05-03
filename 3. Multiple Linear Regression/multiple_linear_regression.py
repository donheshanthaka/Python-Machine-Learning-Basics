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


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])

# State column
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)


# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicitng the test results
y_pred = regressor.predict(X_test)



# Building the optimal model using backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)

# Ordinary Least Squares
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# Check the summary and remove the highest significant variable from the array
regressor_OLS.summary()

# x2 was the highest on this check (Always check the index from original X, not X_opt)
X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()


regressor_OLS.summary()


# x1 was the least on this check
X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()


regressor_OLS.summary()


# x4 was the least on this check
X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()


regressor_OLS.summary()


# x4 was the least on this check
X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()


# x5 was the least on this check
X_opt = np.array(X[:, [0, 3]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()




