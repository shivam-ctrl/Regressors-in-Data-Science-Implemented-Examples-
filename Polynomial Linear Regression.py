#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 00:11:21 2021

@author: shivamaditya
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values # we used 1:2 instead of 1 to treat X as a matrix
#print(X)
Y = dataset.iloc[:, -1].values
#print(Y)

# Here we won't be splitting the dataset as it too small and so we are going to use the complete data for training

# Fitting Simple Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X = np.array(X).reshape(-1, 1) # changing format to fit data(necessary)
regressor.fit(X, Y)

# Fitting Polinomial Linear Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# Putting more columns of powers of x here we have taken upto power n
#poly_reg = PolynomialFeatures(degree=2)
poly_reg = PolynomialFeatures(degree=9) # The more expansion of x powers(till higher degree of x) we consider the more accurate our model gets
X_poly = poly_reg.fit_transform(X) # Table of column of values x^0, x^1, x^2
#print(X_poly)
from sklearn.linear_model import LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising the Linear Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Salary Predictor (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# Accuracy of Prediction(Linear Regression)
from sklearn.metrics import r2_score
score = r2_score(Y, regressor.predict(X))
print('Accuracy : ',score)

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Salary Predictor (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# Accuracy of Prediction(Polynomial Regression)
from sklearn.metrics import r2_score
score_2 = r2_score(Y, lin_reg_2.predict(X_poly))
print('Accuracy : ',score_2)
