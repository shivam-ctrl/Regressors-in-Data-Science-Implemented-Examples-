#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 00:17:03 2021

@author: shivamaditya
"""

# Random Forest Regression

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Python/Position_Salaries.csv')
X = dataset.iloc[:, 1].values # we used 1:2 instead of 1 to treat X as a matrix
#print(X)
Y = dataset.iloc[:, -1].values # Prediction variable
#print(Y)
# Reshaping 1D arrays
X = np.array(X).reshape(-1, 1) # changing format to fit data(necessary) 
Y = np.array(Y).reshape(-1, 1) # changing format to fit data(necessary)

# Fitting Random Forest Regression to the Dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0) # n_estimators : Number of trees
regressor.fit(X, Y)

#Prediction a new result
prediction_values = np.array([6.5]).reshape(-1, 1) # Predicting for a single vale 6.5
y_pred = regressor.predict(prediction_values) # Our prediction will be stored here 

# Visualising the Random Forest Regression results
plt.scatter(X, Y, color='red') # Real points
plt.plot(X, regressor.predict(X), color='blue') # Our regression model
plt.title('Truth or Bluff(Decision Tree Regression) : non realistic')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualise the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff(Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
