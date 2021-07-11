#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 03:01:59 2021

@author: shivamaditya
"""

# Support Vector Regression

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values # we used 1:2 instead of 1 to treat X as a matrix
#print(X)
Y = dataset.iloc[:, -1].values # Prediction variable
#print(Y)
# Reshaping 1D arrays
X = np.array(X).reshape(-1, 1) # changing format to fit data(necessary) 
Y = np.array(Y).reshape(-1, 1) # changing format to fit data(necessary)

# Feature Scaling (In most of the regressions Feature scaling is done on its own, but not in SVR)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)
#print(X) and print(Y)


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)


#Prediction a new result
prediction_values = sc_X.transform(np.array([6.5]).reshape(-1, 1)) # Predicting for a single vale 6.5
y_pred = sc_Y.inverse_transform(regressor.predict(prediction_values)) # Our prediction will be stored here 

# Visualising the SVR results
plt.scatter(X, Y, color='red') # Real points
plt.plot(X, regressor.predict(X), color='blue') # Our regression model
plt.title('Truth or Bluff(SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
''' Also try without feature scaling and visualise '''

