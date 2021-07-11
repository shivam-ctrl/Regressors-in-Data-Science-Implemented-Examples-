# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Multiple Linear Regression

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values       # Independent variable
Y = dataset.iloc[:, 4].values       # Dependent variable
# Check X and Y

# Categorical Data Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # for dummy encoding
from sklearn.compose import ColumnTransformer
# State Column
# Simple Encoding
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#print(X)
# Dummy Encoding
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)
#print(X)

# Avoiding Dummy Variable Trap
X = X[:, 1:] # Removing one(first) dummy variable
#print(X)

# Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# Check the above variables

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

# Calculating the Accuracy of Predictions
from sklearn.metrics import r2_score
score = r2_score(Y_test, Y_pred)
print(score)

# Building the optimal model using Backward Elimination
# Assuming SL = 5%(0.05)
import statsmodels.api as sm
# Putting a constant b0 in the beginning
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # Final X columns will be stored here
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary() # To get the p values

X_opt = X[:, [0, 1, 3, 4, 5]] # Removing highest p value variable
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]] # Removing highest p value variable
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]] # Removing highest p value variable
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]] # Removing highest p value variable
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
# Now no p value is greater than 0.05, so this is our final model

X_opt = X[:, [3]]
# Splitting dataset into training and test set(for new model)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_opt, Y, test_size = 0.2, random_state = 0)
# Check the above variables

# Fitting Multiple Linear Regression to the Training set(for new model)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results(for new model)
Y_pred = regressor.predict(X_test)

# Calculating the Accuracy of Predictions(for new model)
from sklearn.metrics import r2_score
score = r2_score(Y_test, Y_pred)
print(score) # We could see that our accuracy increased






