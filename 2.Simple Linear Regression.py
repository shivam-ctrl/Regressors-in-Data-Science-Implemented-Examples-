# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Simple Linear Regression

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
dataset = pd.read_csv('/Users/shivamaditya/Desktop/[Tutsgalaxy.com] - Machine Learning A-Z™ Hands-On Python & R In Data Science/additional files/2.Regression/Section 6 - Simple Linear Regression/Python/Salary_Data.csv')
X = dataset.iloc[:, 0]       # Independent variable
Y = dataset.iloc[:, 1]       # Dependent variable
# Check X and Y

# Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
# Check the above variables

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train = np.array(X_train).reshape(-1, 1) # changing format to fit data(necessary)
regressor.fit(X_train, Y_train)

# Predicting the Test set results
X_test = np.array(X_test).reshape(-1, 1) # changing format to fit data(necessary)
Y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, Y_train, color = 'red') # training observation points
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # trained regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'red') # Observed points
plt.plot(X_test, Y_pred, color = 'yellow') # Regression line(same line as above)
plt.scatter(X_test, Y_pred, color = 'black') # Predicted points according to line(all points will be on the line)
plt.title('Salary vs Experience (Test set results)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Calculating the Accuracy of Predictions
from sklearn.metrics import r2_score
score = r2_score(Y_test, Y_pred)
print(score)