import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###Importing Data set
data=pd.read_csv('../data_files/Position_Salaries.csv')
##upper bound is excluded else X would have been a vector rather than a matrix
X=data.iloc[:,1:2].values
Y=data.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly)

###Visualization Linear Regression Model

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

###Visualization Polynomial Regression Model

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color="blue")
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

