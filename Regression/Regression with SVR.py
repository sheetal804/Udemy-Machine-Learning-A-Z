import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###Importing Data set
data=pd.read_csv('../data_files/Position_Salaries.csv')
##upper bound is excluded else X would have been a vector rather than a matrix
X=data.iloc[:,1:2].values
Y=data.iloc[:,2].values

###Fitting SVR to dataset
from sklearn.svm import SVR
regressor=SVR(kernel="rbf")
regressor.fit(X,Y)



###Predicting new dataset
y_pred=regressor.predict(6.5)


###Visualization SVR results

plt.scatter(X,Y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

