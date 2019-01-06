import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###Importing Data set
data=pd.read_csv('../data_files/Position_Salaries.csv')
##upper bound is excluded else X would have been a vector rather than a matrix
X=data.iloc[:,1:2].values
Y=data.iloc[:,2:3].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,Y)

###Predicting new dataset
y_pred = regressor.predict(np.array([[6.5]]))

X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Random Forest Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
