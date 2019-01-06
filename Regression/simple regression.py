import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###Reading the dataset
dataset=pd.read_csv("../data_files/Salary_Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

##Training and test set split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

###Simple Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

###Predict the model
y_predict=regressor.predict(X_test)


###Training data plot
plt.scatter(X_train,Y_train,edgecolors='red')
plt.plot(X_train,regressor.predict(X_train),color="blue")
# plt.plot(X_test,y_predict,color="yellow")
# plt.plot(X_test,Y_test,color="pink")
plt.title("Salary Vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt._show

###Testing data plot
plt.scatter(X_test,y_predict,color="yellow")
plt.plot(X_test,Y_test,color="pink")
plt.title("Salary Vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt._show
