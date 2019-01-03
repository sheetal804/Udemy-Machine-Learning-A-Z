import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###Importing Data set
data=pd.read_csv('../data_files/Data.csv')
X=data.iloc[:,:-1].values
y=data.iloc[:,3].values

### Data Cleaning Handling Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
X[:,1:3]=imputer.fit_transform(X[:,1:3])

### Encoding Nominal Values
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
label_encoder=LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
y=label_encoder.fit_transform(y)

### Splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

###Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)