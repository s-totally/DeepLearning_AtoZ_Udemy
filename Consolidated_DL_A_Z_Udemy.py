# Create environment as below: From anaconda prompt: create -n tensorflow_env python=3.5 anaconda
# activate tensorflow_env
# conda install theano
# conda install mingw libpython
# pip install tensorflow
# pip install keras
# Then run spyder or jupyter notebook

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################3
# Data preprocessing
###############################3

dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()
dataset.shape

X = dataset.iloc[:, 3: 13]
y = dataset.iloc[:, 13]
print(X.shape, y.shape)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label encoding of gender variable
label_encoder_gender = LabelEncoder()
X.iloc[:, 2] = label_encoder_gender.fit_transform(X.iloc[:, 2])
X= pd.DataFrame(X)
X[0:3]

X.rename(columns={'Gender':'Male'}, inplace=True)

# DUmmy variable encoding of country 
X= pd.get_dummies(X, columns=['Geography'])

# Drop one of the dummies 

X = X.drop(['Geography_France'], axis=1)

# Train test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)