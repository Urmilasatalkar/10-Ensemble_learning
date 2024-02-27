# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 00:08:18 2024

@author: rucha
"""

'''
4.	Data privacy is always an important factor to safeguard their
 customers' details. For this, password strength is an important 
 metric to track. Build an ensemble model to classify the user’s
 password strength.

Minimize: Reduce the risk of unauthorized access by accurately classifying weak passwords.

Maximize: Enhance user security by accurately identifying strong passwords.

Business constraints
Maintain high model accuracy to avoid false classifications

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('C:/9-ML/Ensemble_learning/Ensemble_Password_Strength')
df

df.columns
# Index(['characters', 'characters_strength'], dtype='object')

#data dict
df.dtypes

'''
characters             object       nominal     relevent
characters_strength     int64       numeric     relevent
dtype: object

'''

#as there are only two cols in this dataset out of which one is nominal 
#and another is target column so there are no outliers

#we need to perform label encoding on 1st column

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['characters'] = df['characters'].astype(str)
df['characters'] = label_encoder.fit_transform(df['characters'])
df

print(df.isnull().sum())

'''
characters             0
characters_strength    0
dtype: int64

'''

#there no null values

#dataset is ready for further processing 

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
print(y)
x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

#create adaboost classifier

ada_model =  AdaBoostClassifier(n_estimators=100,learning_rate = 1)

#n_estimator is no. of weak learners
#learning rate , it contributes weight of weak leaners, bydefault it is 1
#train the model


model = ada_model.fit(x_train, y_train)
#predict the results

y_pred =  model.predict(x_test)

print('accuracy = ',metrics.accuracy_score(y_test, y_pred))

#let us try for another base model

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
#here base model is changed

Ada_model = AdaBoostClassifier(n_estimators=50, base_estimator=lr, learning_rate=1)

model = Ada_model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print('accuracy = ',metrics.accuracy_score(y_test, y_pred))



