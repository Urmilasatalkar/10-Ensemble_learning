# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:05:27 2024

@author: rucha
"""

#assignment 1
#ensemble technique


'''
1.	Given is the diabetes dataset. Build an ensemble model to 
correctly classify the outcome variable and improve your model
prediction by using GridSearchCV. You must apply Bagging, Boosting,
Stacking, and Voting on the dataset.  

****business objective

minimize - 
Minimize the misclassification rate of diabetes diagnosis

maximize -
Maximize the efficiency of diabetes diagnosis to optimize healthcare resources, improve patient outcomes, and reduce healthcare costs associated with undiagnosed or misdiagnosed cases.

constraints -
Model Interpretability (Ensure that the model is interpretable and transparent )

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/9-ML/Ensemble_learning/Diabeted_Ensemble.csv')
df

df.columns
'''Index([' Number of times pregnant', ' Plasma glucose concentration',
       ' Diastolic blood pressure', ' Triceps skin fold thickness',
       ' 2-Hour serum insulin', ' Body mass index',
       ' Diabetes pedigree function', ' Age (years)', ' Class variable'],
      dtype='object')'''

df.describe()

'''
df.describe()
Out[8]: 
        Number of times pregnant  ...   Age (years)
count                 768.000000  ...    768.000000
mean                    3.845052  ...     33.240885
std                     3.369578  ...     11.760232
min                     0.000000  ...     21.000000
25%                     1.000000  ...     24.000000
50%                     3.000000  ...     29.000000
75%                     6.000000  ...     41.000000
max                    17.000000  ...     81.000000

[8 rows x 8 columns]
'''

#data dictionary
df.dtypes

'''
Number of times pregnant          int64     quantitative    relevent    
 Plasma glucose concentration      int64    quantitative    relevent
 Diastolic blood pressure          int64    quantitative    relevent
 Triceps skin fold thickness       int64    quantitative    relevent
 2-Hour serum insulin              int64    quantitative    relevent
 Body mass index                 float64    quantitative    relevent
 Diabetes pedigree function      float64    quantitative    relevent
 Age (years)                       int64    quantitative    relevent
 Class variable                   object    qualitative (nominal)    relevent
dtype: object
'''

#check null values
print(df.isnull().sum())

'''
Number of times pregnant        0
 Plasma glucose concentration    0
 Diastolic blood pressure        0
 Triceps skin fold thickness     0
 2-Hour serum insulin            0
 Body mass index                 0
 Diabetes pedigree function      0
 Age (years)                     0
 Class variable                  0
dtype: int64
'''
#there are no null values

#data set is clean

#checking outliers

sns.boxplot(df[' Number of times pregnant'])
sns.boxplot(df[' Plasma glucose concentration'])
sns.boxplot(df[' Diastolic blood pressure'])
sns.boxplot(df[' Triceps skin fold thickness'])
sns.boxplot(df[' 2-Hour serum insulin'])
sns.boxplot(df[' Body mass index'])
sns.boxplot(df[' Diabetes pedigree function'])
sns.boxplot(df[' Age (years)'])


#all the columns have outliers

#1
iqr = df[' Number of times pregnant'].quantile(0.75)-df[' Number of times pregnant'].quantile(0.25)
iqr
q1=df[' Number of times pregnant'].quantile(0.25)
q3=df[' Number of times pregnant'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df[' Number of times pregnant']=  np.where(df[' Number of times pregnant']>u_limit,u_limit,np.where(df[' Number of times pregnant']<l_limit,l_limit,df[' Number of times pregnant']))
sns.boxplot(df[' Number of times pregnant'])

#2
iqr = df[' Plasma glucose concentration'].quantile(0.75)-df[' Plasma glucose concentration'].quantile(0.25)
iqr
q1=df[' Plasma glucose concentration'].quantile(0.25)
q3=df[' Plasma glucose concentration'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df[' Plasma glucose concentration']=  np.where(df[' Plasma glucose concentration']>u_limit,u_limit,np.where(df[' Plasma glucose concentration']<l_limit,l_limit,df[' Plasma glucose concentration']))
sns.boxplot(df[' Plasma glucose concentration'])


#3
iqr = df[' Diastolic blood pressure'].quantile(0.75)-df[' Diastolic blood pressure'].quantile(0.25)
iqr
q1=df[' Diastolic blood pressure'].quantile(0.25)
q3=df[' Diastolic blood pressure'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df[' Diastolic blood pressure']=  np.where(df[' Diastolic blood pressure']>u_limit,u_limit,np.where(df[' Diastolic blood pressure']<l_limit,l_limit,df[' Diastolic blood pressure']))
sns.boxplot(df[' Diastolic blood pressure'])

#4
iqr = df[' Triceps skin fold thickness'].quantile(0.75)-df[' Triceps skin fold thickness'].quantile(0.25)
iqr
q1=df[' Triceps skin fold thickness'].quantile(0.25)
q3=df[' Triceps skin fold thickness'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df[' Triceps skin fold thickness']=  np.where(df[' Triceps skin fold thickness']>u_limit,u_limit,np.where(df[' Triceps skin fold thickness']<l_limit,l_limit,df[' Triceps skin fold thickness']))
sns.boxplot(df[' Triceps skin fold thickness'])

#5
iqr = df[' 2-Hour serum insulin'].quantile(0.75)-df[' 2-Hour serum insulin'].quantile(0.25)
iqr
q1=df[' 2-Hour serum insulin'].quantile(0.25)
q3=df[' 2-Hour serum insulin'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df[' 2-Hour serum insulin']=  np.where(df[' 2-Hour serum insulin']>u_limit,u_limit,np.where(df[' 2-Hour serum insulin']<l_limit,l_limit,df[' 2-Hour serum insulin']))
sns.boxplot(df[' 2-Hour serum insulin'])

#6
iqr = df[' Body mass index'].quantile(0.75)-df[' Body mass index'].quantile(0.25)
iqr
q1=df[' Body mass index'].quantile(0.25)
q3=df[' Body mass index'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df[' Body mass index']=  np.where(df[' Body mass index']>u_limit,u_limit,np.where(df[' Body mass index']<l_limit,l_limit,df[' Body mass index']))
sns.boxplot(df[' Body mass index'])

#7
iqr = df[' Diabetes pedigree function'].quantile(0.75)-df[' Diabetes pedigree function'].quantile(0.25)
iqr
q1=df[' Diabetes pedigree function'].quantile(0.25)
q3=df[' Diabetes pedigree function'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df[' Diabetes pedigree function']=  np.where(df[' Diabetes pedigree function']>u_limit,u_limit,np.where(df[' Diabetes pedigree function']<l_limit,l_limit,df[' Diabetes pedigree function']))
sns.boxplot(df[' Diabetes pedigree function'])

#8
iqr = df[' Age (years)'].quantile(0.75)-df[' Age (years)'].quantile(0.25)
iqr
q1=df[' Age (years)'].quantile(0.25)
q3=df[' Age (years)'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df[' Age (years)']=  np.where(df[' Age (years)']>u_limit,u_limit,np.where(df[' Age (years)']<l_limit,l_limit,df[' Age (years)']))
sns.boxplot(df[' Age (years)'])

#we need to perform label encoding on last col

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

df[' Class variable'] = label_encoder.fit_transform(df[' Class variable'])


# now data set is ready for further processing
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
