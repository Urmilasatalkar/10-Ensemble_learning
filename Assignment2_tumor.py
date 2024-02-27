# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 23:01:55 2024

@author: rucha
"""

#assignment 2 Ensemble learning

'''
2.	Most cancers form a lump called a tumour. But not all lumps are
 cancerous. Doctors extract a sample from the lump and examine it to
 find out if it’s cancer or not. Lumps that are not cancerous are 
 called benign (be-NINE). Lumps that are cancerous are called malignant 
 (muh-LIG-nunt). Obtaining incorrect results (false positives and 
 false negatives) especially in a medical condition such as cancer
 is dangerous. So, perform Bagging, Boosting, Stacking, and Voting
 algorithms to increase model performance and provide your insights 
 in the documentation.
 
 Business objective
 
 maximize -
 accuracy
 
 minimize - 
 false negatives
 false positives
 
 constrain - 
 Model Interpretability (Ensure that the model is interpretable and transparent )
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/9-ML/Ensemble_learning/Tumor_Ensemble.csv')
df

df.columns

'''
Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'points_mean', 'symmetry_mean', 'dimension_mean', 'radius_se',
       'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'points_se', 'symmetry_se',
       'dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
       'area_worst', 'smoothness_worst', 'compactness_worst',
       'concavity_worst', 'points_worst', 'symmetry_worst', 'dimension_worst'],
      dtype='object')
'''

#data dictionary
df.dtypes
'''
id                     int64    quantitative    irrelevent
diagnosis             object    nominal         relevent
radius_mean          float64    quantitative    relevent
texture_mean         float64    quantitative    relevent
perimeter_mean       float64    quantitative    relevent
area_mean            float64    quantitative    relevent
smoothness_mean      float64    quantitative    relevent
compactness_mean     float64    quantitative    relevent
concavity_mean       float64    quantitative    relevent
points_mean          float64    quantitative    relevent
symmetry_mean        float64    quantitative    relevent
dimension_mean       float64    quantitative    relevent
radius_se            float64    quantitative    relevent
texture_se           float64    quantitative    relevent
perimeter_se         float64    quantitative    relevent
area_se              float64    quantitative    relevent
smoothness_se        float64    quantitative    relevent
compactness_se       float64    quantitative    relevent
concavity_se         float64    quantitative    relevent
points_se            float64    quantitative    relevent
symmetry_se          float64    quantitative    relevent
dimension_se         float64    quantitative    relevent
radius_worst         float64    quantitative    relevent
texture_worst        float64    quantitative    relevent
perimeter_worst      float64    quantitative    relevent
area_worst           float64    quantitative    relevent
smoothness_worst     float64    quantitative    relevent
compactness_worst    float64    quantitative    relevent
concavity_worst      float64    quantitative    relevent
points_worst         float64    quantitative    relevent
symmetry_worst       float64    quantitative    relevent
dimension_worst      float64    quantitative    relevent
dtype: object

'''

df.describe()
'''
                id  radius_mean  ...  symmetry_worst  dimension_worst
count  5.690000e+02   569.000000  ...      569.000000       569.000000
mean   3.037183e+07    14.127292  ...        0.290076         0.083946
std    1.250206e+08     3.524049  ...        0.061867         0.018061
min    8.670000e+03     6.981000  ...        0.156500         0.055040
25%    8.692180e+05    11.700000  ...        0.250400         0.071460
50%    9.060240e+05    13.370000  ...        0.282200         0.080040
75%    8.813129e+06    15.780000  ...        0.317900         0.092080
max    9.113205e+08    28.110000  ...        0.663800         0.207500

[8 rows x 31 columns]

'''
print(df.isnull().sum())

'''
id                   0
diagnosis            0
radius_mean          0
texture_mean         0
perimeter_mean       0
area_mean            0
smoothness_mean      0
compactness_mean     0
concavity_mean       0
points_mean          0
symmetry_mean        0
dimension_mean       0
radius_se            0
texture_se           0
perimeter_se         0
area_se              0
smoothness_se        0
compactness_se       0
concavity_se         0
points_se            0
symmetry_se          0
dimension_se         0
radius_worst         0
texture_worst        0
perimeter_worst      0
area_worst           0
smoothness_worst     0
compactness_worst    0
concavity_worst      0
points_worst         0
symmetry_worst       0
dimension_worst      0
dtype: int64

'''

#there are no null values

##data set is clean

#checking outliers

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Identify outliers using the IQR method
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any()

# Get the columns with outliers
columns_with_outliers = outliers[outliers].index.tolist()
print("Columns with outliers:", columns_with_outliers)

'''
Columns with outliers: ['area_mean', 'area_se', 'area_worst',
'compactness_mean', 'compactness_se', 'compactness_worst', 
'concavity_mean', 'concavity_se', 'concavity_worst', 
'dimension_mean', 'dimension_se', 'dimension_worst', 'id', 
'perimeter_mean', 'perimeter_se', 'perimeter_worst', 
'points_mean', 'points_se', 'radius_mean', 'radius_se', 
'radius_worst', 'smoothness_mean', 'smoothness_se', 
'smoothness_worst', 'symmetry_mean', 'symmetry_se', 
'symmetry_worst', 'texture_mean', 'texture_se', 'texture_worst']

'''
#1
iqr = df['area_mean'].quantile(0.75)-df['area_mean'].quantile(0.25)
iqr
q1=df['area_mean'].quantile(0.25)
q3=df['area_mean'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df['area_mean']=  np.where(df['area_mean']>u_limit,u_limit,np.where(df['area_mean']<l_limit,l_limit,df['area_mean']))
sns.boxplot(df['area_mean'])

#2
iqr = df['area_se'].quantile(0.75)-df['area_se'].quantile(0.25)
iqr
q1=df['area_se'].quantile(0.25)
q3=df['area_se'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df['area_se']=  np.where(df['area_se']>u_limit,u_limit,np.where(df['area_se']<l_limit,l_limit,df['area_se']))
sns.boxplot(df['area_se'])

#3
iqr = df['area_worst'].quantile(0.75)-df['area_worst'].quantile(0.25)
iqr
q1=df['area_worst'].quantile(0.25)
q3=df['area_worst'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df['area_worst']=  np.where(df['area_worst']>u_limit,u_limit,np.where(df['area_worst']<l_limit,l_limit,df['area_worst']))
sns.boxplot(df['area_worst'])


#4
iqr = df['compactness_mean'].quantile(0.75)-df['compactness_mean'].quantile(0.25)
iqr
q1=df['compactness_mean'].quantile(0.25)
q3=df['compactness_mean'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df['compactness_mean']=  np.where(df['compactness_mean']>u_limit,u_limit,np.where(df['compactness_mean']<l_limit,l_limit,df['compactness_mean']))
sns.boxplot(df['compactness_mean'])

#5
iqr = df['compactness_se'].quantile(0.75)-df['compactness_se'].quantile(0.25)
iqr
q1=df['compactness_se'].quantile(0.25)
q3=df['compactness_se'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df['compactness_se']=  np.where(df['compactness_se']>u_limit,u_limit,np.where(df['compactness_se']<l_limit,l_limit,df['compactness_se']))
sns.boxplot(df['compactness_se'])

#6
iqr = df['compactness_worst'].quantile(0.75)-df['compactness_worst'].quantile(0.25)
iqr
q1=df['compactness_worst'].quantile(0.25)
q3=df['compactness_worst'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df['compactness_worst']=  np.where(df['compactness_worst']>u_limit,u_limit,np.where(df['compactness_worst']<l_limit,l_limit,df['compactness_worst']))
sns.boxplot(df['compactness_worst'])

#7
iqr = df['compactness_worst'].quantile(0.75)-df['compactness_worst'].quantile(0.25)
iqr
q1=df['compactness_worst'].quantile(0.25)
q3=df['compactness_worst'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df['compactness_worst']=  np.where(df['compactness_worst']>u_limit,u_limit,np.where(df['compactness_worst']<l_limit,l_limit,df['compactness_worst']))
sns.boxplot(df['compactness_worst'])

#8
iqr = df['concavity_mean'].quantile(0.75) - df['concavity_mean'].quantile(0.25)
q1 = df['concavity_mean'].quantile(0.25)
q3 = df['concavity_mean'].quantile(0.75)

l_limit = q1 - 1.5 * iqr
u_limit = q3 + 1.5 * iqr

df['concavity_mean'] = np.where(df['concavity_mean'] > u_limit, u_limit,
                                np.where(df['concavity_mean'] < l_limit, l_limit, df['concavity_mean']))
sns.boxplot(df['concavity_mean'])

#9
iqr = df['concavity_se'].quantile(0.75) - df['concavity_se'].quantile(0.25)
q1 = df['concavity_se'].quantile(0.25)
q3 = df['concavity_se'].quantile(0.75)

l_limit = q1 - 1.5 * iqr
u_limit = q3 + 1.5 * iqr

df['concavity_se'] = np.where(df['concavity_se'] > u_limit, u_limit,
                              np.where(df['concavity_se'] < l_limit, l_limit, df['concavity_se']))
sns.boxplot(df['concavity_se'])


# Function to treat outliers for a specific column
def treat_outliers(column_name):
    iqr = df[column_name].quantile(0.75) - df[column_name].quantile(0.25)
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)

    l_limit = q1 - 1.5 * iqr
    u_limit = q3 + 1.5 * iqr
    df[column_name] = np.where(df[column_name] > u_limit, u_limit,
                               np.where(df[column_name] < l_limit, l_limit, df[column_name]))
    
    # Plot boxplot after treating outliers
    sns.boxplot(df[column_name])

# Treat outliers for each specified column
columns_with_outliers = ['area_mean', 'area_se', 'area_worst',
                         'compactness_mean', 'compactness_se', 'compactness_worst', 
                         'concavity_mean', 'concavity_se', 'concavity_worst', 
                         'dimension_mean', 'dimension_se', 'dimension_worst', 'id', 
                         'perimeter_mean', 'perimeter_se', 'perimeter_worst', 
                         'points_mean', 'points_se', 'radius_mean', 'radius_se', 
                         'radius_worst', 'smoothness_mean', 'smoothness_se', 
                         'smoothness_worst', 'symmetry_mean', 'symmetry_se', 
                         'symmetry_worst', 'texture_mean', 'texture_se', 'texture_worst']

for col in columns_with_outliers:
    treat_outliers(col)
    
#label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

    

#now dataset is ready for further treatment
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


x = df.iloc[:,0:]
x.columns
x = x.drop(x['diagnosis'])
y = df.iloc[:,1]
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



