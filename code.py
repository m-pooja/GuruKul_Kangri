import numpy as np
from math import sqrt
x=8
y=sqrt(x)
y
import pandas as pd
import sklearn as sk
data=pd.read_excel("titanic3.xls")

data.head()
data.head(20)
data.columns
data.cabin
data['cabin'][:30]
dcab=data['cabin'][:30]

dcab.dtypes()
type(dcab)
type(data)
data.cabin.isnull()
data.cabin.isnull().sum()
data.cabin.size
data.size
data.shape
data.isnull()
data.isnull().sum()
data.cabin.dropna()
data.cabin
data.cabin.dropna(inplace=True)
data.cabin
data.dropna(inplace=True)
data=pd.read_excel("titanic3.xls")

data.cabin.fillna()

data.cabin.fillna(df.cabin.mean(), inplace=True)
data.cabin.fillna(data.cabin.mean(), inplace=True)
data.age.fillna(data.age.mean(), inplace=True)

data.age.hust()
data.age.hist()
data.cabin.hist()
data.isnull.sum()
data.isnull().sum()
data.boat.hist()
data.isnull().sum()
data.body.hist()
from pandas.tools.plotting import scatter_matrix
pd.scatter_matrix(data)
data2=data[:4][:]
pd.scatter_matrix(data2)
data2=data[:][:4]
pd.scatter_matrix(data2)

import matplotlib.pyplot as plt
plt.scatter(data.age, data.survived)
plt.scatter(data.age[data.age>60], data.survived)

plt.scatter(data.age[data.age>60], data.survived[data.age>60])
plt.boxplot(data.age)
data.age.max()
data.age.idxmax()
data.describe()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
scdata= sc_X.fit_transform(data)
scdata= sc_X.fit_transform(data.age)

scdata
from sklearn.preprocessing import Binarizer

from sklearn.preprocessing import Binarizer
bind = Binarizer()
gender_b=bind(data.sex)
gender_b=bind.fit_transform(data.sex)

gender_b=bind.fit(data.sex)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
d=l.fit(data.sex)

dd=l.transform(d)

d
dd
dd=l.transform(data.sex)

dd
data.sex=l.transform(data.sex)

target=data.survived
del(data.survived)

data.columns
features= data['pclass',  'name', 'sex', 'age', 'sibsp', 'parch', 'ticket',
       'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest']

features= data['pclass',  'name', 'sex', 'age', 'sibsp', 'parch', 'ticket','fare', 'cabin', 'embarked', 'boat', 'body']

from sklearn.cross_validation import train_test_split

features= data[['pclass',  
'name', 'sex', 'age', 'sibsp', 
'parch', 'ticket','fare', 
'cabin', 'embarked', 'boat', 
'body']]
X_train,X_test,Y_train,Y_test=train_test_split(features,target)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

lr.fit(X_train,Y_train)

del(X_train['name'])
lr.fit(X_train,Y_train)

lr.fit(X_train[['age','pclass','sex']],Y_train)

y_pred=lr.predict(X_test)

y_pred=lr.predict(X_test[['age','pclass','sex']])

from sklearn.metrics import accuracy_score,

from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_scoreclassification_report

from sklearn.metrics import classification_report
print(confusion_matrix(Y_test,y_pred))

print(accuracy_score(Y_test,y_pred))#50%

print(classification_report(Y_test,y_pred))
