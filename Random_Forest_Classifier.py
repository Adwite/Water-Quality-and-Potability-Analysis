import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import statsmodels
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_column",None)
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

os.chdir(r'D:\Data Science\Capstone Project')
df=pd.read_csv('water_potability.csv')
data=pd.DataFrame(df)
data.head()
data.shape

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
data.shape

data.isnull().sum()

data['ph'].fillna(data['ph'].mean(),inplace=True)
data['Sulfate'].fillna(data['ph'].mean(),inplace=True)
data['Trihalomethanes'].fillna(data['ph'].mean(),inplace=True)

data.isnull().sum()

data_copy=data.copy(deep=True)
col=['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
features=data_copy[col]
target=data_copy.drop(col,axis=1)

features_scaled=StandardScaler().fit_transform(features)
features_scaled

cols=['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
features_updated=pd.DataFrame(features_scaled,columns=cols,index=features.index)
features_updated.head()

x=pd.DataFrame(features_updated)
x.shape

y=pd.DataFrame(target)
y.shape


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 10)
print("X_train",x_train.shape)
print("y_train",y_train.shape)
print("X_test",x_test.shape)
print("y_test",y_test.shape)

model=RandomForestClassifier(n_estimators=100,random_state=10)
model.fit(x_train,y_train)

print(model.score(x_train,y_train))
prediction_test=model.predict(x_test)
print(y_test,prediction_test)

import pickle
pickle.dump(model,open('model.pkl','wb'))


model=pickle.load(open('model.pkl','rb'))
print(model.predict([[8.832142,176.808661,12171.024549,7.475336,321.257628,395.387337,12.481552,65.867189,3.914512]]))
