# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:50:34 2024

@author: jcp
"""

# Sample model to be mounted

import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv("./preprocessed/merged_data.csv", index_col=0)

# Import the LabelEncoder class.
from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder as an object.
le = LabelEncoder()

# Perform label encoding with fit_transform().
df['소멸위험등급'] = le.fit_transform(df['소멸위험등급'])

# Additional preprocessing
df = df.replace("Ⅹ",0)
# Convert Object type columns to numeric
cols = df.columns[1:]
for col in df.columns[1:]:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col].str.replace('[\s,]', '', regex=True), errors='raise')
        
# %%
'''스케일링'''
# Robust scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled = scaler.fit_transform(df.iloc[:, 1:-2])
df.iloc[:, 1:-2] = pd.DataFrame(scaled, columns = df.iloc[:, 1:-2].columns)

# %%

# Separate 2021 data into test data
test = df[df['행정구역'].str.contains('_2021')]

train = df[~df['행정구역'].str.contains('_2021')]

X = train.iloc[:, 1:-2]
y = train['소멸위험등급']

# %%


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)


# %%

# LightGBM
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

# LightGBM model initialization
lgb_clf = lgb.LGBMClassifier(random_state=1)
'''
num_leaves: 트리의 잎사귀 수.
learning_rate: 학습률.
n_estimators: 트리의 수.
max_depth: 트리의 최대 깊이.
min_child_samples: 리프 노드의 최소 샘플 수.
subsample: 데이터의 일부를 샘플링할 비율.
colsample_bytree: 각 트리를 학습할 때 사용하는 특성의 비율.'''
# Hyperparameter space definition
param_dist_lgb = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [-1, 10, 20],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Perform RandomizedSearchCV
rand_search_lgb = RandomizedSearchCV(lgb_clf, param_distributions=param_dist_lgb, n_iter=50, cv=5, scoring='accuracy', random_state=1)
rand_search_lgb.fit(X_train, y_train.values.ravel())  # y_train.values.ravel() converts y_train to a one-dimensional array.

print('Best parameters:', rand_search_lgb.best_params_)
print('Best score:', round(rand_search_lgb.best_score_, 4))



# Evaluated as the optimal model
best_lgb_model = rand_search_lgb.best_estimator_
y_pred_lgb = best_lgb_model.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")

# Output confusion matrix and classification report of prediction results (optional)
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_test, y_pred_lgb)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred_lgb)
print("Classification Report:")
print(class_report)

# %%
'''2021 등급 예측'''
# 2021
X_2021 = test.iloc[:,1:-2]
y_2021 = test['소멸위험등급']


y_pred = best_lgb_model.predict(X_2021)

print(f"LightGBM Accuracy: {accuracy_score(y_2021, y_pred):.4f}")

# %%
import joblib

joblib.dump(best_lgb_model, 'lgb.pkl')

# gbm_pickle = joblib.load('lgb.pkl')