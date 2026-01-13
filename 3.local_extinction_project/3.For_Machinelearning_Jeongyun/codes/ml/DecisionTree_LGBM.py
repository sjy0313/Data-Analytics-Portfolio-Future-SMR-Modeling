# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:04:35 2024

@author: jcp
"""

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

y = df['소멸위험등급']
X = df.iloc[:,1:-2]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
# Reasons for using RandomizedSearchCV
# Unlike GridSearchCV, RandomizedSearchCV does not try all combinations of the hyperparameter space;
# Randomly selects a user-specified number (n_iter) of combinations. This allows you to explore a variety of settings while saving computation time.
# Randomly select combinations within the range of hyperparameters and test them. It is faster than grid search, but the probability of finding the optimal combination may be lower.

#%%
# Decision Tree DecisionTree
'''
criterion : 분할 성능 측정 기능

min_samples_split : 노드를 분할하기 위한 최소한의 샘플 데이터수로, 과적합을 제어하는데 주로 사용함.
작게 설정할 수록 분할 노드가 많아져 과적합 가능성이 높아짐

max_depth : 트리의 최대 깊이, 깊이가 깊어지면 과적합될 수 있음.

max_features : 최적의 분할을 위해 고려할 최대 feature 개수
(default = None : 데이터 세트의 모든 피처를 사용)

samples_leaf : 리프노드가 되기 위해 필요한 최소한의 샘플 데이터수 (과적합 제어 용도), 작게 설정 필요

max_leaf_nodes : 리프노드의 최대 개수

param_distributions : 튜닝을 위한 대상 파라미터, 사용될 파라미터를 딕셔너리 형태로 넣어준다.

n_iter : 파라미터 검색 횟수

best score: 최고 평균 정확도 수치
'''
# Find the optimal parameter combination
dt_clf = DecisionTreeClassifier(random_state=1)

param_dist = {
    'criterion':['gini','entropy'], 
    'max_depth':[None,2,3,4,5,6], 
    'max_leaf_nodes':[None,2,3,4,5,6,7], 
    'min_samples_split':[2,3,4,5,6], 
    'min_samples_leaf':[1,2,3], 
    #'max_features':[None,'sqrt','log2',3,4,5]
    }

rand_search = RandomizedSearchCV(dt_clf, param_distributions = param_dist, n_iter = 50, cv = 5, scoring = 'accuracy', refit=True)
rand_search.fit(X_train, y_train)

print('best parameters : ', rand_search.best_params_)

print('best score : ', round(rand_search.best_score_, 4))

result = pd.DataFrame(rand_search.cv_results_)

#%%
from sklearn.tree import DecisionTreeClassifier

# Initialize model with optimal parameters
best_params = rand_search.best_params_
best_model = DecisionTreeClassifier(
    criterion=best_params['criterion'],
    max_depth=best_params['max_depth'],
    max_leaf_nodes=best_params['max_leaf_nodes'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=1
)
# model training
best_model.fit(X_train, y_train)
#%%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predictions on test data
y_pred = best_model.predict(X_test)

# Model performance evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

#%%
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



#%%
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Initialize LightGBM model using optimal hyperparameters
best_params = rand_search_lgb.best_params_
lgb_clf_best = lgb.LGBMClassifier(
    num_leaves=best_params['num_leaves'],
    learning_rate=best_params['learning_rate'],
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_child_samples=best_params['min_child_samples'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    random_state=1
)

# Train model with optimal parameters
lgb_clf_best.fit(X_train, y_train.values.ravel())

# Make predictions on test data
y_pred_lgb = lgb_clf_best.predict(X_test)

# Accuracy evaluation
accuracy = accuracy_score(y_test, y_pred_lgb)
print(f"LightGBM Accuracy with optimized parameters: {accuracy:.4f}")

# Output confusion matrix and classification report of prediction results (optional)
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_test, y_pred_lgb)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred_lgb)
print("Classification Report:")
print(class_report)



#%%
#XGboost
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


# data partitioning
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

# XGBoost model settings
xgb_clf = xgb.XGBClassifier(random_state=1)

# Hyperparameter grid settings
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4]
}

# Perform RandomizedSearchCV
rand_search = RandomizedSearchCV(xgb_clf, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=1)
rand_search.fit(X_train, y_train)

# Check optimal hyperparameters and performance
print('Best parameters:', rand_search.best_params_)
print('Best score:', round(rand_search.best_score_, 4))

# Make and evaluate predictions with optimal models
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test)
print('Test Accuracy:', accuracy_score(y_test, y_pred))
'''
Best parameters: {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.3, 'gamma': 0.4, 'colsample_bytree': 0.6}
Best score: 0.3629
Test Accuracy: 0.46551724137931033'''
#%%
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create an XGBoost model with optimal hyperparameters
best_params = {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.3, 'gamma': 0.4, 'colsample_bytree': 0.6}
xgb_clf_optimized = xgb.XGBClassifier(**best_params, random_state=1)

# Learn with the optimal model
xgb_clf_optimized.fit(X_train, y_train)

# perform predictions
y_pred = xgb_clf_optimized.predict(X_test)

# Accuracy evaluation
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy:', accuracy)

# Confusion matrix and classification report output
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('Classification Report:')
print(classification_report(y_test, y_pred))