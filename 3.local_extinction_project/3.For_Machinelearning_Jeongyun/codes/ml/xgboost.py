# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 08:55:49 2024

@author: jcp
"""

import pandas as pd
# data load
df = pd.read_csv("./preprocessed/merged_data.csv", index_col=0)
df2 = pd.read_csv("./preprocessed/merged_data.csv", index_col=0)

# Load the LabelEncoder class.
from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder as an object.
le = LabelEncoder()

# Perform label encoding with fit_transform().
df['소멸위험등급'] = le.fit_transform(df['소멸위험등급'])

# %%


# %%
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
xgb_model=XGBClassifier()

y = df['소멸위험등급']

y.value_counts()

X = df.iloc[:,1:-2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Hyperparameter settings
params = {
    # 'objetive': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate' : 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0}

# model training
num_rounds = 100
lgbm_model = lgb.train(params, train_data, num_rounds)

# Model evaluation
y_pred = lgbm_model.predict(X_test)
y_pred = y_pred
# accuracy = sum(y_pred == y_test) / len(y_test)
# print(f"Accuracy: {accuracy}")