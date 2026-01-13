# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:03:07 2024

@author: Shin
"""

#%%
# Gridsearch
# It takes too long because it checks the generalization performance for all hyperparameter candidates.

# Randomsearch
# It takes less time than GridSearch, but since it literally selects a few “random” items and checks them, accuracy may be somewhat lower.

# Bayesian Optimization
# It is more likely to find better hyperparameters in fewer attempts than random or grid search.

# LightGBM with bayseian optimizer
import lightgbm as lgbm
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate
import pandas as pd
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data loading and preprocessing
t = pd.read_csv('C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/2015_소멸등급.csv')
d = pd.read_excel("C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/교육/EXCEL/교육_2015_전국.xlsx")

X =  d[['교원_1인당_학생수_유치원', '교원_1인당_학생수_초등학교', '교원_1인당_학생수_중학교', '교원_1인당_학생수_고등학교',
        '유치원_학급당 학생 수 (명)', '초등학교_학급당 학생 수 (명)', '중학교_학급당 학생 수 (명)','고등학교_학급당 학생 수 (명)',
        '학교교과 교습학원 (개)', '평생직업 교육학원 (개)', '사설학원당 학생수 (명)','유치원생 수', '초등학생 수']]

y = t['2015_등급'] - 1  # Adjust class labels to be [0, 1, 2, 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)


# Create objective function
def lgbm_cv(learning_rate, num_leaves, max_depth, min_child_weight, colsample_bytree, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
    model = lgbm.LGBMClassifier(learning_rate=learning_rate,
                                n_estimators = 300,
                                #boosting = 'dart',
                                num_leaves = int(round(num_leaves)),
                                max_depth = int(round(max_depth)),
                                min_child_weight = int(round(min_child_weight)),
                                colsample_bytree = colsample_bytree,
                                feature_fraction = max(min(feature_fraction, 1), 0),
                                bagging_fraction = max(min(bagging_fraction, 1), 0),
                                lambda_l1 = max(lambda_l1, 0),
                                lambda_l2 = max(lambda_l2, 0)
                               )
    scoring = {'roc_auc_score': make_scorer(roc_auc_score)}
    result = cross_validate(model, X, y, cv=5, scoring=scoring)
    auc_score = result["test_roc_auc_score"].mean()
    return auc_score
# Search target section of input value
pbounds = {'learning_rate' : (0.0001, 0.05),
           'num_leaves': (300, 600),
           'max_depth': (2, 25),
           'min_child_weight': (30, 100),
           'colsample_bytree': (0, 0.99),
           'feature_fraction': (0.0001, 0.99),
           'bagging_fraction': (0.0001, 0.99),
           'lambda_l1' : (0, 0.99),
           'lambda_l2' : (0, 0.99),
          }

'''learning_rate : 보통 0.01~ 정도로 설정합니다. 세부 조정을 위해서는 0.0001~정도로 설정해도 무방합니다.

num_leaves : 250정도로 설정해도 무방합니다. 300~600 정도로 설정했습니다.

max_depth : -1 로 설정하면 무한대로 트리가 길어집니다. 9~ 정도로 설정하는게 무방하나 조금 더 넓은 범위로 설정했습니다.

feature_fraction, bagging_fraction : 0과 1 사이의 범위로 설정했습니다.'''
# object creation
lgbmBO = BayesianOptimization(f = lgbm_cv, pbounds = pbounds, verbose = 2, random_state = 0 )
lgbmBO.maximize(init_points=5, n_iter = 20, acq='ei', xi=0.01)
lgbmBO.max

# Apply parameters
fit_lgbm = lgbm.LGBMClassifier(learning_rate=lgbmBO.max['params']['learning_rate'],
                               num_leaves = int(round(lgbmBO.max['params']['num_leaves'])),
                               max_depth = int(round(lgbmBO.max['params']['max_depth'])),
                               min_child_weight = int(round(lgbmBO.max['params']['min_child_weight'])),
                               colsample_bytree=lgbmBO.max['params']['colsample_bytree'],
                               feature_fraction = max(min(lgbmBO.max['params']['feature_fraction'], 1), 0),
                               bagging_fraction = max(min(lgbmBO.max['params']['bagging_fraction'], 1), 0),
                               lambda_l1 = lgbmBO.max['params']['lambda_l1'],
                               lambda_l2 = lgbmBO.max['params']['lambda_l2']
                               )
model = fit_lgbm.fit(X,y)

import joblib
joblib.dump(model, 'lgbmBO_201006.pkl')

pred_y = model.predict(y_test)
#%%

# XGBoost model
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Bayesian optimization
param_space = {
    'learning_rate': (0.01, 1.0, 'log-uniform'),
    'max_depth': (3, 10),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'gamma': (0, 0.5),
    'n_estimators': (50, 300)
}

bayes_search = BayesSearchCV(
    estimator=xgb_clf,
    search_spaces=param_space,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Perform optimization
bayes_search.fit(X_train, y_train)

# Optimal hyperparameter output
print('Best parameters:', bayes_search.best_params_)
print('Best cross-validation score:', bayes_search.best_score_)

# Performance evaluation on test data
y_pred = bayes_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('Test Accuracy:', accuracy)
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

'''Best parameters: OrderedDict([('colsample_bytree', 0.5), ('gamma', 0.03349959227575049), ('learning_rate', 0.027511984523544558), ('max_depth', 9), ('min_child_weight', 10), ('n_estimators', 100), ('subsample', 0.622218434807881)])
Best cross-validation score: 0.39159663865546224
Test Accuracy: 0.3793103448275862
Confusion Matrix:
[[ 0  3  0  5]
 [ 1  2  0 12]
 [ 1  2  2  8]
 [ 0  4  0 18]]
Classification Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         8
           1       0.18      0.13      0.15        15
           2       1.00      0.15      0.27        13
           3       0.42      0.82      0.55        22

    accuracy                           0.38        58
   macro avg       0.40      0.28      0.24        58
weighted avg       0.43      0.38      0.31        58
'''