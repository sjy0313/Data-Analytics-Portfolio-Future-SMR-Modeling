# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:26:06 2024

@author: jcp
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:01:50 2024

@author: pjc62
"""

import pandas as pd
file = './data/매매_실거래가격(비교)_수정_v11.xlsx'
# df = pd.read_excel(data)
# df = pd.read_excel(data,parse_dates=['시점'])
df = pd.read_excel(file,index_col='연도_월')


# %%
df.index

# Change date column from float type to str type + handle October
df.index = df.index.astype(str).map(lambda x: x + '0' if len(x) < 7 else x)
df.columns
data = df.iloc[:,1:]

target = df.loc[:,['실거래가격지수']]

# %%
'''
상관관계분석
'''
# correlation matrix
df_corr = df.corr()

# %%
def reg():
    import matplotlib.font_manager as fm
    import os

    user_name = os.getlogin()

    fontpath = [f'C:/Users/{user_name}/AppData/Local/Microsoft/Windows/Fonts']
    font_files = fm.findSystemFonts(fontpaths=fontpath)
    for fpath in font_files:
        
        fm.fontManager.addfont(fpath)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
reg()
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['figure.dpi'] = 140


# %%

# plt.figure(figsize=(20,20))
# sns.set(font_scale=0.6)
# sns.heatmap(df_corr, annot=True, cbar=False)
# plt.show

# %%
# Correlation analysis between variables
 # Sorted in order of highest correlation with the actual transaction price index.
 # Ideal correlation coefficient: 0.3~0.7
corr_order = df.corr().loc['지지율':,'실거래가격지수'].abs().sort_values(ascending=False)
corr_order
# Comprehensive real estate tax_tax rate_individual 0.897827
# Consumer Price Index 0.820183
# PIR index_nationwide 0.628749
# Approval rating 0.342047
# Deposit collateral loan 0.308376
# Apartment sales transaction volume 0.287109
# Home mortgage loan 0.185357
# Trading supply and demand trend 0.165482
# Base interest rate 0.150050
# Name: Real transaction price index, dtype: float64

# %%
# Real transaction price index distribution
sns.displot(x='실거래가격지수',kind='hist', data=df)
plt.show()
    # Confirm that it is left-biased
    
# %%
'''
피처 스케일링
'''
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

scaler.fit(data)
df_scaled = scaler.transform(data)

df.iloc[:, 1:] = df_scaled[:,:]
df.head()

# %%
'''주성분 분석'''
from sklearn.decomposition import PCA

pca = PCA(random_state=1004)
pca.fit_transform(df.iloc[:, 1:])

print(pca.explained_variance_ratio_)
explained_variance = pca.explained_variance_ratio_

plt.rcParams['figure.figsize'] = (15, 7)
plt.plot(range(1, df.iloc[:,1:].shape[1]+1), pca.explained_variance_ratio_)
plt.xlabel("number of Principal Components", fontsize=12)
plt.ylabel("% of Variance Explained", fontsize=12)
plt.show()

# %%
pca = PCA(n_components=7, random_state=1004)
df_pca = pca.fit_transform(df.iloc[:, 1:])


# %%
from sklearn.model_selection import train_test_split
X_data = df_pca
y_data = df.loc[:, '실거래가격지수']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# (105, 2) (105,)
# (27, 2) (27,)
# %%

'''
베이스라인 모델 - 선형회귀
'''
from sklearn.linear_model import LinearRegression
import numpy as np
lr = LinearRegression()
lr.fit(X_train, y_train)

print('결정계수:',lr.score(X_test,y_test))
print('회귀계수:',np.round(lr.coef_, 1))
print('상수항:',np.round(lr.intercept_, 1))

# %%
# prediction
y_test_pred = lr.predict(X_test)

# Distribution of predicted and actual values
plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(y_test, label='실제값')
ax2 = sns.kdeplot(y_test_pred, label='예측값', ax=ax1)
plt.legend()
plt.show()

# %%
'''
모델 성능 평가
'''
# evaluation
from sklearn.metrics import mean_squared_error
y_train_pred = lr.predict(X_train)

train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f" % train_mse)

test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f" % test_mse)

# Train MSE:2.1746
# Test MsE:4.4011
# The smaller the model, the better the model performance.

# %%
# K-Fold cross-validation
from sklearn.model_selection import cross_val_score
lr = LinearRegression()
mse_scores = -1*cross_val_score(lr, X_train, y_train, cv=20,
                                scoring='neg_mean_squared_error')
print("개별 Fold MSE:", np.round(mse_scores, 4))
print("평균 MSE:%.4f" % np.mean(mse_scores))

for score in mse_scores:
    if score > 30:
        print('개선필요')
# Individual Fold MSE: [7.7913 2.0703 6.6352 7.921 1.626 ]
# Average MSE:5.2088

