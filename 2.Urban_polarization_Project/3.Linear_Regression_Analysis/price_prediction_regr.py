# -*- coding: utf-8 -*-
"""

Created on Mon Jun 17 15:56:49 2024

@author: Shin
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 거리 및 가격 데이터 생성 (수정된 데이터)
data = {
    'distance_to_school': [500, 1000, 1500, 2000, 2500],
    'area': [82, 82, 82, 82, 82],
    'distance to hospital': [1000, 2000, 3000, 4000, 5000],
    'distance to park': [500, 1000, 1500, 2000, 2500],
    'price': [50000, 48000, 46000, 44000, 42000]  # 선형 관계로 수정된 가격
}

df = pd.DataFrame(data)

X = df[['distance_to_school', 'area', 'distance to hospital', 'distance to park']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 예측데이터
new_data = pd.DataFrame({
    'distance_to_school': [1500],
    'area': [82],
    'distance to hospital': [1000],
    'distance to park': [1200]
})

# 데이터 스케일링
new_data_scaled = scaler.transform(new_data)

# 예측 수행
predicted_price = model.predict(new_data_scaled)
print(f"Predicted Price for new data: {predicted_price[0]} 만원")
