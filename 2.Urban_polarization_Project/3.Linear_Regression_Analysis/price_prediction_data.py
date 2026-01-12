# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:36:29 2024

@author: Shin
"""

import numpy as np
base_price = 25000  # 2억5천만 원

# 3%씩 저렴해지는 값 계산
decreased_prices = []
for i in range(1, 15):
    decreased_price = base_price * (1 - 0.03 * i)
    decreased_prices.append(decreased_price)

# 결과 출력
for i, price in enumerate(decreased_prices):
    print(f"{i+1}번째 값: {price:.2f} 원")




# 주어진 데이터
areas = np.array([25] * 29)  # 25평 데이터 29개
prices = np.array([25000, 25500, 24750, 25750, 25250, 24500, 24250, 26000, 26250, 26500,
                   23000, 23250, 23500, 23750, 22500, 22750, 25500, 25750, 26000, 26250,
                   26500, 26750, 27000, 27250, 27500, 27750, 28000, 28250, 28500])

# 평당 가격 기준 설정
price_per_area = 1000  # 1평당 1000만원

# 예측을 위한 상승/하락 비율 설정
decrease_percentage = 0.03  # 3% 저렴
increase_percentage = 0.05  # 5% 비싸게

# 예측값 계산
predicted_prices = []

for i in range(len(prices)):
    if prices[i] < 25000:  # 2억5천보다 작은 경우
        predicted_price = prices[i] * (1 - decrease_percentage)
    elif prices[i] >= 25000:  # 2.5억 이상인 경우
        predicted_price = prices[i] * (1 + increase_percentage)
    
    predicted_prices.append(predicted_price)

# 결과 출력
for i in range(len(predicted_prices)):
    print(f"기존 가격: {prices[i]}, 예측 가격: {predicted_prices[i]}")

