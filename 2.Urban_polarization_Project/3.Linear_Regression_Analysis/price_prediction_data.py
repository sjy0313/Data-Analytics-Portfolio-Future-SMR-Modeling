# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:36:29 2024

@author: Shin
"""

import numpy as np
base_price = 25000  # 250 million won

# Calculate the value that becomes cheaper by 3%
decreased_prices = []
for i in range(1, 15):
    decreased_price = base_price * (1 - 0.03 * i)
    decreased_prices.append(decreased_price)

# Result output
for i, price in enumerate(decreased_prices):
    print(f"{i+1}번째 값: {price:.2f} 원")




# given data
areas = np.array([25] * 29)  # 29 pieces of 25 pyeong data
prices = np.array([25000, 25500, 24750, 25750, 25250, 24500, 24250, 26000, 26250, 26500,
                   23000, 23250, 23500, 23750, 22500, 22750, 25500, 25750, 26000, 26250,
                   26500, 26750, 27000, 27250, 27500, 27750, 28000, 28250, 28500])

# Setting price standards per pyeong
price_per_area = 1000  # 10 million won per pyeong

# Set rise/fall rate for forecasting
decrease_percentage = 0.03  # 3% cheaper
increase_percentage = 0.05  # 5% more expensive

# Calculate forecasts
predicted_prices = []

for i in range(len(prices)):
    if prices[i] < 25000:  # If less than 250 million
        predicted_price = prices[i] * (1 - decrease_percentage)
    elif prices[i] >= 25000:  # In case of more than 250 million won
        predicted_price = prices[i] * (1 + increase_percentage)
    
    predicted_prices.append(predicted_price)

# Result output
for i in range(len(predicted_prices)):
    print(f"기존 가격: {prices[i]}, 예측 가격: {predicted_prices[i]}")

