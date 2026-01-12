# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:03:13 2024

@author: Shin
"""

import pandas as pd

df= pd.read_excel("C:/Users/Shin/Documents/Final_project/Data/세종특별자치시(2015~2022)/세종시고등_학교별(도담동).xlsx")
print(df.dtypes)

df.iloc[1] = df.iloc[1].astype(int)
df.iloc[2] = df.iloc[2].astype(int)

merged_row = df.iloc[1] + df.iloc[2]


merged_df = pd.DataFrame([merged_row])

# 수정된 데이터프레임을 새 파일에 저장
merged_df.to_excel("C:/Users/Shin/Documents/Final_project/Data/세종특별자치시(2015~2022)/세종시고등_학교별(도담동_결합).xlsx")
