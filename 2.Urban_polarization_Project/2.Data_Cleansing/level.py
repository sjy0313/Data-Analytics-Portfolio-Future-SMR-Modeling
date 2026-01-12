# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:18:53 2024

@author: Shin
"""

import pandas as pd

df1= pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/worst_copy.xlsx")
df2= pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/best_copy.xlsx")



numeric_df = df1.select_dtypes(include='number')
seoul_worst = numeric_df.quantile([0.25, 0.50, 0.75, 1.00])

# 결과 출력
print("각 열의 사분위수:")
print(seoul_worst)

# 5등급 
'''
각 열의 사분위수:
      Subway  Primary_School   ...  Supermarket    Park
0.25   326.0            462.0  ...        803.0   897.0  5
0.50   498.0            608.0  ...       1300.0  1700.0  4
0.75   656.0            918.0  ...       1600.0  2300.0  3
1.00  1400.0           1417.0  ...       2200.0  2800.0  2
'''


numeric_df = df2.select_dtypes(include='number')
seoul_best = numeric_df.quantile([0.25, 0.50, 0.75, 1.00])

# 결과 출력
print("각 열의 사분위수:")
print(seoul_best)
'''각 열의 사분위수:
      Subway  Primary_School   ...  Supermarket    Park
0.25   287.0            364.0  ...        426.0   167.0
0.50   413.0            514.0  ...        655.0   398.0
0.75   527.0            752.0  ...       1200.0  1600.0
1.00   981.0           1712.0  ...       2100.0  3000.0'''


combined_df = seoul_worst.add(seoul_best, fill_value=0)

# 각 열의 값을 2로 나누기
rank3 = combined_df / 2

# 결과 출력
print("3등급:")
print(rank3)
'''
      Subway  Primary_School   ...  Supermarket    Park
0.25   306.5            413.0  ...        614.5   532.0
0.50   455.5            561.0  ...        977.5  1049.0
0.75   591.5            835.0  ...       1400.0  1950.0
1.00  1190.5           1564.5  ...       2150.0  2900.0
'''



sb_mean = df1['Subway'].mean()
#%%
'''
Subway               584.00
Primary_School       673.04
Middle_School        737.52
High_School          830.36
General_Hospital    1537.04
Supermarket         1262.88
Park                1593.16
dtype: float64
'''
#%%
#int_means = means.astype(int)
#%%
# 평균
def calculate_column_means(data):
    means = []

    for df in data:
        # 각 데이터프레임의 첫 번째 열을 제외한 나머지 열에 대해 평균 계산
        column_means = df.iloc[:, 1:].mean()
        means.append(column_means)
    
    return means

# 함수 호출하여 각 데이터프레임의 열 평균 계산
data = [df1, df2]
means = calculate_column_means(data)

# 결과 출력
for i, mean in enumerate(means):
    print(f"df{i+1}의 각 열의 평균:")
    print(mean)
    print()

#%%
df3 = pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/worst_copy.xlsx")
df4 = pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/best_copy.xlsx")

#  중앙값
def calculate_column_median(data):
    median = []

    for df in data:
        # 각 데이터프레임의 첫 번째 열을 제외한 나머지 열에 대해 평균 계산
        column_median = df.iloc[:, 1:].median()
        median.append(column_median)
    
    return median

# 함수 호출하여 각 데이터프레임의 열 평균 계산
data = [df3, df4]
median = calculate_column_median(data)

# 결과 출력
for i, median in enumerate(median):
    print(f"df{i+1}의 각 열의 중앙값:")
    print(median)
    
    


#%%
# 서울시 가장 비싼 아파트로부터 인프라까지의 평균거리 : 
rank5 = means[1].astype(int) # 12 ~ 30억
print(rank5)
'''
Subway               425
Primary_School       586
Middle_School        447
High_School          747
General_Hospital    1306
Supermarket          890
Park                 974
dtype: int32'''


# 서울시 5등급 2~5억
rank1 = means[0].astype(int)
print(rank1)
'''
Subway               584
Primary_School       673
Middle_School        737
High_School          830
General_Hospital    1537
Supermarket         1262
Park                1593
dtype: int32'''



























