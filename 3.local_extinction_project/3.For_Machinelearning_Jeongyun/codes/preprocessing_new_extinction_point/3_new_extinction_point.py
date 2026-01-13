# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:45:49 2024

@author: pjc62
"""
import pandas as pd

# Load existing extinction index data
ext_df = pd.read_csv('./preprocessed/기존_소멸위험지수_2015-2023.csv')

# Load log (move-in/move-out) data
log_df = pd.read_csv('./preprocessed/전입전출_log처리_2015-2023.csv')


'''개선 소멸지수 데이터 생성'''
# Specify the 'Administrative district name' column as the index for calculations.
ext_df = ext_df.set_index('행정구역(동읍면)별')
log_df = log_df.set_index('행정구역(시군구)별')

ext_df.info()
log_df.info()

# Uniform ext_df and log_df column names
ext_df.columns = log_df.columns

# Existing extinction risk index^2 + log (moving in/moving out)
new_ep = ext_df ** 2 + log_df

# Convert indexes back to columns
new_ep.reset_index(inplace=True)
new_ep.rename(columns={'index' : '행정구역'}, inplace=True)

# Remove trial data

# cities = 'Seoul Metropolitan City, Busan Metropolitan City, Incheon Metropolitan City, Daegu Metropolitan City, Daejeon Metropolitan City, Gwangju Metropolitan City, Ulsan Metropolitan City, Gyeonggi-do, Chungcheongbuk-do, Chungcheongnam-do, South Jeolla Province, Gyeongsangbuk-do, Gyeongsangnam-do, Gangwon Special Self-Governing Province, Jeonbuk Special Self-Governing Province, Jeju Special Self-Governing Province'.split(', ')
# for city in cities:
# new_ep = new_ep[new_ep['administrative district'] != city]
    
# Heat removal other than 2015-2021
new_ep = new_ep.loc[:,'행정구역':'2021']

# %%
# save file
new_ep.to_csv('./preprocessed/개선_소멸위험지수_2015-2021.csv')
new_ep.to_excel('./preprocessed/개선_소멸위험지수_2015-2021.xlsx')

# %%
'''소멸 위험 등급 나누기'''
# Convert annual data into one data each
df_long = new_ep.melt(id_vars=['행정구역'], var_name='항목', value_name='소멸위험지수')

# Combine administrative districts and items to create a new 'Administrative District' column
df_long['행정구역'] = df_long['행정구역'] + '_' + df_long['항목']

# Remove 'Item' column
df_long = df_long.drop(columns=['항목'])

# All negative values ​​below 0 are treated as 0.
df_long.loc[df_long['소멸위험지수'] < 0,'소멸위험지수'] = 0

# %%
# Switch to Extinction Risk Level
df_long.describe()

# Extinction risk index quartile calculation
quantiles = df_long['소멸위험지수'].quantile([0.25, 0.5, 0.75])

# Definition of extinction risk rating function
def assign_grade(value):
    if value <= quantiles[0.25]:
        return 'A'
    elif value <= quantiles[0.5]:
        return 'B'
    elif value <= quantiles[0.75]:
        return 'C'
    else:
        return 'D'
    
# Add extinction risk level column
df_long['소멸위험등급'] = df_long['소멸위험지수'].apply(assign_grade)

# %%
# Save as file
df_long.to_csv('./preprocessed/소멸위험등급_2015-2021.csv')
df_long.to_excel('./preprocessed/소멸위험등급_2015-2021.xlsx')

# %%
# Extraction of grade D (area at risk of extinction)
d = df_long[df_long['소멸위험등급'] == 'D']
c = df_long[df_long['소멸위험등급'] == 'C']
b = df_long[df_long['소멸위험등급'] == 'B']
a = df_long[df_long['소멸위험등급'] == 'A']

filtered_df = a[a['행정구역'].str.contains('2021')]
filtered_df2 = a[~a['행정구역'].str.contains('2021')]
