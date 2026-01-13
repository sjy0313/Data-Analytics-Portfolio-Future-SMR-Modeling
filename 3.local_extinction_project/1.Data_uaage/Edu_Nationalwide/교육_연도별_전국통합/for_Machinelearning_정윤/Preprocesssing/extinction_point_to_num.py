# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:49:34 2024

@author: Shin
"""

import pandas as pd

# Load improvement extinction index data
new_ep = pd.read_csv('C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/개선_소멸위험지수_2015-2023.csv')

# Convert indexes back to columns
new_ep.reset_index(inplace=True)
new_ep.rename(columns={'index' : '행정구역'}, inplace=True)

# Remove trial data

cities = '서울특별시, 부산광역시, 인천광역시, 대구광역시, 대전광역시, 광주광역시, 울산광역시, 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도, 강원특별자치도, 전북특별자치도, 제주특별자치도'.split(', ')
for city in cities:
    new_ep = new_ep[new_ep['행정구역(동읍면)별'] != city]
    
# Heat removal other than 2015-2021
new_ep = new_ep.loc[:,'행정구역(동읍면)별':'2021']

# %%
# save file
new_ep.to_csv('C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/개선_소멸위험지수_2015-2023(시도제거).csv', index=False)

# %%
'''소멸 위험 등급 나누기'''
new_ep = pd.read_csv('C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/개선_소멸위험지수_2015-2023(시도제거).csv')

# Convert annual data into one data each
df_long = new_ep.melt(id_vars=['행정구역(동읍면)별'], var_name='항목', value_name='소멸위험지수')

# Combine administrative districts and items to create a new 'Administrative District' column
df_long['행정구역(동읍면)별'] = df_long['행정구역(동읍면)별'] + '_' + df_long['항목']

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

grade = df_long.iloc[:, [0, 2]]
grade['소멸위험등급'] = grade['소멸위험등급'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4})
grade.to_csv('C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/소멸등급(2015~2021).csv',index=False)




# %%
# Extraction of grade D (area at risk of extinction)
d = df_long[df_long['소멸위험등급'] == 'D']
c = df_long[df_long['소멸위험등급'] == 'C']
b = df_long[df_long['소멸위험등급'] == 'B']
a = df_long[df_long['소멸위험등급'] == 'A']


# filtered_df = a[a['By administrative district (dong-eup-myeon)'].str.contains('2021')]
# filtered_df2 = a[~a['By administrative district (dong-eup-myeon)'].str.contains('2021')]
#%%
# Extract only 2015 grades
df_long.columns.values[2] = '2015_등급'
df_level = df_long.iloc[:229,[2]]
df_level.drop(columns=['2015'], inplace=True)

#%%

# Convert categorical data to numeric

df_level['2015_등급'] = df_level['2015_등급'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4})
df_level.to_csv('C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/2015_소멸등급.csv',index=False)
#%%

grade_f = df_long.to_csv('C:/Users/Shin/Documents/Final_Project/Mysql/data/grade.csv', index=False)

grade = pd.read_csv('C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/소멸등급(2015~2021).csv')
gra = grade.iloc[:,1:]
grad = pd.concat([df_long,gra], axis=1)
grade_f = grad.to_csv('C:/Users/Shin/Documents/Final_Project/Mysql/data/grade.csv', index=False, header=False)





