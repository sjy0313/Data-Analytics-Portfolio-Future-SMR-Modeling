# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:52:45 2024

@author: jcp
"""
'''데이터 통합'''

import pandas as pd

# Load dependent variable data
ext = pd.read_csv('./preprocessed/개선_소멸위험지수_2015-2021.csv', index_col=0)

ext_grade = pd.read_csv('./preprocessed/소멸위험등급_2015-2021.csv', index_col=0)

# Load independent variables data
edu = pd.read_excel('./preprocessed/종합파일들/교육종합.xlsx')
cure = pd.read_excel('./preprocessed/종합파일들/보건종합.xlsx')
water_sup = pd.read_excel('./preprocessed/종합파일들/상하수도종합.xlsx')
local = pd.read_excel('./preprocessed/지역사회건설/지역사회건설_2015~2021_전국_종합.xlsx')
management = pd.read_excel('./preprocessed/종합파일들/행정.xlsx')
social = pd.read_excel('./preprocessed/종합파일들/사회보호.xlsx')
job = pd.read_excel('./preprocessed/종합파일들/업종종합_v0.1.xlsx')

# Preprocessing of loaded data such as column names, etc.
edu.rename(columns={'행정구역(동읍면)별':'행정구역'}, inplace=True)
cure.rename(columns={'시도별':'행정구역'}, inplace=True)
water_sup.rename(columns={'시도별':'행정구역'}, inplace=True)
local.rename(columns={'시도별':'행정구역'}, inplace=True)
management.rename(columns={'시군구':'행정구역'}, inplace=True)
social.rename(columns={'시도별':'행정구역'}, inplace=True)
job.rename(columns={'시도별':'행정구역'}, inplace=True)

# %%
# Remove trial data
def remove_sido(data): # Trial data removal function
    
    cities = '서울특별시, 부산광역시, 인천광역시, 대구광역시, 대전광역시, 광주광역시, 울산광역시, 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도, 강원특별자치도, 전북특별자치도, 제주특별자치도'.split(', ')
    years = list(range(2015,2022))
    cities_years = []
    for city in cities:
        for year in years:
            cities_years.append(city+'_'+str(year))
    return data[~data['행정구역'].isin(cities_years)]

ext_grade = remove_sido(ext_grade)
edu = remove_sido(edu)
cure = remove_sido(cure)
water_sup = remove_sido(water_sup)
local = remove_sido(local)
management = remove_sido(management)
social = remove_sido(social)
job = remove_sido(job)


# %%
# Find data with different administrative district names
# Merge two data frames (based on administrative district)
# merged_df = pd.merge(edu, job, on='administrative district', how='outer', indicator=True)

# # Filter only administrative districts that do not exist in common in the merge results
# different_values_df = merged_df[merged_df['_merge'] != 'both']
# %%
# merge
merged_df = pd.merge(edu, cure, on='행정구역')
merged_df = pd.merge(merged_df, water_sup, on='행정구역')
# merged_df = pd.merge(merged_df, local, on='Administrative district')
merged_df = pd.merge(merged_df, management, on='행정구역')
# merged_df = pd.merge(merged_df, social, on='administrative district')
# merged_df = pd.merge(merged_df, job, on='administrative district')
merged_df = pd.merge(merged_df, ext_grade, on='행정구역')

# Check for missing values
missing_count_per_column = merged_df.isnull().sum() # doesn't exist
# %%
# save file
merged_df.to_csv('./preprocessed/merged_data.csv')
merged_df.to_excel('./preprocessed/merged_data.xlsx')
# %%
