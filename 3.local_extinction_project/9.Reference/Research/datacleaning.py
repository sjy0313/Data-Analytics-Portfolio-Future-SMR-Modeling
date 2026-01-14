# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:36:27 2024

@author: Shin
"""
# Read genre data
import pandas as pd

# When receiving genre data, I should have received the top 100.
# Received up to 195th place. Original change (up to 100th place)
# Data validating process
df2 = pd.read_excel('./Project/Genrelist_of_bestseller2021.xlsx')
df2.drop(df2.index[100:], inplace=True) # Data reduction (195 volumes -> 100 volumes)
df2.to_excel('./Project/Genrelist_of_bestseller2021.xlsx', index=False)
  
# Save as dataframe in 4 genre file list
excel_f = [] 
for year in range(2020, 2024):
    excel_f.append(pd.read_excel(f"./Project/Genrelist_of_bestseller{year}.xlsx"))
    print(excel_f)
    

# Enumeration with dictionary data type
df_dict = {f"df{i}": df for i, df in enumerate(excel_f, 1)}
print(df_dict)

# The process of cleansing each year's data depends on the type of genre.
# Since each year is different, extract the desired data by year

# Calculate the frequency of each genre based on the listed dataframe
# 2020
df_gen = df_dict['df1']
# For each data frame, convert the frequency count of the 'Genre' column into a dictionary.
genre_dict = df_gen['장르'].value_counts().to_dict()
# Convert dictionary to data frame and output
genre_df_2020 = pd.DataFrame([genre_dict])
genre_df_2020.rename(index = { 0 :'권수'}, inplace=True)

# 2021
df_gen = df_dict['df2']
# For each data frame, convert the frequency count of the 'Genre' column into a dictionary.
genre_dict = df_gen['장르'].value_counts().to_dict()
# Convert dictionary to data frame and output
genre_df_2021 = pd.DataFrame([genre_dict])
genre_df_2021.rename(index = { 0 :'권수'}, inplace=True)

# 2022
df_gen = df_dict['df3']
# For each data frame, convert the frequency count of the 'Genre' column into a dictionary.
genre_dict = df_gen['장르'].value_counts().to_dict()
# Convert dictionary to data frame and output
genre_df_2022 = pd.DataFrame([genre_dict])
genre_df_2022.rename(index = { 0 :'권수'}, inplace=True)

# 2023
df_gen = df_dict['df4']
# For each data frame, convert the frequency count of the 'Genre' column into a dictionary.
genre_dict = df_gen['장르'].value_counts().to_dict()
# Convert dictionary to data frame and output
genre_df_2023 = pd.DataFrame([genre_dict])
genre_df_2023.rename(index = { 0 :'권수'}, inplace=True)

# Through genre_df_202[], we were able to confirm that the number of genres by year was different.
# Outliers found while receiving data for 2022:

df3 = pd.read_excel('./Project/Genrelist_of_bestseller2022.xlsx')

import numpy as np
npt = np.array(df3)
noprint = np.where(npt == '절판')
# Line 1/Line 92 Values ​​are searched on the web and changed.
# Change the ‘genre’ value of ‘out of print’ in the first row to ‘self-development’
df3.loc[0, '장르'] = '자기계발'
# Change the 'Genre' value in line 93, 'Out of Print', to 'Foreign Language'.
df3.loc[92, '장르'] = '외국어'
df3.to_excel('./Project/Genrelist_of_bestseller2022.xlsx')

genre_dicts = df3['장르'].value_counts().to_dict()
genre_df_2022 = pd.DataFrame([genre_dicts])
genre_df_2022.rename(index = { 0 :'권수'}, inplace=True)


genre_df_2020.to_excel('./Project/Num_of_Genrelist2020.xlsx')
genre_df_2021.to_excel('./Project/Num_of_Genrelist2021.xlsx')
genre_df_2022.to_excel('./Project/Num_of_Genrelist2022.xlsx')
genre_df_2023.to_excel('./Project/Num_of_Genrelist2023.xlsx')