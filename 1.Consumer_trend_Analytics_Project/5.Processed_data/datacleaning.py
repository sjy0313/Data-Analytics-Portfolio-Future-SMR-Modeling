# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:36:27 2024

@author: Shin
"""
# Read genre data
import pandas as pd

# Data cleansing for 2022:
    
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

# 2021 Data Cleansing:

# When receiving genre data, I should have received the top 100.
# Received up to 196th place. Original change (up to 100th place)
# Data validating process
df2 = pd.read_excel('./Project/Genrelist_of_bestseller2021.xlsx')
df2.drop(df2.index[100:], inplace=True) # Data reduction (196 volumes -> 100 volumes)
df2.to_excel('./Project/Genrelist_of_bestseller2021.xlsx', index=False)
#%%

  
# 4 genre files -> Save to empty list
excel_f = [] 
for year in range(2020, 2024):
    excel_f.append(pd.read_excel(f"./Project/Genrelist_of_bestseller{year}.xlsx"))
    print(excel_f)
    
    
# Because the value of excel_f is a dataframe,
# The values ​​were grouped into a list data type as shown above.
# In excel_f, the values ​​of the four files are made into variables that can be assigned to the function in order.

# Process 4 years at once
import pandas as pd

def cleans_integrate(excel_f):
    genre_freq_years = {} # Dictionary to store 'genre' frequency by year
    for idx, df in enumerate(excel_f, start = 1):
        genre_freq = df['장르'].value_counts().to_dict() # Calculate the frequency of the 'Genre' column and convert it to a dictionary
        genre_df = pd.DataFrame([genre_freq]) # Dictionary -> dataframe conversion
        # In f'{idx}', returns index values ​​from 1 to 4 according to the order of idx values ​​listed above.
        genre_df.rename(index={0: f'{idx}'}, inplace=True) 
        genre_freq_years[f'{idx}'] = genre_df
        
    return genre_freq_years
# Call 'cleans_integrate' function [Data Cleaning and Integration]
result = cleans_integrate(excel_f)


#%%


# Data preprocessing by year

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


genre_df_2020.to_excel('./Project/Num_of_Genrelist2020.xlsx')
genre_df_2021.to_excel('./Project/Num_of_Genrelist2021.xlsx')
genre_df_2022.to_excel('./Project/Num_of_Genrelist2022.xlsx')
genre_df_2023.to_excel('./Project/Num_of_Genrelist2023.xlsx')