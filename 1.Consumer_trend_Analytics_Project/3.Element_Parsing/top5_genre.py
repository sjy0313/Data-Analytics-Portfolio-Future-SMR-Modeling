# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:38:18 2024

@author: Shin
"""
import pandas as pd
# 4 genre files -> Save to empty list
excel_f = [] 
for year in range(2020, 2024):
    excel_f.append(pd.read_excel(f"./Miniproject_2_excel data/Genrelist_of_bestseller{year}.xlsx"))
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
# result Result = {key: 1~4, value: 4 years' worth of dataframe}
result_list = list(result.values()) # Insert value into DataFrame value list
result_integrate = pd.concat(result_list) # Create a DataFrame by combining values

# Create a dataframe with the top 5 fields:
top5 = result_integrate.iloc[:4, :5]
top5.index = range(2020, 2024)

sum_row = top5.sum(axis=0)
sum_row.name = '합계'

# Using transpose(), add a 'total' row instead of a new column
top5 = pd.concat([top5, pd.DataFrame(sum_row).transpose()], axis=0)
top5 = top5.rename_axis('연도')
top5.to_excel('./Miniproject_2_excel data/top5_Genre.xlsx', index=True)


fiction = top5['소설'].value_counts().to_dict()
sum_fiction = sum(fiction.keys()) #66

tot_book = [100,100,100,100]
result_integrate['총 권수'] = tot_book 
result_integrate.index = range(2020, 2024)
result_integrate = result_integrate.rename_axis('연도')
result_integrate.to_excel('./Excel_file/Genre_stat.xlsx', index=True)