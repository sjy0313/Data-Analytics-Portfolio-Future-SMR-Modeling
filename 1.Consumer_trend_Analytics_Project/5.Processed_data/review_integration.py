# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:27:07 2024

@author: 신정윤
"""
import pandas as pd
df1 = pd.read_excel('C:/Users/신정윤/Documents/Python/새 폴더/MYSELF24/DataAnalysis/project/book_info(2020).xlsx')
df2 = pd.read_excel('C:/Users/신정윤/Documents/Python/새 폴더/MYSELF24/DataAnalysis/project/book_info(2021).xlsx')
df3 = pd.read_excel('C:/Users/신정윤/Documents/Python/새 폴더/MYSELF24/DataAnalysis/project/book_info(2022).xlsx')
df4 = pd.read_excel('C:/Users/신정윤/Documents/Python/새 폴더/MYSELF24/DataAnalysis/project/book_info(2023).xlsx')
# ./project/book_info(2023).xlsx


#%%

import pandas as pd
# Import Excel file
def book_info():
    excel_bookinfo = []  

    for year in range(2020, 2024):
        excel_bookinfo.append(pd.read_excel(f"./project/book_info({year}).xlsx"))
    return excel_bookinfo
# 'book_rv' [data cleansing and integration]
def book_rv():
    shortreview_by_years = {}
    excel_files = book_info()
    for idx, df in enumerate(excel_files, start=1):
        rv_freq = df['shortreview'].value_counts().to_dict() 
        genre_df = pd.DataFrame([rv_freq]) # Dictionary -> dataframe conversion
        # In f'{idx}', returns index values ​​from 1 to 4 according to the order of idx values ​​listed above.
        genre_df.rename(index={0: f'{idx}'}, inplace=True) 
        shortreview_by_years[f'{idx}'] = genre_df
            
    return shortreview_by_years
    

result = book_rv()

# Save Excel file

# result Result = {key: 1~4, value: 4 years' worth of dataframe}
result_list = list(result.values()) # Inserting into the list of values ​​that are DataFrame
result_integrate = pd.concat(result_list) # Create a DataFrame by combining values
years = result_integrate.set_index(pd.Index(['2020','2021','2022','2023']), inplace = True) # Specify index

tot_book = [100,100,100,100]
result_integrate['총 권수'] = tot_book 

shortreview = result_integrate.rename_axis('연도')
shortreview.rename(columns={'Unnamed: 0':'리뷰'}, inplace=True)

shortreview.to_excel('./project/Review_stat.xlsx', index=True)




#%%
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

sr1 = df1['shortreview'].value_counts().to_dict()
sr1_1 = pd.Series(sr1)
sre1 = pd.DataFrame([sr1_1])
sr2 = df2['shortreview'].value_counts().to_dict()
sr2_2 = pd.Series(sr2)
sre2 = pd.DataFrame([sr2_2])
sr3 = df3['shortreview'].value_counts().to_dict()
sr3_3 = pd.Series(sr3)
sre3 = pd.DataFrame([sr3_3])
sr4 = df4['shortreview'].value_counts().to_dict()
sr4_4 = pd.Series(sr4)
sre4 = pd.DataFrame([sr4_4])

series = [sre1, sre2, sre3, sre4]
results = pd.concat(series, join='outer')
results.index = range(2020, 2024)
results = results.rename_axis('연도')


results.to_excel('C:/Users/신정윤/Documents/Python/새 폴더/MYSELF24/DataAnalysis/shortreview_stat.xlsx', index=True)
#%%































