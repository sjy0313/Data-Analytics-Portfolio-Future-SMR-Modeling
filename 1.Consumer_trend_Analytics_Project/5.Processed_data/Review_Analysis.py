# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 20:24:35 2024

@author: 신정윤
"""

#%%

import pandas as pd

def book_info():
    excel_bookinfo = []  # List to store book information for each year

    for year in range(2020, 2024):
        excel_bookinfo.append(pd.read_excel(f"./Excel_file/book_info({year}).xlsx"))
    return excel_bookinfo

print(book_info()) # Load 4 years worth of data

excel_files = book_info()



#%%
def book_review():
    reviews = []  # 4 years 3 elements + genre information DataFrame
    for year in range(2020, 2024):
        df_features = pd.read_excel(f"./Excel_file/book_info({year}).xlsx")
        df_genre = pd.read_excel(f"./Excel_file/Genrelist_of_bestseller{year}.xlsx")
        review = pd.concat([df_features, df_genre], axis=1)
        reviews.append(review)  # Add each combined data frame
        
    combined_reviews = pd.concat(reviews, axis=0)   # row direction join
    combined_reviews.reset_index(drop=True, inplace=True)
    return combined_reviews

for_rev = book_review()

# Consumer psychology through reviews
shortreview_freq = for_rev['shortreview'].value_counts().to_dict()
# Top3 comments left by consumers review
# 1. “I’m focused.”
# 2. “Thank you”
# 3. “It’s helpful.”


# What field do the top3 reviews belong to?
shortreview_top1= for_rev[for_rev['shortreview'].str.contains('집중돼요')]
which_genre = shortreview_top1['장르'].value_counts()
df1 = pd.DataFrame(which_genre)
# Utilizing Pandas' df.nlargest(n, columns, keep='first')
# n : Number of rows to output
# columns: Columns that will be the basis for sorting
# keep: Print from the top for first, from below for last, print all for all.
top3_genre_1 = df1.nlargest(3, 'count') 
# Number of rows to print: top3 most frequently appearing genres, column: 'count'

# Top2 -> ‘Thank you’
shortreview_top2= for_rev[for_rev['shortreview'].str.contains('고마워요')]
which_genre2 = shortreview_top2['장르'].value_counts()
df2 = pd.DataFrame(which_genre2)
top3_genre_2 = df2.nlargest(3, 'count')
# Top3 -> ‘It’s helpful’
shortreview_top3= for_rev[for_rev['shortreview'].str.contains('도움돼요')]
which_genre3 = shortreview_top3['장르'].value_counts()
df3 = pd.DataFrame(which_genre3)
top3_genre_3 = df3.nlargest(3, 'count')




#%%
'''
def book_rv():
    for seq in range(0, 5):
        shortreview_by_years = {}
        for idx, df in enumerate(excel_files, start = 1):
            rv_freq = df['shortreview'].value_counts().to_dict() 
            genre_df = pd.DataFrame([genre_freq]) # Dictionary -> dataframe conversion
            # In f'{idx}', returns index values ​​from 1 to 4 according to the order of idx values ​​listed above.
            genre_df.rename(index={0: f'{idx}'}, inplace=True) 
            shortreview_by_years[f'{idx}'] = genre_df
            
        return shortreview_by_years
    # Calling the 'book_rv' function [Data Cleaning and Integration]
    
result = book_rv(excel_files)

import pandas as pd

def book_info():
    excel_bookinfo = []  # List to store book information for each year

    for year in range(2020, 2024):
        excel_bookinfo.append(pd.read_excel(f"./project/book_info({year}).xlsx"))
    return excel_bookinfo

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
    # Calling the 'book_rv' function [Data Cleaning and Integration]

result = book_rv()
print(result)'''