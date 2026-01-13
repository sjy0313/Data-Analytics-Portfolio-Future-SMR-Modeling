# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:03:15 2024

@author: Shin
"""

import pandas as pd

def book_info():
    excel_bookinfo = []  # List to store book information for each year

    for year in range(2020, 2024):
        excel_bookinfo.append(pd.read_excel(f"./Miniproject_2_excel data/book_info({year}).xlsx"))
    return excel_bookinfo

print(book_info()) # Load 4 years worth of data

excel_files = book_info()



#%%
def book_review():
    reviews = []  # 4 years 3 elements + genre information DataFrame
    for year in range(2020, 2024):
        df_features = pd.read_excel(f"./Miniproject_2_excel data/book_info({year}).xlsx")
        df_genre = pd.read_excel(f"./Miniproject_2_excel data/Genrelist_of_bestseller{year}.xlsx")
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


# Excluding duplicate books from year to year, what fields have received ‘thank you’ reviews?
shortreview_top2 # Among the books that received a ‘Thank you’ review, duplicate books were included.
dup_deleted = shortreview_top2.drop_duplicates() # Remove duplicates
fic_or_poet_cnt = dup_deleted[dup_deleted['shortreview'].str.contains('고마워요')] 
fic_cnt = fic_or_poet_cnt['장르'].value_counts() # Number of volumes occupied by field





   

# Top3 -> ‘It’s helpful’
shortreview_top3= for_rev[for_rev['shortreview'].str.contains('도움돼요')]
which_genre3 = shortreview_top3['장르'].value_counts()
df3 = pd.DataFrame(which_genre3)
top3_genre_3 = df3.nlargest(3, 'count')





