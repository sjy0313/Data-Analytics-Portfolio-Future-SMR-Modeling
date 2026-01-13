# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:11:31 2024

@author: Shin
"""

# Read product code data as csv file

import pandas as pd
df = pd.read_excel("./data/bestseller_books_2023.xlsx") # Import Kyobo Bookstore Excel file
df.head()

df.to_excel("./data/bestseller_books_2023-1.xlsx")
df.info()
# The data type of ‘product code’ is int64 -> object(str)
df['상품코드'] = df['상품코드'].astype('str')
    
df_goods = df[['상품코드']] # Change series->dataframe by wrapping [] one more time.
df_goods.to_csv("./data/상품코드.csv", index=False) # Delete index column
# Create a dataframe with the sales product ID column
df_id = df[['판매상품ID']]