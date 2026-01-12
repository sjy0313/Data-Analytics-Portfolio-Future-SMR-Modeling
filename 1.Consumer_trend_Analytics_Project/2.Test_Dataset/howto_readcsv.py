# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:11:31 2024

@author: Shin
"""

# 상품코드 데이터 csv파일로 읽기

import pandas as pd
df = pd.read_excel("./data/bestseller_books_2023.xlsx") # 교보문고 액셀파일 불러오기
df.head()

df.to_excel("./data/bestseller_books_2023-1.xlsx")
df.info()
# '상품코드'의 자료형을 int64 -> object(str)
df['상품코드'] = df['상품코드'].astype('str')
    
df_goods = df[['상품코드']] # []를 한번 더 싸줘서 series->dataframe 변경.
df_goods.to_csv("./data/상품코드.csv", index=False) # index열 삭제
# 판매상품ID 열로 dataframe 만들기
df_id = df[['판매상품ID']]