# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:36:19 2024

@author: Shin
"""

# 장르와 위에서 web_scroll 함수를 활용해 도출한 다른 요소들(author/product/shortreview)의 결합

# MP-Genre-final.py 에서만든 장르 dataframe 가져오기

import pandas as pd
df_features = pd.read_excel('./project/book_info.xlsx')
df_genre = pd.read_excel('./Project/Genrelist_of_bestseller2023.xlsx')
# 열 기준 병합하기

df_bestseller2023 = pd.concat([df_features, df_genre], axis=1)

# 몇가지 고유의 장르들이 2023년 배스트샐러에 채택 되었는지
genres = df_genre["장르"].unique()
print(len(genres)) # 14
# 요약정보
df_bestseller2023.describe()
'''
         Product                                Author shortreview   장르
count        100                                   100         100  100
unique       100                                    99           8   14
top     세이노의 가르침  David Cho ·  해커스어학연구소   · 2023.07.24        도움돼요   소설
freq           1                                     2          48   22
'''
# 총 8종류의 한줄평(shortreview)과 14종류의 장르가 존재함을 알 수 있고
# 48개의 (도움돼요)한줄평으로 가장많은 횟수를 차지했으며
# 22권의 소설책 확인되었다 직접 확인해보자.
#%%
# 장르 별 bestseller책들이 차지하는 비중을 구해보자.
# 시리즈 객체의 고유값개수를 세는데 사용 : value_counts() 매서드
df_bestseller2023['장르'].value_counts() 
'''
장르
소설         22
경제/경영      17
자기계발       15
인문         14
외국어         6
시/에세이       5
어린이(초등)     5
과학          5
만화          4
역사/문화       2
정치/사회       2
건강          1
컴퓨터/IT      1
청소년         1
Name: count, dtype: int64
'''
