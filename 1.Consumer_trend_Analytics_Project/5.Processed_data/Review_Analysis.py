# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 20:24:35 2024

@author: 신정윤
"""

#%%

import pandas as pd

def book_info():
    excel_bookinfo = []  # 각 연도별 도서 정보를 저장할 리스트

    for year in range(2020, 2024):
        excel_bookinfo.append(pd.read_excel(f"./Excel_file/book_info({year}).xlsx"))
    return excel_bookinfo

print(book_info()) # 4년치 데이터 불러오기

excel_files = book_info()



#%%
def book_review():
    reviews = []  # 4개년 3요소+장르 정보 DataFrame
    for year in range(2020, 2024):
        df_features = pd.read_excel(f"./Excel_file/book_info({year}).xlsx")
        df_genre = pd.read_excel(f"./Excel_file/Genrelist_of_bestseller{year}.xlsx")
        review = pd.concat([df_features, df_genre], axis=1)
        reviews.append(review)  # 각각의 합쳐진 데이터프레임 추가해주기
        
    combined_reviews = pd.concat(reviews, axis=0)   # 행방향 결합
    combined_reviews.reset_index(drop=True, inplace=True)
    return combined_reviews

for_rev = book_review()

# 리뷰로 알아보는 소비자들의 심리
shortreview_freq = for_rev['shortreview'].value_counts().to_dict()
# 소비자가 남긴 top3 comment review
# 1. "집중돼요"
# 2. "고마워요"
# 3. "도움돼요"


# top3 review 는 어느 분야에 속할까?  
shortreview_top1= for_rev[for_rev['shortreview'].str.contains('집중돼요')]
which_genre = shortreview_top1['장르'].value_counts()
df1 = pd.DataFrame(which_genre)
# Pandas 의 df.nlargest(n, columns, keep='first') 활용
# n :출력할 행의 수
# columns : 정렬의 기준이 될 열
# keep : first면 위부터, last면 아래부터, all이면 모두 출력
top3_genre_1 = df1.nlargest(3, 'count') 
# 출력할 행의 수 : top3 가장 많이 출현한 장르 , 열 : 'count' 

# Top2 -> '고마워요'
shortreview_top2= for_rev[for_rev['shortreview'].str.contains('고마워요')]
which_genre2 = shortreview_top2['장르'].value_counts()
df2 = pd.DataFrame(which_genre2)
top3_genre_2 = df2.nlargest(3, 'count')
# Top3 -> '도움돼요'
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
            genre_df = pd.DataFrame([genre_freq]) # 딕셔너리 -> dataframe 변환
            # f'{idx}'에서 idx값을 위에 열거 순에 따라 인덱스 값을 1~4까지 반환
            genre_df.rename(index={0: f'{idx}'}, inplace=True) 
            shortreview_by_years[f'{idx}'] = genre_df
            
        return shortreview_by_years
    # 'book_rv' 함수 호출 [데이터 정제 및 통합]
    
result = book_rv(excel_files)

import pandas as pd

def book_info():
    excel_bookinfo = []  # 각 연도별 도서 정보를 저장할 리스트

    for year in range(2020, 2024):
        excel_bookinfo.append(pd.read_excel(f"./project/book_info({year}).xlsx"))
    return excel_bookinfo

def book_rv():
    shortreview_by_years = {}
    excel_files = book_info()
    for idx, df in enumerate(excel_files, start=1):
        rv_freq = df['shortreview'].value_counts().to_dict() 
        genre_df = pd.DataFrame([rv_freq]) # 딕셔너리 -> dataframe 변환
        # f'{idx}'에서 idx값을 위에 열거 순에 따라 인덱스 값을 1~4까지 반환
        genre_df.rename(index={0: f'{idx}'}, inplace=True) 
        shortreview_by_years[f'{idx}'] = genre_df
            
    return shortreview_by_years
    # 'book_rv' 함수 호출 [데이터 정제 및 통합]

result = book_rv()
print(result)'''