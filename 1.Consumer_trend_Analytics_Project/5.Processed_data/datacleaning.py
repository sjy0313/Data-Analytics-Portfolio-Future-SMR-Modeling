# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:36:27 2024

@author: Shin
"""
# 장르 데이터 읽어오기 
import pandas as pd

# 2022년 데이터 정제 : 
    
#genre_df_202[] 통해 연도별 장르 개수가 다름을 확인 할 수 있었다
# 2022년 데이터를 받아오는 과정에서 이상값 발견: 

df3 = pd.read_excel('./Project/Genrelist_of_bestseller2022.xlsx')

import numpy as np
npt = np.array(df3)
noprint = np.where(npt == '절판')
# 1행/ 92행 값 웹 상에서 검색 후 변경해줌.
# 1번째 행의 '장르' 값인 '절판'을 '자기계발'로 변경
df3.loc[0, '장르'] = '자기계발'
# 93번째 행의 '장르' 값인 '절판'을 '외국어'로 변경
df3.loc[92, '장르'] = '외국어'
df3.to_excel('./Project/Genrelist_of_bestseller2022.xlsx')

genre_dicts = df3['장르'].value_counts().to_dict()
genre_df_2022 = pd.DataFrame([genre_dicts])
genre_df_2022.rename(index = { 0 :'권수'}, inplace=True)

# 2021년 데이터 정제 : 

# 장르 데이터를 받아올 떄 100위 까지 받아왔어야 했는데
# 196위까지 받아옴. 원본 변경(100위까지)
# Data validating process
df2 = pd.read_excel('./Project/Genrelist_of_bestseller2021.xlsx')
df2.drop(df2.index[100:], inplace=True) # 데이터 축소(196권 -> 100권)
df2.to_excel('./Project/Genrelist_of_bestseller2021.xlsx', index=False)
#%%

  
# 4개의 장르 파일 -> 빈 리스트에 저장
excel_f = [] 
for year in range(2020, 2024):
    excel_f.append(pd.read_excel(f"./Project/Genrelist_of_bestseller{year}.xlsx"))
    print(excel_f)
    
    
# excel_f 의 value값이 dataframe이기 때문에,
#value값을 위와 같이 리스트자료형으로 묶어주었다. 
#excel_f에서 4개파일의 value들을 차례대로 함수에 할당할 수 있는 변수로 만들어줌. 

# 4개년도 한번에 처리
import pandas as pd

def cleans_integrate(excel_f):
    genre_freq_years = {} # 연도 별 '장르'빈도수 저장할 딕셔너리 
    for idx, df in enumerate(excel_f, start = 1):
        genre_freq = df['장르'].value_counts().to_dict() # '장르'열 빈도수 계산 후 딕셔너리로 변환
        genre_df = pd.DataFrame([genre_freq]) # 딕셔너리 -> dataframe 변환
        # f'{idx}'에서 idx값을 위에 열거 순에 따라 인덱스 값을 1~4까지 반환
        genre_df.rename(index={0: f'{idx}'}, inplace=True) 
        genre_freq_years[f'{idx}'] = genre_df
        
    return genre_freq_years
# 'cleans_integrate' 함수 호출 [데이터 정제 및 통합]
result = cleans_integrate(excel_f)


#%%


# 연도 별로 데이터전처리

# 열거된 dataframe바탕으로 장르 별 빈도수 계산
# 2020년
df_gen = df_dict['df1']
# 각 데이터프레임에 대해 '장르'열의 빈도수를 딕셔너리로 변환
genre_dict = df_gen['장르'].value_counts().to_dict()
# 딕셔너리를 데이터프레임으로 변환하여 출력
genre_df_2020 = pd.DataFrame([genre_dict])
genre_df_2020.rename(index = { 0 :'권수'}, inplace=True)

# 2021년
df_gen = df_dict['df2']
# 각 데이터프레임에 대해 '장르'열의 빈도수를 딕셔너리로 변환
genre_dict = df_gen['장르'].value_counts().to_dict()
# 딕셔너리를 데이터프레임으로 변환하여 출력
genre_df_2021 = pd.DataFrame([genre_dict])
genre_df_2021.rename(index = { 0 :'권수'}, inplace=True)

# 2022년
df_gen = df_dict['df3']
# 각 데이터프레임에 대해 '장르'열의 빈도수를 딕셔너리로 변환
genre_dict = df_gen['장르'].value_counts().to_dict()
# 딕셔너리를 데이터프레임으로 변환하여 출력
genre_df_2022 = pd.DataFrame([genre_dict])
genre_df_2022.rename(index = { 0 :'권수'}, inplace=True)

# 2023년
df_gen = df_dict['df4']
# 각 데이터프레임에 대해 '장르'열의 빈도수를 딕셔너리로 변환
genre_dict = df_gen['장르'].value_counts().to_dict()
# 딕셔너리를 데이터프레임으로 변환하여 출력
genre_df_2023 = pd.DataFrame([genre_dict])
genre_df_2023.rename(index = { 0 :'권수'}, inplace=True)


genre_df_2020.to_excel('./Project/Num_of_Genrelist2020.xlsx')
genre_df_2021.to_excel('./Project/Num_of_Genrelist2021.xlsx')
genre_df_2022.to_excel('./Project/Num_of_Genrelist2022.xlsx')
genre_df_2023.to_excel('./Project/Num_of_Genrelist2023.xlsx')