# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:38:18 2024

@author: Shin
"""
import pandas as pd
# 4개의 장르 파일 -> 빈 리스트에 저장
excel_f = [] 
for year in range(2020, 2024):
    excel_f.append(pd.read_excel(f"./Miniproject_2_excel data/Genrelist_of_bestseller{year}.xlsx"))
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
#result 결과 = {key : 1~4 , value : 4년치 dataframe}
result_list = list(result.values()) # DataFrame인 value값 리스트에 넣기
result_integrate = pd.concat(result_list) # value값들을 합쳐 DataFrame만들기

# top5로 분야로 dataframe만들기 :
top5 = result_integrate.iloc[:4, :5]
top5.index = range(2020, 2024)

sum_row = top5.sum(axis=0)
sum_row.name = '합계'

# transpose() 활용을 통해 새로운 열이아닌 '합계'행 추가
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