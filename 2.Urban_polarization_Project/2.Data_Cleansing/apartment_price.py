# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:11:55 2024

@author: Shin
"""
# 최고가 + 최저가 아파트 정보 통합  
import pandas as pd
df1= pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/Excel/Seoul_city.xlsx")
df2= pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/Excel/Seoul_city_w.xlsx")

df1.describe()
'''       Subway  Primary_School   ...         Park  transaction price(won)
count   25.00000        25.000000  ...    25.000000            2.500000e+01
mean   425.08000       586.360000  ...   974.440000            1.721660e+09
std    187.13162       389.045293  ...   996.891464            7.984222e+08
min    147.00000        77.000000  ...    16.000000            7.615000e+08
25%    287.00000       364.000000  ...   167.000000            1.135830e+09
50%    413.00000       514.000000  ...   398.000000            1.635000e+09
75%    527.00000       752.000000  ...  1600.000000            2.022000e+09
max    981.00000      1712.000000  ...  3000.000000            3.950000e+09'''

apartment = pd.concat([df1, df2], axis=0, ignore_index = True)
df_byprice = apartment.sort_values(by = 'transaction price(won)', ascending = False)
# 아파트 정보
apartment_info_df = df_byprice.select_dtypes(include='object')
#%%
# 숫자형 데이터 타입을 가진 열들로만 dataframe 생성 : 

numeric_df = df_byprice.select_dtypes(include='number')
seoul_apartment = numeric_df.quantile([0.0, 0.25, 0.50, 0.75, 1.00])
seoul_apartment.describe()
print(seoul_apartment['transaction price(won)'])
'''
1     120000000
2     298125000
3     705750000
4    1585000000
5    3950000000'''
# 숫자를 정수로 변환하는 람다함수 : 
seoul_apartment = seoul_apartment.applymap(lambda x: int(x) if isinstance(x, (int, float)) else x)
# 내림정렬 : 
    
price_by_rank = seoul_apartment.sort_values(by = 'transaction price(won)', ascending = False)


seoul_apartment.index.name = '등급'
seoul_apartment.index = ['1', '2', '3', '4', '5']


transaction_price = [3950000000,1585000000,705750000,298125000,120000000]
seoul_apartment['transaction price(won)'] = transaction_price
#%%
def calculate_column_quantile(data):
   quantile = []

    for df in data:
        # 각 데이터프레임의 첫 번째 열을 제외한 나머지 열에 대해 평균 계산
        data = data.iloc[-1].quantile([0.0, 0.25, 0.50, 0.75, 1.00])
        quantile.append(data)
    
    return means

# 함수 호출하여 각 데이터프레임의 열 평균 계산
data = [apartment, df1, df2]
quantile = calculate_column_means(data)

# 결과 출력

for i, quantile in enumerate(means):
    print(f"df{i}의 평균:")
    print(mean)
    print()
#%%
# 50개 아파트 데이터 평균 
def calculate_column_means(data):
    quantiles= []

    for df in data:
        # 각 데이터프레임의 첫 번째 열을 제외한 나머지 열에 대해 평균 계산
        quantiles = df.iloc[-1].mean()
        means.append(quantiles)
    
    return quantiles

# 함수 호출하여 각 데이터프레임의 열 평균 계산
data = [apartment, df1, df2]
means = calculate_column_means(data)

# 결과 출력

for i,quantile in enumerate(quantiles):
    print(f"df{i}의 평균:")
    print(quantile)
    print()
# 전체평균 = 3등급 
'''
Subway               504.54
Primary_School       629.70
Middle_School        592.64
High_School          788.78
General_Hospital    1422.00
Supermarket         1076.66
Park                1283.80'''
# 고가 아파트들의 평균 = 1등급
'''
Subway               425.08
Primary_School       586.36
Middle_School        447.76
High_School          747.20
General_Hospital    1306.96
Supermarket          890.44
Park                 974.44'''
# 저가 아파트들의 평균 = 5등급
'''
Subway               584.00
Primary_School       673.04
Middle_School        737.52
High_School          830.36
General_Hospital    1537.04
Supermarket         1262.88
Park                1593.16'''

# means[1] 전체평균
rank3 = pd.DataFrame(means[1], columns=['mean_value'])
rank1 = pd.DataFrame(means[2], columns=['mean_value'])
rank5 = pd.DataFrame(means[3], columns=['mean_value'])
#%%
# 2등급 계산 :
combined_df_1 = rank3.add(rank1, fill_value=0)

# 각 열의 값을 2로 나누기
rank2 = combined_df_1 / 2
#%%
# 4등급 계산 :
combined_df_2 = rank3.add(rank5, fill_value=0)

# 각 열의 값을 2로 나누기
rank4 = combined_df_2 / 2

#%%
def modify_column_names(df_list):
    # List to store modified dataframes
    modified_dfs = []
    
    # Modify column names for each dataframe in the list
    for i, df in enumerate(df_list):
        new_columns = {col: f"{i+1}등급" for col in df.columns}
        modified_df = df.rename(columns=new_columns)
        modified_dfs.append(modified_df)
    
    return modified_dfs

def merge_dataframes(dfs):
    # Modify column names first
    modified_dfs = modify_column_names(dfs)
    
    # Create keys for multi-level indexing
    keys = [f"{i+1}등급" for i in range(len(modified_dfs))]
    
    # Concatenate dataframes along axis=1 with keys
    merged_df = pd.concat(modified_dfs, axis=1, keys=keys)
    
    # Promote the current header to be the new header
    merged_df.columns = merged_df.columns.droplevel(1)
    
    return merged_df

data = [rank1, rank2, rank3, rank4, rank5]

# Merge and print the result
merged_dataframe = merge_dataframes(data)

# Print the merged dataframe
print("통합된 데이터프레임:")
print(merged_dataframe)
#%%
merged_dataframe.to_excel('D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/Excel/seoulcity/price_rank.xlsx')
