# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:29:21 2024

@author: Shin
"""

import pandas as pd
df1= pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/best_copy.xlsx")
df2= pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/worst_copy.xlsx")


numeric_df = df1.select_dtypes(include='number')
seoul_best = numeric_df.quantile([0.25, 0.50, 0.75, 1.00])


new_index = ['A', 'B', 'C', 'D']

# Change the index value to a new value
seoul_best.index.name = '순위'
seoul_best.index = ['1', '2', '3', '4']



numeric_df = df2.select_dtypes(include='number')
seoul_worst = numeric_df.quantile([0.25, 0.50, 0.75, 1.00])

# (['1st priority', '2nd priority', '3rd priority', '4th priority'])

# Result output
print("각 열의 사분위수:")
print(seoul_worst)
'''
각 열의 사분위수:
      Subway  Primary_School   ...    Park  transaction price(won)
0.25   326.0            462.0  ...   897.0             247000000.0
0.50   498.0            608.0  ...  1700.0             297500000.0
0.75   656.0            918.0  ...  2300.0             380000000.0
1.00  1400.0           1417.0  ...  2800.0             650000000.0
'''
# Result output
print("각 열의 사분위수:")
print(seoul_best)

#%%

print(seoul_best)

'''
각 열의 사분위수:
      Subway  Primary_School   ...    Park  transaction price(won)
0.25   287.0            364.0  ...   167.0            1.135830e+09 1,135,830,000
0.50   413.0            514.0  ...   398.0            1.635000e+09 1,635,000,000
0.75   527.0            752.0  ...  1600.0            2.022000e+09 2,022,000,000
1.00   981.0           1712.0  ...  3000.0            3.950000e+09 3,950,000,000
'''

transaction_price = [3950000000,202200000,1635000000,1135830000]
seoul_best['transaction price(won)'] = transaction_price

seoul_best.to_excel('D:\WORKSPACE\github\MYSELF24\Python\MiniProject-May\seoulcity\price.xlsx')
#%%

numeric_df = df2.select_dtypes(include='number')
seoul_worst = numeric_df.quantile([0.25, 0.50, 0.75, 1.00])

seoul_worst.index.name = '순위'
seoul_worst.index = ['1', '2', '3', '4']

transaction_price = [650000000,380000000,297500000,247000000]
seoul_worst['transaction price(won)'] = transaction_price


seoul_worst.to_excel('D:\WORKSPACE\github\MYSELF24\Python\MiniProject-May\seoulcity\price_worst.xlsx')







#%%
combined_df = seoul_worst.add(seoul_best, fill_value=0)

# Divide the values ​​in each column by 2
rank3 = combined_df / 2

# Result output
print("3등급:")
print(rank3)

numeric_df = rank3.select_dtypes(include='number')
rank3 = numeric_df.quantile([0.25, 0.50, 0.75, 1.00])

rank3.index.name = '순위'
rank3.index = ['1', '2', '3', '4']

transaction_price = [2300000000,1201000000,966250000,691415000]
rank3['transaction price(won)'] = transaction_price


rank3.to_excel('D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/Excel/seoulcity/price_rank3.xlsx')





'''
3등급:
      Subway  Primary_School   ...    Park  transaction price(won)
0.25   306.5            413.0  ...   532.0            6.914150e+08 691,415,000
0.50   455.5            561.0  ...  1049.0            9.662500e+08 966,250,000
0.75   591.5            835.0  ...  1950.0            1.201000e+09 1,201,000,000
1.00  1190.5           1564.5  ...  2900.0            2.300000e+09 2,300,000,000
'''





