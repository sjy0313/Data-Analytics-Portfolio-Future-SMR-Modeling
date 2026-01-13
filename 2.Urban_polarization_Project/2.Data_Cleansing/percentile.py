# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:05:46 2024

@author: Shin
"""
import pandas as pd
df1= pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/Excel/seoulcity/best_copy.xlsx")

numeric_df = df1.select_dtypes(include='number')
seoul_best = numeric_df.quantile([0.25, 0.50, 0.75, 1.00])

# Result output
print("각 열의 사분위수:")
print(seoul_best)
'''
각 열의 사분위수:
      Subway  Primary_School   ...    Park  transaction price(won)
0.25   287.0            364.0  ...   167.0            1.135830e+09 1,135,830,000
0.50   413.0            514.0  ...   398.0            1.635000e+09 1,635,000,000
0.75   527.0            752.0  ...  1600.0            2.022000e+09 2,022,000,000
1.00   981.0           1712.0  ...  3000.0            3.950000e+09 3,950,000,000
'''
new_data= df1[df1['transaction price(won)'] >= 2000000000] 
print(new_data)


#%%
def calculate_column_means(data):
    means = []

    for df in data:
        # Calculate the average of all columns except the first column in each dataframe
        column_means = df.iloc[:, 1:7].mean()
        means.append(column_means)
    
    return means

# Calculate the column average of each data frame by calling a function
data = [df1]
means = calculate_column_means(data)

# Result output
for i, mean in enumerate(means):
    print(f"df{i+1}의 각 열의 평균:")
    print(mean)
    print()





'''
1등급 : 
df1의 각 열의 평균:
Subway               425.08
Primary_School       586.36
Middle_School        447.76
High_School          747.20
General_Hospital    1306.96
Supermarket          890.44
dtype: float64
'''



