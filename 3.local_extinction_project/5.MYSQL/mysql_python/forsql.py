# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:10:40 2024

@author: Shin
"""

import pandas as pd
# health
df = pd.read_excel("C:/Users/Shin/Documents/Final_Project/Mysql/data/health.xlsx")  

csv_file_path = "C:/Users/Shin/Documents/Final_Project/Mysql/data/health.csv"
df.to_csv(csv_file_path, index=False, header=False)
print(csv_file_path)
#%%
# education
df = pd.read_excel("C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/교육/교육종합_v0.2.xlsx")  
csv_file_path = "C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/교육/education.csv"
df.to_csv(csv_file_path, index=False, header=False)
print(csv_file_path)

#%%
# Grade summary
df3 = pd.read_excel("C:/Users/Shin/Documents/Final_Project/Mysql/data/등급.xlsx")  
csvfile = "C:/Users/Shin/Documents/Final_Project/Mysql/data/grade.csv"
df3.to_csv(csvfile, index=False, header=False)
print(csv_file_path)



#%%
import pandas as pd
df1 = pd.read_excel("C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/등급.xlsx") 
csvfile = "C:/Users/Shin/Documents/Final_Project/Mysql/data/csv/grade.csv"
df1.to_csv(csvfile, index=False, header=False)


'''
df2 = df.iloc[:,1]
df1 = df.iloc[:,25:]

level = pd.concat([df2,df1],axis=1, ignore_index = True)

df1
csv_file_path = "C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/교육/merged.csv"
level.to_csv(csv_file_path, index=False, header=False)

csv_file_path = "C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/교육/merged1.csv"
df1.to_csv(csv_file_path, index=False, header=False)
print(csv_file_path)
'''
#%%
# administration
df4 = pd.read_excel("C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/행정_종합_2015~2021_v1.xlsx") 
csvfile = "C:/Users/Shin/Documents/Final_Project/Mysql/data/administration.csv"
df4.to_csv(csvfile, index=False, header=False)

# traffic
df5 = pd.read_excel("C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/교통_2015~2021_전국_종합_v1.xlsx") 
csvfile = "C:/Users/Shin/Documents/Final_Project/Mysql/data/traffic.csv"
df5.to_csv(csvfile, index=False, header=False)
#%%
# housing, transportation
import pandas as pd

# Load the data
df7 = pd.read_excel("C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/주거_2015~2021_전국_종합_v1.xlsx")

df8 = pd.read_excel("C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/교통_2015~2021_전국_종합_v1.xlsx")
def remove_sido(data):
    # List of cities
    cities = [
        '서울특별시', '부산광역시', '인천광역시', '대구광역시', '대전광역시', '광주광역시', 
        '울산광역시', '경기도', '충청북도', '충청남도', '전라남도', '경상북도', '경상남도', 
        '강원특별자치도', '전북특별자치도', '제주특별자치도'
    ]
    
    # List of years
    years = list(range(2015, 2022))
    
    # Create city_year combinations
    cities_years = [f"{city}_{year}" for city in cities for year in years]
    
    # Filter out rows where '행정구역' is in the cities_years list
    return data[~data['시도별'].isin(cities_years)]


df7_filtered = remove_sido(df7)
df8_filtered = remove_sido(df8)


csvfile = "C:/Users/Shin/Documents/Final_Project/Mysql/data/dwelling.csv"
df7.to_csv(csvfile, index=False, header=False)
csvfile1 = "C:/Users/Shin/Documents/Final_Project/Mysql/data/traffic.csv"
df8.to_csv(csvfile1, index=False, header=False)


#%%
# social protection
df8 = pd.read_excel("C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/사회보호.xlsx")

csvfile = "C:/Users/Shin/Documents/Final_Project/Mysql/data/csv/socialprotection.csv"
df8.to_csv(csvfile, index=False, header=False)
#%%

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# GBM model
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X_train, y_train)
y_pred_gbm = gbm.predict(X_test)
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)
print(f"GBM Accuracy: {accuracy_gbm:.4f}")

# LightGBM model
lgbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
print(f"LightGBM Accuracy: {accuracy_lgbm:.4f}")
