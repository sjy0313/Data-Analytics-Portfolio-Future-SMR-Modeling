# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:28:47 2024

@author: Shin
"""

import pandas as pd

# file list
files = ['유치원통합', '초등학교통합', '중학교통합', '고등학교통합']

# Initialize an empty data frame dictionary to store data by year
annual_data = {str(year): pd.DataFrame() for year in range(2015, 2022)}

# Read data from each file and separate data by year
for file in files:
    file_path = f"C:/Users/Shin/Documents/Final_Project/Data/교육_전국/학급당학생수/학급당학생수_{file}.xlsx"
    teacher = pd.read_excel(file_path)
    
    for year in range(2015, 2022):
        year_str = str(year)
        
        if year_str in teacher.columns:
            # Merge data frames
            if annual_data[year_str].empty:
                annual_data[year_str] = teacher[[year_str]]
            else:
                annual_data[year_str] = pd.concat([annual_data[year_str], teacher[[year_str]]], axis=1)
        else:
            print(f"열 {year_str}가 파일 {file}에 없습니다.")

# Save data by year
output_base_path = "C:/Users/Shin/Documents/Final_Project/Data/교육_전국/학급당학생수/"

for year in range(2015, 2022):
    year_str = str(year)
    output_file_path = f"{output_base_path}학급당_{year}_전국.xlsx"
    if not annual_data[year_str].empty:
        annual_data[year_str].to_excel(output_file_path, index=False)
        print(f"Saved {output_file_path}")
    else:
        print(f"No data for year {year_str}.")

