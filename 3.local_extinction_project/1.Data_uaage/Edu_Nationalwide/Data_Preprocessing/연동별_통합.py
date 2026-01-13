# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:38:02 2024

@author: Shin
"""

import pandas as pd

# Initialize an empty data frame dictionary to store data by year
annual_data = {str(year): pd.DataFrame() for year in range(2015, 2022)}

# file list
files = ['학생수_유치원통합', '학생수_초등학교통합']

# Number of students per teacher calculation function
def students_per_class(kin, year):
    student_col = str(year)
   
    if student_col in kin.columns:
        return kin[[student_col]]
    else:
        print(f"열 {student_col}가 데이터에 없습니다.")
        return None

# Calculate and save yearly data by file
for file in files:
    file_path = f"C:/Users/Shin/Documents/Final_Project/Data/교육_전국/유_초등학생수/{file}.xlsx"
    kin = pd.read_excel(file_path)
    
    for year in range(2015, 2022):
        result = students_per_class(kin, year)
        if result is not None:
            # Change column name format to '{year}_{file}'
            column_name = f"{year}_{file}"
            if annual_data[str(year)].empty:
                annual_data[str(year)] = result.rename(columns={str(year): column_name})
            else:
                annual_data[str(year)] = pd.concat([annual_data[str(year)], result.rename(columns={str(year): column_name})], axis=1)

# Save data by year
output_base_path = "C:/Users/Shin/Documents/Final_Project/Data/교육_전국/유_초등학생수/"

for year, df in annual_data.items():
    output_file_path = f"{output_base_path}/{year}_학생수_전국.xlsx"
    if not df.empty:
        df.to_excel(output_file_path, index=False)
        print(f"Saved {output_file_path}")
    else:
        print(f"No data for year {year}.")
