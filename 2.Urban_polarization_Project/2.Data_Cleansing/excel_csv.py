# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:03:54 2024

@author: Shin
"""

import os
import pandas as pd

# Specify Excel file path and CSV file path
excel_file_path = r'D:/WORKSPACE/github/MYSELF24/Python/best_copy.xlsx'
pd.read_ecel(xcel_file_path)
csv_file_path = r'D:/WORKSPACE/github/MYSELF24/Python/best.csv'

# Check if file exists
if os.path.isfile(excel_file_path):
    print("파일이 존재합니다.")
else:
    print("파일이 존재하지 않습니다. 경로를 다시 확인하세요.")

# Read Excel file
try:
    df = pd.read_excel('D:/WORKSPACE/github/MYSELF24/Python/best_copy.xlsx', engine='openpyxl')
    # Save data frame as CSV file
    df.to_csv('D:/WORKSPACE/github/MYSELF24/Python/best.csv', index=False)
    print("변환이 완료되었습니다!")
except FileNotFoundError:
    print(f"지정된 경로에 파일이 존재하지 않습니다: {'D:/WORKSPACE/github/MYSELF24/Python/best_copy.xlsx'}")
except Exception as e:
    print(f"다른 오류가 발생했습니다: {e}")
