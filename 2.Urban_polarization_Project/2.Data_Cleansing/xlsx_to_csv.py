# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:49:43 2024

@author: Shin
"""
'''
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Gangwondo_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Gwangju_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Gyeongbuk_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Gyeonggi_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Gyeongnam_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Incheon_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Jeju_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Jeonbuk_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Jeonnam_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Sejong_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Seoul_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Ulsan_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Busan_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Chungbuk_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Chungnam_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Daegu_mv.xlsx",
"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/Daejeon_mv.xlsx"
'''

#%%
   
import pandas as pd

city_mv = ['Gangwondo_mv', 'Gwangju_mv', 'Gyeongbuk_mv', 'Gyeonggi_mv', 'Gyeongnam_mv', 'Incheon_mv',
           'Jeju_mv', 'Jeonbuk_mv', 'Jeonnam_mv', 'Sejong_mv', 'Seoul_mv', 'Ulsan_mv', 'Busan_mv', 'Chungbuk_mv',
           'Chungnam_mv', 'Daegu_mv', 'Daejeon_mv'] 

csv_file_paths = []

for list_city in city_mv:
    
    df = pd.read_excel(f"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/{list_city}.xlsx")  
    # When converting to a csv file, 0 in October is deleted and x + 0 to prevent 2 duplicate data.
    df['Date'] = df['Date'].astype(str).map(lambda x: x + '0' if len(x) < 7 else x)
    csv_file_path = f"D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city//csvfd/{list_city}.csv"
    df.to_csv(csv_file_path, index=False, header=False)
    csv_file_paths.append(csv_file_path)


print(csv_file_paths)
