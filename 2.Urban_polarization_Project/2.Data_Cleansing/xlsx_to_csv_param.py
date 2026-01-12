# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:01:07 2024

@author: Shin
"""

import pandas as pd
file_path = "D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/Parameter.xlsx"
df= pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/Parameter.xlsx")
df['Supply_demands_trend'] = df['Supply_demand_trend'].round(2)
df.to_excel(file_path, index=False)
df.drop(columns=['Supply_demand_trend'], inplace=True)
df['Date'] = df['Date'].astype(str).map(lambda x: x + '0' if len(x) < 7 else x)
csv_file_path = ("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/csvfd/Parameter.csv")
df.to_csv(csv_file_path, index=False)


df1= pd.read_csv("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/city/csvfd/Parameter.csv")
df1.drop(columns=['Supply_demand_trend'], inplace=True)

