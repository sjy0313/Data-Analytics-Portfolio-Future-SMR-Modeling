# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:44:51 2024

@author: Shin
"""

import pandas as pd

df = pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/Edu_infra.xlsx")  
csv_file_path = "D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May//Edu_infra.csv"
df.to_csv(csv_file_path, index=False)




    
   

   
   
