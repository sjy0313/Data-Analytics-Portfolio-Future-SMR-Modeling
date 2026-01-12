# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:50:03 2024

@author: Shin
"""

import pandas as pd
file_path = "D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/learning_data.csv"
df= pd.read_csv(file_path)

df.to_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/learning_data.xlsx")

