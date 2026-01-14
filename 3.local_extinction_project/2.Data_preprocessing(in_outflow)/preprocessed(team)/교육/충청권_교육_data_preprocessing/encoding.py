# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:35:14 2024

@author: Shin
"""

import pandas as pd

# CSV file path to convert
input_file_path = 'C:/Users/YS702/Desktop/LAST_PROJECT/일산화탄소_10년_RawData.csv'  # Enter actual path

# Reading into a dataframe using Pandas' read_csv function

# Read with UTF-8 encoding
#df = pd.read_csv(input_file_path, encoding='utf-8')

# Read with cp949 encoding
#df = pd.read_csv(input_file_path, encoding='cp949')

# Read with euc-kr encoding
df = pd.read_csv(input_file_path, encoding='euc-kr')

# Converted CSV file storage path
output_file_path = 'C:/Users/YS702/Desktop/LAST_PROJECT/일산화탄소_10년_RawData(euc-kr).csv'


# Save CSV file with UTF-8 encoding using to_csv function
df.to_csv(output_file_path, index=False, encoding='euc-kr')


# Data frame information output (optional)
print(df.info())