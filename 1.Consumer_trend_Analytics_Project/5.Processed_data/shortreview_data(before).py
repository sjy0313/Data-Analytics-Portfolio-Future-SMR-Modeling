# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:27:07 2024

@author: 신정윤
"""
import pandas as pd
df1 = pd.read_excel('C:/Users/신정윤/Documents/Python/새 폴더/MYSELF24/DataAnalysis/project/book_info(2020).xlsx')
df2 = pd.read_excel('C:/Users/신정윤/Documents/Python/새 폴더/MYSELF24/DataAnalysis/project/book_info(2021).xlsx')
df3 = pd.read_excel('C:/Users/신정윤/Documents/Python/새 폴더/MYSELF24/DataAnalysis/project/book_info(2022).xlsx')
df4 = pd.read_excel('C:/Users/신정윤/Documents/Python/새 폴더/MYSELF24/DataAnalysis/project/book_info(2023).xlsx')

df1 
sum(df1.duplicated(subset=['shortreview'])) # Number of duplicate rows: 95
sr1 = df1['shortreview'].value_counts().to_dict()
sr1_1 = pd.Series(sr1)
sre1 = pd.DataFrame([sr1_1])
sr2 = df2['shortreview'].value_counts().to_dict()
sr2_2 = pd.Series(sr2)
sre2 = pd.DataFrame([sr2_2])
sr3 = df3['shortreview'].value_counts().to_dict()
sr3_3 = pd.Series(sr3)
sre3 = pd.DataFrame([sr3_3])
sr4 = df4['shortreview'].value_counts().to_dict()
sr4_4 = pd.Series(sr4)
sre4 = pd.DataFrame([sr4_4])

series = [sre1, sre2, sre3, sre4]
results = pd.concat(series, join='outer')
results.index = range(2020, 2024)
results = results.rename_axis('연도')


results.to_excel('C:/Users/신정윤/Documents/Python/새 폴더/MYSELF24/DataAnalysis/shortreview_stat.xlsx', index=True)
