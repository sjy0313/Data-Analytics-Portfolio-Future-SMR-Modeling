# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:11:39 2024

@author: Shin
"""
import pandas as pd
import matplotlib.pylot as plt

df1 = pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/worst_copy.xlsx")
df2 = pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/best_copy.xlsx")

# 서울시 대장아파트와 싼아파트로 부터 인프라 까지의 거리
# 대장아파트
'''
Subway               425
Primary_School       586
Middle_School        447
High_School          747
General_Hospital    1306
Supermarket          890
Park                 974
'''
# 가장 저렴한 아파트
'''
Subway               584
Primary_School       673
Middle_School        737
High_School          830
General_Hospital    1537
Supermarket         1262
Park                1593
'''


import plotly.graph_objects as go

# 데이터
infrastructure = ['Subway', 'Primary_School', 'Middle_School', 'High_School', 'General_Hospital',
                  'Supermarket', 'Park']

# 그래프 생성
fig = go.Figure(data=[
    go.Bar(name='fancy', x=infrastructure, y=[425, 586, 447, 747, 1306, 890, 974]),
    go.Bar(name='cheap', x=infrastructure, y=[584, 673, 737, 830, 1537, 1262, 1593])
])

# 막대 그래프 모드 변경
fig.update_layout(barmode='group')

# 그래프 표시
fig.show()
