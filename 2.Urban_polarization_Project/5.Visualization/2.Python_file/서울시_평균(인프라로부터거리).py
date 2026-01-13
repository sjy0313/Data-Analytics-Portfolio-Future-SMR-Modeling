# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:11:39 2024

@author: Shin
"""
import pandas as pd
import matplotlib.pylot as plt

df1 = pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/worst_copy.xlsx")
df2 = pd.read_excel("D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/best_copy.xlsx")

# Distance from Daejang Apartment and Cheap Apartment in Seoul to infrastructure
# Daejang Apartment
'''
Subway               425
Primary_School       586
Middle_School        447
High_School          747
General_Hospital    1306
Supermarket          890
Park                 974
'''
# cheapest apartment
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

# data
infrastructure = ['Subway', 'Primary_School', 'Middle_School', 'High_School', 'General_Hospital',
                  'Supermarket', 'Park']

# Graph creation
fig = go.Figure(data=[
    go.Bar(name='fancy', x=infrastructure, y=[425, 586, 447, 747, 1306, 890, 974]),
    go.Bar(name='cheap', x=infrastructure, y=[584, 673, 737, 830, 1537, 1262, 1593])
])

# Change bar graph mode
fig.update_layout(barmode='group')

# graph display
fig.show()
