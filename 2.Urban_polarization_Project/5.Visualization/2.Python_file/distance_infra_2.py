#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:




