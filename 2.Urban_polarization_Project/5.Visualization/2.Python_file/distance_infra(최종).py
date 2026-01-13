#!/usr/bin/env python
# coding: utf-8

# In[9]:


import plotly.graph_objects as go

infrastructure = ['지하철', '초등학교', '중학교', '고등학교', '종합병원',
                  '대형마트', '공원']
fig = go.Figure()
fig.add_trace(go.Bar(x=infrastructure,
                y=[425, 586, 447, 747, 1306, 890, 974],
                name='최고가 아파트',
                marker_color='rgb(255, 215, 0)'
                ))
fig.add_trace(go.Bar(x=infrastructure,
                y=[584, 673, 737, 830, 1537, 1262, 1593],
                name='최저가 아파트',
                marker_color='rgb(128, 128, 128)'
                ))

fig.update_layout(
    title='과연 아파트가 비싸면 부대시설로 부터 거리가 가까운거 맞아요?',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='거리(m)',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1, # gap between bars of the same location coordinate.
    xaxis_tickangle= 0,  # Set x-axis label slope
)
fig.show()


# In[ ]:




