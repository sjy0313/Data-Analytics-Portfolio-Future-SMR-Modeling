#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.graph_objects as go

infrastructure = ['Subway', 'Primary_School', 'Middle_School', 'High_School', 'General_Hospital',
                  'Supermarket', 'Park']
fig = go.Figure()
fig.add_trace(go.Bar(x=infrastructure,
                y=[425, 586, 447, 747, 1306, 890, 974],
                name='fancy',
                marker_color='rgb(255, 215, 0)'
                ))
fig.add_trace(go.Bar(x=infrastructure,
                y=[584, 673, 737, 830, 1537, 1262, 1593],
                name='cheap',
                marker_color='rgb(128, 128, 128)'
                ))

fig.update_layout(
    title='Are you sure about the distance?',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='distance(meter)',
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
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
fig.show()


# In[ ]:




