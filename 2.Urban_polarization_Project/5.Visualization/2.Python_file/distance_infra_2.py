#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:




