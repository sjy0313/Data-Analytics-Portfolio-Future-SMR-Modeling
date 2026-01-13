# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:20:21 2024

@author: Shin
"""
# User classification

import pandas as pd

# columns_name: Age / Budget / Number of family members / Preferred infrastructure
# Create an example dataframe
data = [
    [20, 15000, 1, 'Subway', 'Supermarket'],
    [30, 30000, 3, 'Primary_school', 'General_hospital'],
    [40, 50000, 2, 'Supermarket', 'Subway'],
    [50, 100000, 2, 'High School', 'park'],
    [60, 65000, 4, 'General_hospital', 'Supermarket'],
    [70, 45000, 2, 'Park', 'Subway'],
    [80, 30000, 2, 'General_hospital', 'Supermarket'],
    [90, 200000, 1, 'park', 'Supermarket']
]

index = ['신정윤님', '박주찬님', '오승필님', '성우진님', '이의재님', '파이썬님', '딥러닝님', '판다스님']

columns = ['age', 'budget', 'family_member', 'preference-1', 'preference-2']

information = pd.DataFrame(data, index=index, columns=columns)

information.to_excel('D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/user.xlsx')
#%%

information[information['budget'] <= 50000]
'''  
                 age  budget family_member
      신정윤님   20   15000             1
      박주찬님   30   30000             2
      오승필님   40   50000             4
      파이썬님   70   45000             2
      딥러닝님   80   30000             1'''

rank3 = information[(information['budget'] > 50000) & (information['budget'] < 100000)]
print(rank3)
'''
'          age  budget family_member
이의재님   60   65000             5'''
information[information['budget'] >= 100000]
'''
          age  budget family_member
성우진님   50  100000             3
판다스님   90  200000             3'''



information.to_excel('D:/WORKSPACE/github/MYSELF24/Python/MiniProject-May/seoulcity/user.xlsx')
