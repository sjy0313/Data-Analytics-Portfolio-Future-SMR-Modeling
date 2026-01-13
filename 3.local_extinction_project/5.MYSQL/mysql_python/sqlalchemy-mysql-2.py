# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:19:22 2024

@author: Solero
"""

#%%

# pip install sqlalchemy
# pip install pymysql

#%%

import pandas as pd
from sqlalchemy import create_engine

#%%
    
# Database connection URL format
# DATABASE_URL = 'mysql+pymysql://username:password@localhost:3306/database'
DATABASE_URL = 'mysql+pymysql://shin:1234@192.168.71.233:3306/extinction'
engine = create_engine(DATABASE_URL)

#%%

# SQL query definition
sql = "SELECT * FROM extinction.integrated_data"

data = pd.read_sql_query(sql, engine)

print(data)
#%%
#192.168.71.233