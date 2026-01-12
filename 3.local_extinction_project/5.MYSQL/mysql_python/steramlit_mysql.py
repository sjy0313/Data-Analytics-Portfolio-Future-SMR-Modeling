# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:10:58 2024

@author: Shin
"""

import streamlit as st
import pandas as pd
import mysql.connector

# Retrieve secrets from the secrets.toml file
username = st.secrets["mysql"]["root"]
password = st.secrets["mysql"]["1234"]
host = st.secrets["mysql"]["localhost"]
port = st.secrets["mysql"]["3306"]
database = st.secrets["mysql"]["extinction"]

# Establish a connection to the database
conn = mysql.connector.connect(
    user=username,
    password=password,
    host=host,
    port=port,
    database=database
)

# Execute a query and display the results
df = pd.read_sql("SELECT * FROM extinction.cofog_education", conn)
st.dataframe(df)

# Close the connection
conn.close()

#%%
import streamlit as st

conn = st.connection("extinction")  
df = conn.query("SELECT * FROM extinction.cofog_health")
st.dataframe(df)
#%%
#.streamlit 폴더 안에 secrets.toml 파일을 생성합니다.
#secrets.toml 파일에 데이터베이스 연결 정보를 아래와 같이 추가
#[connections.extinction]
host = "localhost"
port = "3306"
database = "extinction"
username = "root"
password = "1234"
#%%
import streamlit as st
import pandas as pd
import pymysql

# secrets.toml 파일의 정보를 가져옴
connection_details = st.secrets["connections"]["extinction"]

# 데이터베이스 연결 설정
connection = pymysql.connect(
    host=connection_details["host"],
    user=connection_details["username"],
    password=connection_details["password"],
    database=connection_details["database"],
    port=int(connection_details["port"])
)

# 쿼리 실행
query = "SELECT * FROM extinction.cofog_health"
df = pd.read_sql(query, connection)

# 데이터프레임 표시
st.dataframe(df)

