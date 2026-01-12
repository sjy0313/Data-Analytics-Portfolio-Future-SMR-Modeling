# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:55:20 2024

@author: Shin
"""

# MYSQL 
# pip isntall mysqlclient

import MySQLdb

conn = MySQLdb.connect(
    user='root',
    passwd='1234',
    host = 'localhost',
    db='hellodb')
# 커서 추출하기
cur = conn.cursor()
# 테이블 생성하기
#AUTO_INCREMENT 자동으로 id 번호매겨줌.
cur.execute("DROP TABLE IF EXISTS items")
cur.execute('''CREATE TABLE items (
item_id INTEGER PRIMARY KEY AUTO_INCREMENT, 
name TEXT,
price INTEGER) ''')
conn.commit()
# 데이터 추가하기

datum = [("Mango", 7700), ("Kiwi",4000), ("Grape", 8000), ('Banana', 4000)]
# 한건(1행)의 데이터에 대해서는 
cur.executemany("INSERT INTO items(name,price) VALUES(%s, %s)", datum)
conn.commit()
#%%
cur = conn.cursor()
cur.execute("INSERT INTO items(name,price) VALUES(?,?)", datum)
#%%
cur.execute("SELECT * FROM items")
for row in cur.fetchall():
    print(row)
# delete/update 에서만 commit이 필요 select에서는 필요없음.    
conn.close()
#%%
import pandas as pd 

#items_df = pd.read_sql_query("SELECT item_id,name,price FROM items", conn)

sql = "SELECT * FROM items"
cur.execute(sql)
for row in cur.fetchall():
    print(row)

items_df = pd.read_sql_query(sql, conn)
print(items_df)

conn.close()