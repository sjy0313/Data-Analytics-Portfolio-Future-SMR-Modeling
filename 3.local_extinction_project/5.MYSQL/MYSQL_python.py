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
# Extracting the cursor
cur = conn.cursor()
# Creating a table
# AUTO_INCREMENT Automatically numbers IDs.
cur.execute("DROP TABLE IF EXISTS items")
cur.execute('''CREATE TABLE items (
item_id INTEGER PRIMARY KEY AUTO_INCREMENT, 
name TEXT,
price INTEGER) ''')
conn.commit()
# Add data

datum = [("Mango", 7700), ("Kiwi",4000), ("Grape", 8000), ('Banana', 4000)]
# For one item (row 1) of data:
cur.executemany("INSERT INTO items(name,price) VALUES(%s, %s)", datum)
conn.commit()
#%%
cur = conn.cursor()
cur.execute("INSERT INTO items(name,price) VALUES(?,?)", datum)
#%%
cur.execute("SELECT * FROM items")
for row in cur.fetchall():
    print(row)
# Commit is required only for delete/update, not required for select.
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