# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:20:32 2024

@author: Shin
"""

import sqlite3
# Connecting to sqlite database---(*1)
filepath = "test2.sqlite"
conn = sqlite3.connect(filepath)

# Create a table and insert data ---(*2)
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS items")
cur.execute('''CREATE TABLE items (
item_id INTEGER PRIMARY KEY,
name TEXT,
price INTEGER) ''')
conn.commit()
# Enter data ---(*3)
cur = conn.cursor() # Since the purpose of creating the cursor is different, it must be executed separately each time.
# Select only name and price as columns.
cur.execute(
    "INSERT INTO items (name,price) VALUES (?,?)",
    ("Orange", 5200))
conn.commit()

# Inserting multiple data in succession ---(*4)
# Import data using the data sql statement Import it as csv
# Automatic collection of learning data (modularized by building a database with scenarios for each age group)
# Scenario: By age (20s, 30s, 40s, 50s, 60s)
# - Virtual character
'''
A조합 : 20 ~ 30대
- 선호 인프라 : 교통, 여가
- 자산 보유 현황 : 5천만원 ~ 1억 원
(1) 김 : "나는 직장인이고, 지하철역이 가까운 곳을 선호하고, 가끔 스크린골프나 헬스장을 가는 것이 좋다"
(2) 이 : "나는 개인사업자이고, 가게 바로 옆(번화가)에 집이 있었으면 좋겠고, 혼자 있는 시간이 많아 등산을 다니고 싶다"
(3) 박 : "나는 프리랜서이고, 가끔 고객미팅이 있어서 지하철역이 가까운 곳을 선호하고, 가끔 술 마실 수 있으면 좋겠다"

B조합 : 30 ~ 40대
- 선호 인프라 : 학교, 교통, 여가
- 자산 보유 현황 : 2억 원 ~ 5억 원
(1) 오 : "나는 기혼자고 우리 애기가 2살인데, 유치원이랑 가까운 곳을 선호하고, 애기가 뛰어놀 수 있는 놀이터와 커뮤니티 시설이 잘 되어 있었으면 좋겠다"
(2) 성 : "나는 솔로고, 맛집과 술집이 많은 동네에 살고 싶다"
(3) 장 : "나는 부모님과 같이 사는데, 부모님이 편찮으셔서 내가 병원까지 모셔다 드려야 해서, 병원과 가까운 곳이었으면 좋겠다"

C조합 : 40 ~ 50대
- 선호 인프라 : 학교, 병원, 교통
- 자산 보유 현황 : 6억 원 ~ 10억 원
(1) 심 : "맞벌이 가정이고, 우리 애기가 중학생인데, 중학교와 가까웠으면 좋겠고, 가끔가다가 집근처에서 외식할 수 있는 밥집이 가까운 곳이면 좋겠다"
(2) 배 : "외벌이 가정이고, 남편 / 와이프가 프리랜서로 활동해서, 교통이 편리한 곳이었으면 좋겠다"
(3) 하 : "부모님을 모시고 사는데, 가끔 편찮으실 때는 병원에 모셔다 드려야 해서, 병원이랑 가까운 곳이면 좋겠다"

D조합 : 50 ~ 60대
- 선호 인프라 : 병원, 공원
- 자산 보유 현황 : 11억 원 ~ 20억 원
(1) 추 : "애들은 다 독립하고, 남편(와이프)가 게이트볼 구장이 가까운 XX공원 옆에 살았으면 좋겠다고 해서, 그 쪽으로 옮기고 싶다. 평수는 클 필요가 없고, 그냥 2명이서 살기에 적합한 평수면 된다."
(2) 조 : "몸이 아픈 솔로고, 혼자서 병원을 가야하는데 매우 힘들어서, 병원에 걸어서 갈 수 있고, 길목에 오르막길과 내리막길이 없었으면 좋겠다"
(3) 서 : "남편 / 와이프와 사별한 상태고, 몸은 건강한 편이라서 도심지역에 주말농장과 가까운 곳에 있었으면 좋겠다"
'''
cur = conn.cursor()
data = [("Mango", 7700), ("Kiwi",4000), ("Grape", 8000), 
        ("Peach", 9400), ("Persimmon", 7000),('Banana', 4000)]
cur.executemany(
    "INSERT INTO items (name,price) VALUES (?,?)",
    data)
conn.commit()

# Extract data between 4,000 and 7,000 won---(*5)
cur = conn.cursor()
price_range = (4000, 7000)
cur.execute(
    "SELECT * FROM items WHERE price>=? AND price<=?",
    price_range)
cur.execute("SELECT item_id,name,price FROM items ORDER BY price DESC LIMIT 5")
fr_list = cur.fetchall()
for fr in fr_list:
    print(fr)
    
