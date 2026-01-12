# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:19:22 2024

@author: Solero
"""

#%%
#[connections.my_database]
# QLAlchemy를 사용한 예제

# pip install sqlalchemy
# pip install pymysql

#%%

# 데이터베이스 모델 정의: 학생 정보를 저장할 모델을 정의합니다.
from sqlalchemy import create_engine, Column, Integer, String
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

Base = declarative_base()
# table layout [class에 create table 형식으로 쿼리작성]
class Hello(Base):
    __tablename__ = 'hello'

    hid = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

    def __repr__(self):
        return f"<Hello(hid={self.hid}, name='{self.name}', age={self.age})>"
    
#%%
    
# 데이터베이스 연결 및 테이블 생성: 데이터베이스에 연결하고 테이블을 생성합니다.

# SQLite 데이터베이스 파일 생성
DATABASE_URL = 'mysql+pymysql://root:solsql@localhost:3306/hellodb'
engine = create_engine(DATABASE_URL)

# 테이블 생성
Base.metadata.create_all(engine)

#%%

# 데이터 삽입 및 조회: 세션을 생성하여 데이터를 삽입하고 조회합니다.

# 세션 생성
Session = sessionmaker(bind=engine)
session = Session()

# 데이터 삽입
new_data = Hello(hid=9999, name='구구구', age=99)
session.add(new_data)

session.commit()

#%%

# 데이터 조회
hellos = session.query(Hello).all()
for hello in hellos:
    print(hello)

# 세션 종료
session.close()

#%%

#  THE END