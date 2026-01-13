# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:19:22 2024

@author: Solero
"""

#%%
#[connections.my_database]
# Example using QLAlchemy

# pip install sqlalchemy
# pip install pymysql

#%%

# Define database model: Define the model that will store student information.
from sqlalchemy import create_engine, Column, Integer, String
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

Base = declarative_base()
# table layout [Write a query in the class in create table format]
class Hello(Base):
    __tablename__ = 'hello'

    hid = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

    def __repr__(self):
        return f"<Hello(hid={self.hid}, name='{self.name}', age={self.age})>"
    
#%%
    
# Connect to database and create table: Connect to database and create table.

# Create SQLite database file
DATABASE_URL = 'mysql+pymysql://root:solsql@localhost:3306/hellodb'
engine = create_engine(DATABASE_URL)

# Create table
Base.metadata.create_all(engine)

#%%

# Inserting and querying data: Create a session to insert and query data.

# Create session
Session = sessionmaker(bind=engine)
session = Session()

# Insert data
new_data = Hello(hid=9999, name='구구구', age=99)
session.add(new_data)

session.commit()

#%%

# Data inquiry
hellos = session.query(Hello).all()
for hello in hellos:
    print(hello)

# Session ends
session.close()

#%%

#  THE END