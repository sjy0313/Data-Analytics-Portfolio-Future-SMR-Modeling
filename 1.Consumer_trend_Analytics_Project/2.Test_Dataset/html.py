# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:41:48 2024

@author: Shin
"""

# 접근하려는 웹사이트 셀레니움의 웹드라이버로 열기.
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 

driver = Chrome() # 옵션을 지정해 크롬 드라이버의 객체 생성

html = driver.page_source 
soup = BeautifulSoup(html, "lxml")
url = "https://search.kyobobook.co.kr/search?keyword=9788997575169&gbCode=TOT&target=total" # URL 지정

driver.get(url)                    # 웹 브라우저를 실행해 지정한 URL에 접속
driver.implicitly_wait(3)          # 웹 사이트의 내용을 받아오기까지 기다림

print("- 접속한 웹 사이트의 제목:", driver.title) # 접속한 웹 사이트의 제목 출력
print("- 접속한 웹 사이트의 URL:", driver.current_url) # 접속한 웹 사이트의 URL 출력

pd = soup.find("a", attrs = {"class":"tag"})
print(pd)