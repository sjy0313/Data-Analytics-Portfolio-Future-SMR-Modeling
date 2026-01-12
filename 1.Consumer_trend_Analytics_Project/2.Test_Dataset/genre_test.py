# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:29:57 2024

@author: Shin
"""
# 만약 액셀파일을 읽어오지 않을 떄 current directory 확인: 
import os
print(os.getcwd())

import pandas as pd
df = pd.read_excel("./source/data/bestseller_books_2023.xlsx") # 교보문고 액셀파일 불러오기
df.head()

#%%
df_id = df[['판매상품ID']]
# glist = 판매상품 아이디를 list자료형으로 변환
glist = df['판매상품ID'].tolist()
print(glist)


pid1 = df_id.iloc[0].values[0]  
pid1
url2 = f"https://product.kyobobook.co.kr/detail/{pid1}"  # URL 생성

print(url2) # https://product.kyobobook.co.kr/detail/S000200746091

#%%

from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 

driver = Chrome() # 옵션을 지정해 크롬 드라이버의 객체 생성

driver.get(url2)                    # 웹 브라우저를 실행해 지정한 URL에 접속
driver.implicitly_wait(3)          # 웹 사이트의 내용을 받아오기까지 기다림

html = driver.page_source 
soup = BeautifulSoup(html, "lxml")

print("- 접속한 웹 사이트의 제목:", driver.title) # 접속한 웹 사이트의 제목 출력
print("- 접속한 웹 사이트의 URL:", driver.current_url) # 접속한 웹 사이트의 URL 출력

#%%
# 아래 코드는 너무 정보가 많다 더 안쪽에 위치한 요소찾기 필요
#soup.select(".container_wrapper .breadcrumb_item") 

s1 =  soup.find('a', attrs={'class': 'btn_sub_depth'}) # <a> 요소의 첫 번쨰값 '국내도서' 출력.
print(s1) # <a class="btn_sub_depth" href="https://product.kyobobook.co.kr/KOR">국내도서</a>
topic = soup.find_all('a', attrs={'class': 'btn_sub_depth'}) # <a> 요소인 속성값 모두 출력

# 속성값 html내용(BeautifulSoup의 'text'속성 사용하여 요소의 콘텐츠 추출)
genre = []
for t in topic:
    genre.append(t.text)
    print(t.text)

print(genre) # ['국내도서', '자기계발', '성공/처세', '자기관리/처세']
genre[1] # '자기계발'
#%%

import pandas as pd
df = pd.read_excel("./source/data/bestseller_books_2023.xlsx") # 교보문고 액셀파일 불러오기
df.head()


def genre_link(glist):
    genre_data = []
    sampleurl = "https://product.kyobobook.co.kr/detail/"
    for pid in glist:
        genre_data.append(sampleurl + str(pid))
    return genre_data

glist = df['판매상품ID'].tolist()
genre_data = genre_link(glist)
print(genre_data)

print(genre_data)

split_data = [
    genre_data[0:20],
    genre_data[20:40],
    genre_data[40:60],
    genre_data[60:80],
    genre_data[80:100]
]
print(split_data)
#%%
# 데이터 비교 값은 자료 도출되면 python내부적으로 작업처리
# 81~100개 상세페이지링크에 대한 장르정보
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 



driver = Chrome() 

genre_list5 = []
chunk5 = split_data[4]

for url in chunk5:
    driver.get(url)
    driver.implicitly_wait(3)
            
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
            
    genre_elements = soup.find_all('a', attrs={'class': 'btn_sub_depth'})
    if len(genre_elements) >= 2: 
        second_genre = genre_elements[1].text.strip()
        genre_list5.append({'장르': second_genre})

driver.quit()

#%%

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup 

chrome_options = Options()
chrome_options.add_argument("--headless")

genre_list6 = []
chunk6 = split_data[4]
driver = Chrome(options=chrome_options)


for url in chunk6:
            driver.get(url)
            driver.implicitly_wait(3)
            
            html = driver.page_source
            soup = BeautifulSoup(html, "lxml")
            
            genre_elements = soup.find_all('a', attrs={'class': 'btn_sub_depth'})
            if len(genre_elements) >= 2:  # Ensure there are at least 2 elements
                # Extract the text of the second element
                second_genre = genre_elements[1].text.strip()
                genre_list6.append({'장르': second_genre})
            
driver.quit()
    
    




 
  


