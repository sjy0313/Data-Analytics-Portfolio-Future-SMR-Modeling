# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:24:00 2024

@author: Shin
"""

import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup 
def web_scroll(url):
    
    options = Options()
    options.headless = False  # GUI 웹 구현 
    options.add_argument('--window-size=968,1056') # 절반크기 화면 
    driver = webdriver.Chrome(options=options)
    driver.get(url) 
    time.sleep(3) # 웹 로드
    step = 0.9 #웹 페이지의 90%만큼 이동
    scroll = 8 # 총 8번이 스크롤 될 동안 실행
    screen_size = driver.execute_script("return window.screen.height;") # 1056pixel
    while scroll> 0:
        driver.execute_script("window.scrollTo(0,{screen_height}*{step})".format(screen_height=screen_size, step=step))
        step += 0.9
        step+= 0.9
        time.sleep(3) 
        scroll -= 1
    html_text = driver.page_source #웹페이지의 소스코드(html) python에 가져오기
    driver.close() 
    soup = BeautifulSoup(html_text,'lxml') # lxml 파서는 큰 html문서처리에 용이(반면에 html_parser는 간단한 문서처리에 활용)
    return soup
#%%

# 책 품목에서 제목/작가/한줄평 추출
def extract_product_data(soup):
  
    product_data = []

    for product in soup.find_all(attrs = {'class':"prod_item"}):
        name_elem = product.find('a', attrs={'class':'prod_info'})
        author_elem = product.find("span", attrs={"class": "prod_author"})
        shortreview_elem = product.find('span', attrs={"class":"review_quotes_text font_size_xxs"})
        
        if name_elem and author_elem:
            product_data.append({
                'Product': name_elem.text.strip(), # 책의 양쪽 공백제거(데이터의 일관성유지 및 처리과정에서 발생할 수 있는 오류 미연에 방지)
                'Author': author_elem.text.strip(),
                'shortreview': shortreview_elem.text.strip()
            })
    
    return pd.DataFrame(product_data)

link1 = 'https://product.kyobobook.co.kr/bestseller/total?period=004#?page=1&per=50&period=004&ymw=&bsslBksClstCode=A'
link2 = 'https://product.kyobobook.co.kr/bestseller/total?period=004#?page=2&per=50&period=004&ymw=&bsslBksClstCode=A'

main_soup1 = web_scroll(link1)
df_main1 = extract_product_data(main_soup1) 
main_soup2 = web_scroll(link2)
df_main2 = extract_product_data(main_soup2) 
df_features = pd.concat([df_main1, df_main2], ignore_index=True)
#df_features  = df_main1.append(df_main2)
import pandas as pd
directory_loc = './project/book_info.xlsx'
df_features.to_excel(directory_loc, index=False)

#%%
# 장르와 위에서 web_scroll 함수를 활용해 도출한 다른 요소들(author/product/shortreview)의 결합

# MP-Genre-final.py 에서만든 장르 dataframe 가져오기

import pandas as pd
df_features = pd.read_excel('./project/book_info.xlsx')
df_genre = pd.read_excel('./Project/Genrelist_of_bestseller2023.xlsx')
# 열 기준 병합하기

df_bestseller2023 = pd.concat([df_features, df_genre], axis=1)

# 몇가지 고유의 장르들이 2023년 배스트샐러에 채택 되었는지
genres = df_genre["장르"].unique()
print(len(genres)) # 14
# 요약정보
df_bestseller2023.describe()
'''
         Product                                Author shortreview   장르
count        100                                   100         100  100
unique       100                                    99           8   14
top     세이노의 가르침  David Cho ·  해커스어학연구소   · 2023.07.24        도움돼요   소설
freq           1                                     2          48   22
'''
# 총 8종류의 한줄평(shortreview)과 14종류의 장르가 존재함을 알 수 있고
# 48개의 (도움돼요)한줄평으로 가장많은 횟수를 차지했으며
# 22권의 소설책 확인되었다 직접 확인해보자.
#%%
# 장르 별 bestseller책들이 차지하는 비중을 구해보자.
# 시리즈 객체의 고유값개수를 세는데 사용 : value_counts() 매서드
df_bestseller2023['장르'].value_counts() 
'''
장르
소설         22
경제/경영      17
자기계발       15
인문         14
외국어         6
시/에세이       5
어린이(초등)     5
과학          5
만화          4
역사/문화       2
정치/사회       2
건강          1
컴퓨터/IT      1
청소년         1
Name: count, dtype: int64
'''




