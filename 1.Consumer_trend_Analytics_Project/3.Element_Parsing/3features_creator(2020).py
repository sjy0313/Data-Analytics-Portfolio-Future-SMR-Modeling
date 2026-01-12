# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:17:12 2024

@author: Shin
"""

# 아래 3개 정보를 교보문고 연간배스트샐러 페이지에서 뽑아 출력하기
# 책 제목/저자/한줄평 
#  Web-Scrapping Tools
#   -BeautifulSoup 
#   -selenium 
#   -chrome webdriver 
#   -pandas

# 아래 코드는 top100 배스트샐러의 제목/저자/한줄평을 액셀파일로 변환추출해주는 모듈
# 변수 line: 65(1페이지: 1~50위리스트)
#            66(2페이지: 51~100위 리스트) 
#            74(파일이름 변환)

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

link1 = 'https://product.kyobobook.co.kr/bestseller/total?period=004#?page=1&per=50&period=004&ymw=2020&bsslBksClstCode=A'
link2 = 'https://product.kyobobook.co.kr/bestseller/total?period=004#?page=2&per=50&period=004&ymw=2020&bsslBksClstCode=A'

main_soup1 = web_scroll(link1) 
df_main1 = extract_product_data(main_soup1) 
main_soup2 = web_scroll(link2)
df_main2 = extract_product_data(main_soup2) 
df_features = pd.concat([df_main1, df_main2], ignore_index=True)
#df_features  = df_main1.append(df_main2)
import pandas as pd
directory_loc = './project/book_info(2020).xlsx'
df_features.to_excel(directory_loc, index=False)