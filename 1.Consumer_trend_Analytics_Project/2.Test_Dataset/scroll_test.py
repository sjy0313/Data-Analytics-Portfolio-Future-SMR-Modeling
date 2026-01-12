# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:25:46 2024

@author: Shin
"""
# 컴퓨터 화면 크기 파악.(아래 코드는 현재 학원컴퓨터 사이즈를 적용한 것임.)
from selenium import webdriver
driver = webdriver.Chrome()
driver.maximize_window()
print("최대화된 창의 크기 및 위치:", driver.get_window_position(), driver.get_window_size())
driver.quit()
# 최대화된 창의 크기 및 위치: {'x': -8, 'y': -8} {'width': 1936, 'height': 1056}
# {왼쪽 상단 모퉁이의 x좌표, y: 오른쪽 상단 모통이의 y좌표} {창의 너비와 높이}
window_size = {'width': 1936, 'height': 1056}
half_width = window_size['width'] // 2

# 절반크기로 브라우저 크기 축소시켜서 web_scroll 구현
print("절반 크기:", half_width)
# 절반 크기: 968 

#%%
# web-scrapping에 활용할 함수생성

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
        shortre_elem = product.find('span', attrs={"class":"review_quotes_text font_size_xxs"})
        
        if name_elem and author_elem:
            product_data.append({
                'Product': name_elem.text.strip(), # 책의 양쪽 공백제거(데이터의 일관성유지 및 처리과정에서 발생할 수 있는 오류 미연에 방지)
                'Author': author_elem.text.strip(),
                'shortre': shortre_elem.text.strip()
            })
    
    return pd.DataFrame(product_data)
# 중간점검 
'''
main_url = 'https://product.kyobobook.co.kr/bestseller/total?period=004#?page=1&per=50&period=004&ymw=&bsslBksClstCode=A'
main_soup = web_scroll(main_url)
df_main = extract_product_data(main_soup) 
df = pd.concat([df_main], ignore_index=True)
df.to_csv('test_books.csv', index=False)
'''
#%%
# 책의 장르 추출(p137 혼자공부하는 데이터 분석)

import requests
import pandas as pd


excel_file = " ./bestseller_books_2023.xlsx"

df = pd.read_excel(excel_file)

book_id = 



def genre_features(soup):
    url = https://product.kyobobook.co.kr/detail/{}
    
  
  genre_data = []
  for genre_item in soup.find_all("ul", attrs={"class":"tabs swiper-wrapper ui-tabs-nav ui-corner-all ui-helper-reset ui-helper-clearfix ui-widget-header"}):
    genre = genre_item.find("span", attrs={"class":"tab_text"})
    main_links = genre_item.find('button', class_='tab_link')
    if genre is not None:
      genre_data.append({"Genre": genre.text})
    else:
      genre_data.append({"Genre": "NA"})

    if main_links is not None:
      genre_data.append({"Links": 'https://product.kyobobook.co.kr/bestseller/total?period=004#?page=1&per=50&period=004&ymw=&bsslBksClstCode' + main_links.get('href')})
    else:
      genre_data.append({"Links": "NA"})

  df = pd.DataFrame(genre_data)
  df['Genre'] = df['Genre'].str.replace('_nav_books_', '')
  df_links_sub = df.iloc[1:]
  df_links = df_links_sub.copy()
  df_links['Page2'] = df_links.Links.str.replace('_nav_books_1', '_pg_2?ie=UTF8&pg=2')

  return df_links




 if main_links is not None:
    # 'A' 메인 링크가 존재하는 경우
    base_url = "https://product.kyobobook.co.kr/bestseller/total?period=004#?page=1&per=50&period=004&ymw=&bsslBksClstCode="
    genre_data.append({"Links": base_url + "A"})
else:
    # 'A' 메인 링크가 없는 경우
    genre_data.append({"Links": "NA"})

