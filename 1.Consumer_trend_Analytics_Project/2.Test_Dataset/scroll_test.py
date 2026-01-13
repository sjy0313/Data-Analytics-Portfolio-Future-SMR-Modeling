# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:25:46 2024

@author: Shin
"""
# Determine the computer screen size. (The code below applies the current academy computer size.)
from selenium import webdriver
driver = webdriver.Chrome()
driver.maximize_window()
print("최대화된 창의 크기 및 위치:", driver.get_window_position(), driver.get_window_size())
driver.quit()
# Size and position of maximized window: {'x': -8, 'y': -8} {'width': 1936, 'height': 1056}
# {x-coordinate of the upper-left corner, y: y-coordinate of the upper-right corner} {width and height of window}
window_size = {'width': 1936, 'height': 1056}
half_width = window_size['width'] // 2

# Implement web_scroll by reducing the browser size to half the size
print("절반 크기:", half_width)
# Half size: 968

#%%
# Create a function to be used for web-scraping

import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup 
def web_scroll(url):
    
    options = Options()
    options.headless = False  # GUI web implementation
    options.add_argument('--window-size=968,1056') # half size screen
    driver = webdriver.Chrome(options=options)
    driver.get(url) 
    time.sleep(3) # web load
    step = 0.9 # Move as much as 90% of a web page
    scroll = 8 # Executes while scrolling a total of 8 times
    screen_size = driver.execute_script("return window.screen.height;") # 1056pixel
    while scroll> 0:
        driver.execute_script("window.scrollTo(0,{screen_height}*{step})".format(screen_height=screen_size, step=step))
        step += 0.9
        step+= 0.9
        time.sleep(3) 
        scroll -= 1
    html_text = driver.page_source # Import web page source code (html) into python
    driver.close() 
    soup = BeautifulSoup(html_text,'lxml') # The lxml parser is easy to process large HTML documents (on the other hand, html_parser is used for simple document processing).
    return soup


#%%
# Extract title/author/one-line review from book items
def extract_product_data(soup):
  
    product_data = []

    for product in soup.find_all(attrs = {'class':"prod_item"}):
        name_elem = product.find('a', attrs={'class':'prod_info'})
        author_elem = product.find("span", attrs={"class": "prod_author"})
        shortre_elem = product.find('span', attrs={"class":"review_quotes_text font_size_xxs"})
        
        if name_elem and author_elem:
            product_data.append({
                'Product': name_elem.text.strip(), # Remove spaces on both sides of the book (maintain data consistency and prevent errors that may occur during processing)
                'Author': author_elem.text.strip(),
                'shortre': shortre_elem.text.strip()
            })
    
    return pd.DataFrame(product_data)
# Midterm inspection
'''
main_url = 'https://product.kyobobook.co.kr/bestseller/total?period=004#?page=1&per=50&period=004&ymw=&bsslBksClstCode=A'
main_soup = web_scroll(main_url)
df_main = extract_product_data(main_soup) 
df = pd.concat([df_main], ignore_index=True)
df.to_csv('test_books.csv', index=False)
'''
#%%
# Extracting the genre of a book (p137 data analysis for solo study)

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
    # If 'A' main link exists
    base_url = "https://product.kyobobook.co.kr/bestseller/total?period=004#?page=1&per=50&period=004&ymw=&bsslBksClstCode="
    genre_data.append({"Links": base_url + "A"})
else:
    # If there is no 'A' main link
    genre_data.append({"Links": "NA"})

