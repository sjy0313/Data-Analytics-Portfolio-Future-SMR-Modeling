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
        shortreview_elem = product.find('span', attrs={"class":"review_quotes_text font_size_xxs"})
        
        if name_elem and author_elem:
            product_data.append({
                'Product': name_elem.text.strip(), # Remove spaces on both sides of the book (maintain data consistency and prevent errors that may occur during processing)
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
# Combination of genre and other elements (author/product/shortreview) derived using the web_scroll function above

# Import genre dataframe created in MP-Genre-final.py

import pandas as pd
df_features = pd.read_excel('./project/book_info.xlsx')
df_genre = pd.read_excel('./Project/Genrelist_of_bestseller2023.xlsx')
# Merge by column

df_bestseller2023 = pd.concat([df_features, df_genre], axis=1)

# How many unique genres have been selected for Best Salary in 2023?
genres = df_genre["장르"].unique()
print(len(genres)) # 14
# Summary information
df_bestseller2023.describe()
'''
         Product                                Author shortreview   장르
count        100                                   100         100  100
unique       100                                    99           8   14
top     세이노의 가르침  David Cho ·  해커스어학연구소   · 2023.07.24        도움돼요   소설
freq           1                                     2          48   22
'''
# You can see that there are a total of 8 types of shortreviews and 14 types of genres.
# It took up the most number of comments with 48 (helpful) one-line comments.
# 22 novels have been confirmed. Let’s check them out for ourselves.
#%%
# Let’s find the proportion of bestseller books by genre.
# Used to count the number of unique values ​​of a series object: value_counts() method
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




