# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:17:12 2024

@author: Shin
"""

# Print out the three pieces of information below from the Kyobo Bookstore Annual Best Salad page.
# Book title/author/one line review
#  Web-Scrapping Tools
#   -BeautifulSoup 
#   -selenium 
#   -chrome webdriver 
#   -pandas

# The code below is a module that converts and extracts the title/author/one-line review of the top 100 best salads into an Excel file.
# Variable line: 65 (Page 1: 1st to 50th place list)
# 66 (Page 2: List of 51st to 100th)
# 74 (file name conversion)

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