                                                                        # -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:02:07 2024

@author: 신정윤
"""

'''
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By

driver = Chrome()
driver.set_window_size(800, 600)
driver.get("https://www.barnesandnoble.com/b/bestsellers/_/N-2uwh")


#https://www.barnesandnoble.com/w/onyx-storm-rebecca-yarros/1145190835?ean=9781649374189
import requests

isbn = 9781649374189   
url = 'https://www.barnesandnoble.com/w/onyx-storm-rebecca-yarros/1145190835?ean={}'

r = requests.get(url.format(isbn))
'''
'''
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
def parser(url):
    
    options = Options()
    options.headless = False  # hide GUI
    options.add_argument("start-maximized")  # ensure window is full-screen
    driver = webdriver.Chrome(options=options)
    driver.get(url) # makes a request to the URL using chrome driver.
    time.sleep(3) #waits for the page to load
    step = 0.9
    scroll = 8
    screen_size = driver.execute_script("return window.screen.height;")
    while scroll> 0:
        driver.execute_script("window.scrollTo(0,{screen_height}*{i});".format(screen_height = screen_size, i = step))
        step+= 0.9
        time.sleep(3)
        scroll -= 1
    html_text = driver.page_source #grabbing the source code
    driver.close()
    soup = BeautifulSoup(html_text,'lxml') # parsing the HTML code through the LXML parser.
    print(f'Scraping: {soup.title.text}')
    return soup
'''
#%%

# Open the website you want to access using Selenium’s web driver.
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 

driver = Chrome() # Create a Chrome Driver object by specifying options

html = driver.page_source 
soup = BeautifulSoup(html, "lxml")
url = "https://www.amazon.com/gp/bestsellers/2023/books/ref=zg_bsar_cal_ye" # Specify URL

driver.get(url)                    # Launch a web browser and access the specified URL.
driver.implicitly_wait(3)          # Wait for website content to be received

print("- 접속한 웹 사이트의 제목:", driver.title) # Print the title of the accessed website
print("- 접속한 웹 사이트의 URL:", driver.current_url) # Print the URL of the accessed website
import requests
html = requests.get("https://www.amazon.com/gp/bestsellers/2023/books/ref=zg_bsar_cal_ye").text
html[0:100]

pd = soup.find("div", class_="p13n-gridRow _cDEzb_grid-row_3Cywl")
print(pd)
#%%
import pandas as pd
import numpy as np
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup 
def parser(url):
    """This function takes the URL sent and makes a request to the website to grab the HTML source code.
    This is done using selenium and chrome driver. It also scrolls through the webiste to load the page.
    It is then parsed to BeautifulSoup and returns a soupp object"""
    options = Options()
    options.headless = False  # hide GUI
    options.add_argument("start-maximized")  # ensure window is full-screen
    driver = webdriver.Chrome(options=options)
    driver.get(url) # makes a request to the URL using chrome driver.
    time.sleep(3) #waits for the page to load
    step = 0.9
    scroll = 8
    screen_size = driver.execute_script("return window.screen.height;")
    while scroll> 0:
        driver.execute_script("window.scrollTo(0,{screen_height}*{i});".format(screen_height = screen_size, i = step))
        step+= 0.9
        time.sleep(3)
        scroll -= 1
    html_text = driver.page_source #grabbing the source code
    driver.close()
    soup = BeautifulSoup(html_text,'lxml') # parsing the HTML code through the LXML parser.
    print(f'Scraping: {soup.title.text}')
    return soup

#%%

def features(soup, genre):

  product_data = []
  for product in soup.find_all("div", attrs={'id': 'gridItemRoot', 'class': 'a-column a-span12 a-text-center _cDEzb_grid-column_2hIsc'}):
      name = product.find("div", attrs={"class": "_cDEzb_p13n-sc-css-line-clamp-1_1Fn1y"})
      author = product.find("div", attrs={"class": "a-row a-size-small"})
      
      product_data.append({
      'Product': name.text if name else "NA",
      'Author': author.text.strip() if author else "NA"
     
    })
#%%

def genre_features(soup):
  """Extracts genre names and links from the main bestseller page.

  Args:
      soup (BeautifulSoup): The parsed HTML content of the main bestseller page.

  Returns:
      pandas.DataFrame: A DataFrame containing genre names and links.
  """
  genre_data = []
  for genre_item in soup.find_all("div", attrs={"role": "treeitem", "class": "_p13n-zg-nav-tree-all_style_zg-browse-item__1rdKf _p13n-zg-nav-tree-all_style_zg-browse-height-large__1z5B8"}):
      
    genre = genre_item.find("a")
    main_links = genre_item.find("a")
    if genre is not None:
      genre_data.append({"Genre": genre.text})
    else:
      genre_data.append({"Genre": "NA"})

    if main_links is not None:
      genre_data.append({"Links": 'https://amazon' + main_links.get('href')})
    else:
      genre_data.append({"Links": "NA"})

  df = pd.DataFrame(genre_data)
  df['Genre'] = df['Genre'].str.replace('_nav_books_', '')
  df_links_sub = df.iloc[1:]
  df_links = df_links_sub.copy()
  df_links['Page2'] = df_links.Links.str.replace('_nav_books_1', '_pg_2_books?ie=UTF8&pg=2')

  return df_links

def main():
    # Existing code...

    url1 = 'https://www.amazon.com/gp/bestsellers/2023/books/ref=zg_bsar_cal_ye'
    url2 = 'https://www.amazon.com/gp/bestsellers/2023/books/ref=zg_bsar_pg_2_books/ref=zg_bsar_pg_2_books?ie=UTF8&pg=2'

    s1 = parser(url1)
    s2 = parser(url2)
    df1 = pd.DataFrame(features(s1, "All"))
    df2 = pd.DataFrame(features(s2, "All"))
    
    df_final = pd.concat([df1, df2], ignore_index=True)  # Concatenate df1 and df2
    
    genre_links_df = genre_features(s1)
    #links1 = genre_links_df.Page1.tolist()
    #links2 = genre_links_df.Page2.tolist()
    genre = genre_links_df.Genre.tolist()

    # Loop through different genres
    for row in range(len(genre)):
        #s1 = parser(links1[row])
        #s2 = parser(links2[row])
        df1 = pd.DataFrame(features(s1, genre[row]))
        df_final = pd.concat([df_final, df1], ignore_index=True)  # Concatenate df_final with df1
        df2 = pd.DataFrame(features(s2, genre[row]))
        df_final = pd.concat([df_final, df2], ignore_index=True)  # Concatenate df_final with df2

    # Select desired columns in final DataFrame
    df_final = df_final[['Genre', 'Product', 'Author', 'Links']]
    
    # Save final DataFrame to CSV
    df_final.to_csv('Amazon_Bestseller_books.csv', index=False)

if __name__ == '__main__': main()
 












