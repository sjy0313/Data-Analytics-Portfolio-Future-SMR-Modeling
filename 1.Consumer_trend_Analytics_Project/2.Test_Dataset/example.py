# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:19:29 2024

@author: 신정윤
"""
#https://github.com/evil-in/amz_bestsellers_books/blob/main/amz_books_bestseller_scraper.py
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

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

def features(soup, genre):
  """Extracts product data from a parsed BeautifulSoup object.

  Args:
      soup (BeautifulSoup): The parsed HTML content.
      genre (str): The genre category of the products.

  Returns:
      pandas.DataFrame: A DataFrame containing the extracted product data.
  """
  
  product_data = []
  for product in soup.find_all("div", attrs={'id': 'gridItemRoot', 'class': 'a-column a-span12 a-text-center _p13n-zg-list-grid-desktop_style_grid-column__2hIsc'}):
    name = product.find("div", attrs={"class": "_p13n-zg-list-grid-desktop_truncationStyles_p13n-sc-css-line-clamp-1__1Fn1y"})
    author = product.find("div", attrs={"class": "a-row a-size-small"})
    link = product.find("a", attrs={"class": "a-link-normal", "role": "link", "tabindex": "-1"})
    price = product.find("span", attrs={"class": "_p13n-zg-list-grid-desktop_price_p13n-sc-price__3mJ9Z"})
    rating = product.find("span", attrs={"class": "a-icon-alt"})
    review_count = product.find("span", attrs={"class": "a-size-small"})
    rank = product.find("span", attrs={"class": "zg-bdg-text"})

    product_data.append({
      'Ranks': rank.text.strip('#') if rank else "NA",
      'Product': name.text if name else "NA",
      'Author': author.text.strip() if author else "NA",
      'Links': 'https://amazon.in' + link.get('href') if link else "NA",
      'Price': price.text.replace('₹', '') if price else "NA",
      'Rating': rating.text[:3] if rating else "NA",
      'Reviews': re.findall("[0-9]", review_count.text) if review_count else "NA",
    })

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
      genre_data.append({"Links": 'https://amazon.in' + main_links.get('href')})
    else:
      genre_data.append({"Links": "NA"})

  df = pd.DataFrame(genre_data)
  df['Genre'] = df['Genre'].str.replace('_nav_books_', '')
  df_links_sub = df.iloc[1:]
  df_links = df_links_sub.copy()
  df_links['Page2'] = df_links.Links.str.replace('_nav_books_1', '_pg_2?ie=UTF8&pg=2')

  return df_links

def main():
    """ This function controls the execution of the code. The steps involved are:
    1. Scraping the main bestsellers page of amazon books, using the parser(url) function.
    2. Parsing the HTML code to get the data points, using the features(soup_object, genre) function
    3. Getting the genre and the links of the sub-division of bestseller pages, using the genre_features(soup_object) function
    4. Scraping each link for the data, using parser(url) and features(soup_object, genre) """

    url1 = 'https://www.amazon.com/best-sellers-books-Amazon/zgbs/books/ref=zg_bs_nav_books_0'
    
    url2 = 'https://www.amazon.com/best-sellers-books-Amazon/zgbs/books/ref=zg_bs_pg_2_books?_encoding=UTF8&pg=2'
   
    s1 = parser(url1)
    s2 = parser(url2)
    df1 = features(s1, "All")
    df2 = features(s2, "All")
    df_final = df1.append(df2)

    genre_links_df = genre_features(s1)
    links1 = genre_links_df.Page1.tolist()
    links2 = genre_links_df.Page2.tolist()
    genre = genre_links_df.Genre.tolist()

    # looping through the different genres
    for row in range(0, len(genre)):
        s1 = parser(links1[row])
        s2 = parser(links2[row])
        df1 = features(s1, genre[row])
        df_final = df_final.append(df1)
        df2 = features(s2, genre[row])
        df_final = df_final.append(df2)

    df_final = df_final[['Genre','Ranks', 'Product', 'Author', 'Links', 'Price', 'Rating','Reviews']]
    df_final.to_csv('Amazon_Bestseller_books.csv', index = False)

if __name__ == '__main__':
    main()