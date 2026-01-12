# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:50:20 2024

@author: 신정윤
"""

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
prod_list
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


