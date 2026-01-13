# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:25:27 2024

@author: Shin
"""
# genre_creator is a module that extracts chapters that vary depending on the Excel file by year.
# Download the excel file from the web and enter the column corresponding to ‘Sales Product ID’ at Kyobo Bookstore.
# Combined with https://product.kyobobook.co.kr/detail/ to go to the detail page
# This is a module that extracts only the genre information from the details page and saves it as an Excel file.

# Variable line:
# 19: Saved Excel file


# To extract product code (product code exists in HTML), load Excel file:
import pandas as pd
df = pd.read_excel("./project/Genrelist_of_bestseller2022.xlsx") # Import Kyobo Bookstore Excel file

#%%
# Convert details page links from 1st to 100th to list data type.
# glist = Convert sales product ID to list data type
glist = df['판매상품ID'].tolist()
# Detailed page format
# [https://product.kyobobook.co.kr/detail/] + ['Sales Product ID']
# genre_link(glist): Function to complete 100 detailed page links
# genre_data: link to detail page

def genre_link(glist):
    genre_data = []
    sampleurl = "https://product.kyobobook.co.kr/detail/"
    for pid in glist:
        genre_data.append(sampleurl + str(pid))
    return genre_data

genre_data = genre_link(glist)

  
#%%
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 
from selenium.webdriver.chrome.options import Options 

# Process work within Python by creating a Chrome option object
chrome_options = Options()
chrome_options.add_argument("--headless") 
driver = Chrome(options=chrome_options)

genre_list = []
chunk = genre_data
url_cnt = 0

for url in chunk:
    url_cnt += 1
    print("url_cnt:", url_cnt, url)
    driver.get(url)
    driver.implicitly_wait(3)
            
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
    print('soup:', soup != None)
            
    genre_elements = soup.find_all('a', attrs={'class': 'btn_sub_depth'})
    print(len(genre_elements))
    if genre_elements != None:
        if len(genre_elements) >= 2: 
            second_genre = genre_elements[1].text.strip()
            genre_list.append({'장르': second_genre})
        else:
            genre_list.append({'장르': '절판'}) 
    else:
        genre_list.append({'장르': '절판'})
                                                                                            

driver.quit()


#%%

# convert to excel file
Genre_dataframe = pd.DataFrame(genre_list)
Genre_dataframe.to_excel('./Project/Genrelist_of_bestseller2022.xlsx', index=False)
