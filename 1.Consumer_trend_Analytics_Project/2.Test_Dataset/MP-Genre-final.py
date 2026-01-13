# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:33:45 2024

@author: Shin
"""
# To extract product code (product code exists in html)
import pandas as pd
df = pd.read_excel("./source/data/bestseller_books_2023.xlsx") # Import Kyobo Bookstore Excel file
df.head()

#%%
# Convert details page links from 1st to 100th to list data type.
# glist = Convert sales product ID to list data type
glist = df['판매상품ID'].tolist()
# Detailed page format
# [https://product.kyobobook.co.kr/detail/] + ['Sales Product ID']
# genre_link(glist): Function to complete 100 detailed page links
# genre_data: Links to 100 detailed pages

def genre_link(glist):
    genre_data = []
    sampleurl = "https://product.kyobobook.co.kr/detail/"
    for pid in glist:
        genre_data.append(sampleurl + str(pid))
    return genre_data

glist = df['판매상품ID'].tolist()
genre_data = genre_link(glist)
print(genre_data)
#%%

# To prevent spyder overload, 20 detailed page links are cut off.
# Convert s_list to a list of 20 items each (list within list)
split_data = [
    genre_data[0:20],
    genre_data[20:40],
    genre_data[40:60],
    genre_data[60:80],
    genre_data[80:100]
]
print(split_data)
#%%
# Extract genre from 0 to 20 detail pages (extracted as dict data type)
# genre_list1: Contains information about genres (0 to 20) on the detail page []
# chunk1: 20 detail page links
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 

driver = Chrome() 

genre_list1 = []
chunk1 = split_data[0]

for url in chunk1:
    driver.get(url) 
    driver.implicitly_wait(3) # 3 seconds wait (web load)
            
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
            
    genre_elements = soup.find_all('a', attrs={'class': 'btn_sub_depth'})
    # Configuration of 4 component values
    if len(genre_elements) >= 2: # When there are two or more components
        second_genre = genre_elements[1].text.strip() # Extract the second value from element values ​​after removing spaces
        genre_list1.append({'장르': second_genre})

driver.quit()

#%%  
# Genre information for 21 to 40 detailed page links
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 
from selenium.webdriver.chrome.options import Options 

# Process work within Python by creating a Chrome option object
chrome_options = Options()
chrome_options.add_argument("--headless") 
driver = Chrome(options=chrome_options)

genre_list2 = []
chunk2 = split_data[1]


for url in chunk2:
    driver.get(url)
    driver.implicitly_wait(3)
            
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
            
    genre_elements = soup.find_all('a', attrs={'class': 'btn_sub_depth'})
    if len(genre_elements) >= 2: 
        second_genre = genre_elements[1].text.strip()
        genre_list2.append({'장르': second_genre})

driver.quit()

#%%
# Genre information for 41 to 60 detailed page links
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")
driver = Chrome(options=chrome_options)

genre_list3 = []
chunk3 = split_data[2]

for url in chunk3:
    driver.get(url)
    driver.implicitly_wait(3)
            
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
            
    genre_elements = soup.find_all('a', attrs={'class': 'btn_sub_depth'})
    if len(genre_elements) >= 2: 
        second_genre = genre_elements[1].text.strip()
        genre_list3.append({'장르': second_genre})

driver.quit()

#%%
# Genre information for 61 to 80 detailed page links
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")
driver = Chrome(options=chrome_options)

genre_list4 = []
chunk4 = split_data[3]

for url in chunk4:
    driver.get(url)
    driver.implicitly_wait(3)
            
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
            
    genre_elements = soup.find_all('a', attrs={'class': 'btn_sub_depth'})
    if len(genre_elements) >= 2: 
        second_genre = genre_elements[1].text.strip()
        genre_list4.append({'장르': second_genre})

driver.quit()

#%%
# Genre information for 81 to 100 detailed page links
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")
driver = Chrome(options=chrome_options)


genre_list5 = []
chunk5 = split_data[4]

for url in chunk5:
    driver.get(url)
    driver.implicitly_wait(3)
            
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
            
    genre_elements = soup.find_all('a', attrs={'class': 'btn_sub_depth'})
    if len(genre_elements) >= 2: 
        second_genre = genre_elements[1].text.strip()
        genre_list5.append({'장르': second_genre})

driver.quit()

#%%

import pandas as pd
combined_data = genre_list1 + genre_list2 + genre_list3 + genre_list4 + genre_list5
Genre_dataframe = pd.DataFrame(combined_data)

print(Genre_dataframe)
'''
        장르
0      자기계발
1      자기계발
2      자기계발
3        소설
4        소설
..      ...
95    시/에세이
96     자기계발
97  어린이(초등)
98    경제/경영
99      청소년

[100 rows x 1 columns]
'''

Genre_dataframe.to_excel('./Project/Genrelist_of_bestseller2023.xlsx', index=False)



