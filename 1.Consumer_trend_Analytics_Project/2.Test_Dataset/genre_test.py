# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:29:57 2024

@author: Shin
"""
# If the Excel file is not being loaded, check the current directory:
import os
print(os.getcwd())

import pandas as pd
df = pd.read_excel("./source/data/bestseller_books_2023.xlsx") # Import Kyobo Bookstore Excel file
df.head()

#%%
df_id = df[['판매상품ID']]
# glist = Convert sales product ID to list data type
glist = df['판매상품ID'].tolist()
print(glist)


pid1 = df_id.iloc[0].values[0]  
pid1
url2 = f"https://product.kyobobook.co.kr/detail/{pid1}"  # URL generation

print(url2) # https://product.kyobobook.co.kr/detail/S000200746091

#%%

from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 

driver = Chrome() # Create a Chrome Driver object by specifying options

driver.get(url2)                    # Launch a web browser and access the specified URL.
driver.implicitly_wait(3)          # Wait for website content to be received

html = driver.page_source 
soup = BeautifulSoup(html, "lxml")

print("- 접속한 웹 사이트의 제목:", driver.title) # Print the title of the accessed website
print("- 접속한 웹 사이트의 URL:", driver.current_url) # Print the URL of the accessed website

#%%
# The code below contains too much information. You need to find elements located further inside.
#soup.select(".container_wrapper .breadcrumb_item") 

s1 =  soup.find('a', attrs={'class': 'btn_sub_depth'}) # Prints the first value of the <a> element, 'domestic book'.
print(s1) # <a class="btn_sub_depth" href="https://product.kyobobook.co.kr/KOR">국내도서</a>
topic = soup.find_all('a', attrs={'class': 'btn_sub_depth'}) # Prints all attribute values ​​of <a> elements

# Property value html content (extract the element's content using BeautifulSoup's 'text' property)
genre = []
for t in topic:
    genre.append(t.text)
    print(t.text)

print(genre) # ['Domestic books', 'Self-development', 'Success/Life', 'Self-management/Life']
genre[1] # ‘Self-development’
#%%

import pandas as pd
df = pd.read_excel("./source/data/bestseller_books_2023.xlsx") # Import Kyobo Bookstore Excel file
df.head()


def genre_link(glist):
    genre_data = []
    sampleurl = "https://product.kyobobook.co.kr/detail/"
    for pid in glist:
        genre_data.append(sampleurl + str(pid))
    return genre_data

glist = df['판매상품ID'].tolist()
genre_data = genre_link(glist)
print(genre_data)

print(genre_data)

split_data = [
    genre_data[0:20],
    genre_data[20:40],
    genre_data[40:60],
    genre_data[60:80],
    genre_data[80:100]
]
print(split_data)
#%%
# Data comparison values ​​are processed internally in Python once the data is derived.
# Genre information for 81 to 100 detailed page links
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 



driver = Chrome() 

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

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup 

chrome_options = Options()
chrome_options.add_argument("--headless")

genre_list6 = []
chunk6 = split_data[4]
driver = Chrome(options=chrome_options)


for url in chunk6:
            driver.get(url)
            driver.implicitly_wait(3)
            
            html = driver.page_source
            soup = BeautifulSoup(html, "lxml")
            
            genre_elements = soup.find_all('a', attrs={'class': 'btn_sub_depth'})
            if len(genre_elements) >= 2:  # Ensure there are at least 2 elements
                # Extract the text of the second element
                second_genre = genre_elements[1].text.strip()
                genre_list6.append({'장르': second_genre})
            
driver.quit()
    
    




 
  


