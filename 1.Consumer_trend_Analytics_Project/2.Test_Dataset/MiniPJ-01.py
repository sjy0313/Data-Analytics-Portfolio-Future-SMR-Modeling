# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 22:49:34 2024

@author: 신정윤
"""

import pandas as pd

# Import Excel file
df = pd.read_excel("./data/bestseller_books_2023.xlsx")

# Determine the type by extracting the unique value of the “genre” column
genres = df["분야"].unique()
print(len(genres)) # 20
 
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 

driver = Chrome() # Create an object of Chrome Driver by specifying options

url = "https://product.kyobobook.co.kr/bestseller/total?period=004# ?page =1&per=20&perd=004&04 &mw=&bsslBkstCode=A" # URL .
driver.get(url)                    # Launch a web browser and access the specified URL.
driver.implicitly_wait(3)          # Wait for website content to be received

html = driver.page_source          # After connecting, retrieve the HTML code of the page.
soup = BeautifulSoup(html, 'lxml') # Parse HTML code

print("- 접속한 웹 사이트의 제목:", driver.title) # Print the title of the accessed website
print("- 접속한 웹 사이트의 URL:", driver.current_url) # Print the URL of the accessed website

soup1 = BeautifulSoup(html, 'html.parser')
soup2 = soup.find_all(attrs = {'class':"prod_item"})
print(soup2)
pd_link = soup.find('a', attrs={'class':'prod_info'})
print(pd_link)

print(pd_link['href'])
# https://product.kyobobook.co.kr/detail/S000200746091

soup3 = soup1.find('a', attrs={'class':'prod_info'})
print(soup3)

soup4 = soup1.find("span", attrs={"class": "prod_author"})
print(soup4)



#%%
import pandas as pd
import time
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
    soup = BeautifulSoup(html_text,'lxml')
    return soup



def extract_product_data(soup):
  
    product_data = []

    for product in soup.find_all(attrs = {'class':"prod_item"}):
        name_elem = product.find('a', attrs={'class':'prod_info'})
        author_elem = product.find("span", attrs={"class": "prod_author"})
        if name_elem and author_elem:
            product_data.append({
                'Product': name_elem.text.strip(),
                'Author': author_elem.text.strip()
            })
    
    return pd.DataFrame(product_data)

def main():
    # Main bestseller page URLs
    main_url = 'https://product.kyobobook.co.kr/bestseller/total?period=004#?page=1&per=50&period=004&ymw=&bsslBksClstCode=A'
    page2_url = 'https://product.kyobobook.co.kr/bestseller/total?period=004#?page=2&per=50&period=004&ymw=&bsslBksClstCode=A'

    # Parse main and page 2 URLs
    main_soup = parser(main_url)
    page2_soup = parser(page2_url)

    # Extract product data from main and page 2 soups
    df_main = extract_product_data(main_soup)
    df_page2 = extract_product_data(page2_soup)

    # Concatenate data frames
    df_all = pd.concat([df_main, df_page2], ignore_index=True)

    # Save to CSV
    df_all.to_csv('Amazon_Bestseller_books.csv', index=False)

if __name__ == '__main__':
    main()






