# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:41:48 2024

@author: Shin
"""

# Open the website you want to access using Selenium’s web driver.
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup 

driver = Chrome() # Create a Chrome Driver object by specifying options

html = driver.page_source 
soup = BeautifulSoup(html, "lxml")
url = "https://search.kyobobook.co.kr/search?keyword=9788997575169&gbCode=TOT&target=total" # Specify URL

driver.get(url)                    # Launch a web browser and access the specified URL.
driver.implicitly_wait(3)          # Wait for website content to be received

print("- 접속한 웹 사이트의 제목:", driver.title) # Print the title of the accessed website
print("- 접속한 웹 사이트의 URL:", driver.current_url) # Print the URL of the accessed website

pd = soup.find("a", attrs = {"class":"tag"})
print(pd)