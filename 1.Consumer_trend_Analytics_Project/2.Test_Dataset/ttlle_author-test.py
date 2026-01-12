# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:14:19 2024

@author: 신정윤
"""
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup 
import time

def parser(url):
    """Parse the given URL and return the BeautifulSoup object."""
    options = Options()
    options.headless = False  # Show browser GUI (for debugging)
    options.add_argument("start-maximized")  # Maximize browser window
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    # Scroll down to load the page
    screen_height = driver.execute_script("return window.screen.height;")
    for _ in range(8):  # Scroll down 8 times
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
    
    # Get the page source
    html_text = driver.page_source
    driver.quit()
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(html_text, 'html.parser')
    print(f'Scraping: {soup.title.text}')
    return soup

def extract_product_data(soup):
    """Extract product (book) data from the parsed HTML soup."""
    product_data = []

    for product in soup.find_all("div", attrs={'id': 'gridItemRoot', 'class': 'a-column a-span12 a-text-center _p13n-zg-list-grid-desktop_style_grid-column__2hIsc'}):
        name_elem = product.find("div", attrs={"class": "_p13n-zg-list-grid-desktop_truncationStyles_p13n-sc-css-line-clamp-1__1Fn1y"})
        author_elem = product.find("div", attrs={"class": "a-row a-size-small"})
        
        if name_elem and author_elem:
            product_data.append({
                'Product': name_elem.text.strip(),
                'Author': author_elem.text.strip()
            })
    
    return pd.DataFrame(product_data)

def main():
    # Main bestseller page URLs
    main_url = 'https://www.amazon.com/gp/bestsellers/2023/books'
    page2_url = 'https://www.amazon.com/gp/bestsellers/2023/books/ref=zg_bs_pg_2/139-1234567-1234567?ie=UTF8&pg=2'

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