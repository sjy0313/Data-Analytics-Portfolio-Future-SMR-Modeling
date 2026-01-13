# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:03:52 2024

@author: Shin
"""

from selenium import webdriver

# Create Chrome web driver
driver = webdriver.Chrome()

# Check the size and position of the current window
print("현재 창의 크기 및 위치:", driver.get_window_position(), driver.get_window_size())

# Maximize the browser window
driver.maximize_window()

# Check the size and position of a maximized window
print("최대화된 창의 크기 및 위치:", driver.get_window_position(), driver.get_window_size())

# Close the browser window
driver.quit()


window_position = {'x': -8, 'y': -8} # x: x coordinate of the upper left corner / y: y coordinate of the upper right corner
window_size = {'width': 1936, 'height': 1056} # Window width and height information

# Split the horizontal and vertical size of the current window in half
half_width = window_size['width'] // 2
half_height = window_size['height'] // 2

# Result output
print("절반 크기:", half_width, "x", half_height)