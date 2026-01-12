# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:03:52 2024

@author: Shin
"""

from selenium import webdriver

# Chrome 웹 드라이버 생성
driver = webdriver.Chrome()

# 현재 창의 크기와 위치 확인
print("현재 창의 크기 및 위치:", driver.get_window_position(), driver.get_window_size())

# 브라우저 창을 최대화
driver.maximize_window()

# 최대화된 창의 크기와 위치 확인
print("최대화된 창의 크기 및 위치:", driver.get_window_position(), driver.get_window_size())

# 브라우저 창 닫기
driver.quit()


window_position = {'x': -8, 'y': -8} # x: 왼쪽 상단 모퉁이의 x좌표/ y: 오른쪽 상단 모통이의 y좌표
window_size = {'width': 1936, 'height': 1056} # 창의 너비와 높이 정보

# 현재 창의 가로 및 세로 크기를 반으로 나누기
half_width = window_size['width'] // 2
half_height = window_size['height'] // 2

# 결과 출력
print("절반 크기:", half_width, "x", half_height)