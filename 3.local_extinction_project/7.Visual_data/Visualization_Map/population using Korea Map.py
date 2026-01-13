#!/usr/bin/env python
# coding: utf-8

# Analysis of areas at risk of population extinction in Korea
# ### Areas at risk of population extinction:
# 
# Using the analysis method of Lee Sang-ho, an associate researcher at the Korea Employment Information Service, who wrote the report ‘7 Analysis of ‘Local Decline’ in Korea’.
# Comparing the elderly population aged 65 or older with the female population aged 20 to 39 **If the young female population is less than half of the elderly population
# The method is to classify it as a ‘region at risk of extinction’**.
# 

#%%
# ## Obtaining and organizing population data
# KOSIS (National Statistics Portal): kosis.kr
# Population and household > Population sector > General survey population > Total survey population (2015) > Population by gender, age and household composition - City, county and district selection, items for each region and required age group

#%%

import pandas as pd
import numpy as np

import platform
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

path = "c:/Windows/Fonts/malgun.ttf"
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')    

plt.rcParams['axes.unicode_minus'] = False


#%%

population = pd.read_excel('../dataset/population_raw_data.xlsx', header=1)
population.fillna(method='pad', inplace=True)

population.rename(columns = {'행정구역(동읍면)별(1)':'광역시도', 
                             '행정구역(동읍면)별(2)':'시도', 
                             '계':'인구수'}, inplace=True)

#%%

# Delete 'Subtotal' data row from column ('Attempt')
population = population[(population['시도'] != '소계')]

population

#%%

# * Looking at the table above, you can see that the contents of the column called **Item** are divided into **Total population, male population, and female population** for each administrative district.
# * To organize this now, simply use a loop (for) to change it to **Total, Male, Female** and save it as a column called **Separation**.
# * In particular, in order to avoid copy-related warnings in subsequent data processing, it is respecified as the **.copy()** option.
# * And, we decided to delete **item**

#%%

population.is_copy = False

population.rename(columns = {'항목':'구분'}, inplace=True)

population.loc[population['구분'] == '총인구수 (명)', '구분'] = '합계'
population.loc[population['구분'] == '남자인구수 (명)', '구분'] = '남자'
population.loc[population['구분'] == '여자인구수 (명)', '구분'] = '여자'

population


#%%
# ## Calculate areas at risk of depopulation and organize data

#%%

population['20-39세'] = population['20 - 24세'] + population['25 - 29세'] + \
                        population['30 - 34세'] + population['35 - 39세']
    
population['65세이상'] = population['65 - 69세'] + population['70 - 74세'] + \
                        population['75 - 79세'] + population['80 - 84세'] + \
                        population['85 - 89세'] + population['90 - 94세'] + \
                        population['95 - 99세'] + population['100+']
            
population.head(10)


#%%

# * Using **pivot_table**, set **metropolitan cities, cities and provinces** as the index, and set the first vertical column as **division**,
# The value is organized as **Population, 20-39 years old, 65 years or older**.

pop = pd.pivot_table(population, 
                     index = ['광역시도', '시도'], 
                     columns = ['구분'],
                     values = ['인구수', '20-39세', '65세이상'])
pop


#%%

# * Apply the formula to calculate the area at risk of population extinction in the column called **Extinction Rate**.
# * If this ratio is less than 1, it can be viewed as **an area in danger of population extinction**.

pop['소멸비율'] = pop['20-39세','여자'] / (pop['65세이상','합계'] / 2)
pop.head()


#%%

# * Specify whether the area is at risk of extinction as boolean.

pop['소멸위기지역'] = pop['소멸비율'] < 1.0
pop.head()

#%%

pop[pop['소멸위기지역']==True].index.get_level_values(1)


#%%

# * With pivot_table well organized, reset the result attribute of pivot_table with **.reset_index**.

pop.reset_index(inplace=True) 
pop.head()


#%%

# * To release double columns, combine the two column titles and designate them again.

tmp_coloumns = [pop.columns.get_level_values(0)[n] + \
                pop.columns.get_level_values(1)[n] 
                for n in range(0,len(pop.columns.get_level_values(0)))]

pop.columns = tmp_coloumns

pop.head()

#%%


pop.info()


#%%

# ## Create a unique ID per region for map visualization

pop['시도'].unique()

#%%

si_name = [None] * len(pop)

tmp_gu_dict = {'수원':['장안구', '권선구', '팔달구', '영통구'], 
                       '성남':['수정구', '중원구', '분당구'], 
                       '안양':['만안구', '동안구'], 
                       '안산':['상록구', '단원구'], 
                       '고양':['덕양구', '일산동구', '일산서구'], 
                       '용인':['처인구', '기흥구', '수지구'], 
                       '청주':['상당구', '서원구', '흥덕구', '청원구'], 
                       '천안':['동남구', '서북구'], 
                       '전주':['완산구', '덕진구'], 
                       '포항':['남구', '북구'], 
                       '창원':['의창구', '성산구', '진해구', '마산합포구', '마산회원구'], 
                       '부천':['오정구', '원미구', '소사구']}


#%%

for n in pop.index:
    if pop['광역시도'][n][-3:] not in ['광역시', '특별시', '자치시']:
        if pop['시도'][n][:-1]=='고성' and pop['광역시도'][n]=='강원도':
            si_name[n] = '고성(강원)'
        elif pop['시도'][n][:-1]=='고성' and pop['광역시도'][n]=='경상남도':
            si_name[n] = '고성(경남)'
        else:
             si_name[n] = pop['시도'][n][:-1]
                
        for keys, values in tmp_gu_dict.items():
            if pop['시도'][n] in values:
                if len(pop['시도'][n])==2:
                    si_name[n] = keys + ' ' + pop['시도'][n]
                elif pop['시도'][n] in ['마산합포구','마산회원구']:
                    si_name[n] = keys + ' ' + pop['시도'][n][2:-1]
                else:
                    si_name[n] = keys + ' ' + pop['시도'][n][:-1]
        
    elif pop['광역시도'][n] == '세종특별자치시':
        si_name[n] = '세종'
        
    else:
        if len(pop['시도'][n])==2:
            si_name[n] = pop['광역시도'][n][:2] + ' ' + pop['시도'][n]
        else:
            si_name[n] = pop['광역시도'][n][:2] + ' ' + pop['시도'][n][:-1]


#%%

si_name


#%%

# * For use in map visualization, specify the unique name of the administrative district created in the above process as ID.

pop['ID'] = si_name


# In[17]:


del pop['20-39세남자']
del pop['65세이상남자']
del pop['65세이상여자']

pop.head()


#%%

# ## Make a map of our country with Cartogram

#%%

# draw_korea_raw = pd.read_excel('../data/draw_korea_raw.xlsx', encoding="EUC-KR")
draw_korea_raw = pd.read_excel('../dataset/draw_korea_raw.xlsx')
draw_korea_raw


#%%

# * Now, use the **.stack()** command as the opposite of pivot_table to obtain the on-screen coordinates of each administrative district.

draw_korea_raw_stacked = pd.DataFrame(draw_korea_raw.stack())
draw_korea_raw_stacked.reset_index(inplace=True)
draw_korea_raw_stacked.rename(columns={'level_0':'y', 'level_1':'x', 0:'ID'}, 
                              inplace=True)

draw_korea_raw_stacked


#%%
# * Reset the index again...
# * Reset the column name.

draw_korea = draw_korea_raw_stacked


#%%

# * First, when marking on the map in the ID column, separate the lines by city name and district name.

BORDER_LINES = [
    [(5, 1), (5,2), (7,2), (7,3), (11,3), (11,0)], # Incheon
    [(5,4), (5,5), (2,5), (2,7), (4,7), (4,9), (7,9), 
     (7,7), (9,7), (9,5), (10,5), (10,4), (5,4)], # seoul
    [(1,7), (1,8), (3,8), (3,10), (10,10), (10,7), 
     (12,7), (12,6), (11,6), (11,5), (12, 5), (12,4), 
     (11,4), (11,3)], # gyeonggi-do
    [(8,10), (8,11), (6,11), (6,12)], # Gangwon-do
    [(12,5), (13,5), (13,4), (14,4), (14,5), (15,5), 
     (15,4), (16,4), (16,2)], # Chungcheongbuk-do
    [(16,4), (17,4), (17,5), (16,5), (16,6), (19,6), 
     (19,5), (20,5), (20,4), (21,4), (21,3), (19,3), (19,1)], # Jeollabuk-do
    [(13,5), (13,6), (16,6)], # Daejeon
    [(13,5), (14,5)], # Sejong City
    [(21,2), (21,3), (22,3), (22,4), (24,4), (24,2), (21,2)], # gwangju
    [(20,5), (21,5), (21,6), (23,6)], # Jeollanam-do
    [(10,8), (12,8), (12,9), (14,9), (14,8), (16,8), (16,6)], # Chungcheongbuk-do
    [(14,9), (14,11), (14,12), (13,12), (13,13)], # Gyeongsangbuk-do
    [(15,8), (17,8), (17,10), (16,10), (16,11), (14,11)], # daegu
    [(17,9), (18,9), (18,8), (19,8), (19,9), (20,9), (20,10), (21,10)], # busan
    [(16,11), (16,13)], # Ulsan
#     [(9,14), (9,15)], 
    [(27,5), (27,6), (25,6)],
]

#%%


plt.figure(figsize=(8, 11))

# Show region name
for idx, row in draw_korea.iterrows():
    
    # In metropolitan cities, district names often overlap, so city-level names are also displayed.
    # (Jung-gu, Seo-gu)
    if len(row['ID'].split())==2:
        dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])
    elif row['ID'][:2]=='고성':
        dispname = '고성'
    else:
        dispname = row['ID']

    # If the name has more than 3 characters, such as Seodaemun-gu or Seogwipo-si, it is displayed in small letters.
    if len(dispname.splitlines()[-1]) >= 3:
        fontsize, linespacing = 9.5, 1.5
    else:
        fontsize, linespacing = 11, 1.2

    plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                 fontsize=fontsize, ha='center', va='center', 
                 linespacing=linespacing)
    
# Draw city and city boundaries.
for path in BORDER_LINES:
    ys, xs = zip(*path)
    plt.plot(xs, ys, c='black', lw=1.5)

plt.gca().invert_yaxis()
#plt.gca().set_aspect(1)

plt.axis('off')

plt.tight_layout()
plt.show()


#%%

# * Let’s check if there are any problems with the contents of the ID column, which is the key to be used when combining data from pop, which is the result of population analysis, and draw_korea, which is used to draw a map.

set(draw_korea['ID'].unique()) - set(pop['ID'].unique())


#%%

set(pop['ID'].unique()) - set(draw_korea['ID'].unique())


# * According to the results above, you can see that there are more data on cities with administrative districts in pop.
# * It cannot be displayed on the map anyway, so delete it.

#%%

tmp_list = list(set(pop['ID'].unique()) - set(draw_korea['ID'].unique()))

for tmp in tmp_list:
    pop = pop.drop(pop[pop['ID']==tmp].index)
                       
print(set(pop['ID'].unique()) - set(draw_korea['ID'].unique()))


#%%

pop.head()


#%%

# * Now, assuming that the ID columns of pop and draw_korea match, merge the ID as the key.

pop = pd.merge(pop, draw_korea, how='left', on=['ID'])

pop.head()

#%%


# * Now, if the data you want to express on the map from the pop data above is **population total**, these values ​​can be located in each administrative district created earlier.

mapdata = pop.pivot_table(index='y', columns='x', values='인구수합계')
masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)


#%%

mapdata


#%%

masked_mapdata


#%%

# * Let’s make it a function by adding a command to complete the above content and colormap.

def drawKorea(targetData, blockedMap, cmapname):
    gamma = 0.75

    whitelabelmin = (max(blockedMap[targetData]) - 
                                     min(blockedMap[targetData]))*0.25 + \
                                                                min(blockedMap[targetData])

    datalabel = targetData

    vmin = min(blockedMap[targetData])
    vmax = max(blockedMap[targetData])

    mapdata = blockedMap.pivot_table(index='y', columns='x', values=targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)
    
    plt.figure(figsize=(9, 11))
    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=cmapname, 
               edgecolor='#aaaaaa', linewidth=0.5)

    # Show region name
    for idx, row in blockedMap.iterrows():
        # In metropolitan cities, district names often overlap, so city-level names are also displayed.
        # (Jung-gu, Seo-gu)
        if len(row['ID'].split())==2:
            dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])
        elif row['ID'][:2]=='고성':
            dispname = '고성'
        else:
            dispname = row['ID']

        # If the name has more than 3 characters, such as Seodaemun-gu or Seogwipo-si, it is displayed in small letters.
        if len(dispname.splitlines()[-1]) >= 3:
            fontsize, linespacing = 10.0, 1.1
        else:
            fontsize, linespacing = 11, 1.

        annocolor = 'white' if row[targetData] > whitelabelmin else 'black'
        plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                     fontsize=fontsize, ha='center', va='center', color=annocolor,
                     linespacing=linespacing)

    # Draw city and city boundaries.
    for path in BORDER_LINES:
        ys, xs = zip(*path)
        plt.plot(xs, ys, c='black', lw=2)

    plt.gca().invert_yaxis()

    plt.axis('off')

    cb = plt.colorbar(shrink=.1, aspect=10)
    cb.set_label(datalabel)

    plt.tight_layout()
    plt.show()


#%%

# ## Check population status and depopulated areas

#%%

drawKorea('인구수합계', pop, 'Blues')


#%%

# * Expression of areas at risk of population extinction


pop['소멸위기지역'] = [1 if con else 0 for con in pop['소멸위기지역']]
drawKorea('소멸위기지역', pop, 'Reds')


#%%

# ## Check the percentage of female population in the population status

def drawKorea(targetData, blockedMap, cmapname):
    gamma = 0.75

    whitelabelmin = 20.

    datalabel = targetData

    tmp_max = max([ np.abs(min(blockedMap[targetData])), 
                                  np.abs(max(blockedMap[targetData]))])
    vmin, vmax = -tmp_max, tmp_max

    mapdata = blockedMap.pivot_table(index='y', columns='x', values=targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)
    
    plt.figure(figsize=(9, 11))
    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=cmapname, 
               edgecolor='#aaaaaa', linewidth=0.5)

    # Show region name
    for idx, row in blockedMap.iterrows():
        # In metropolitan cities, district names often overlap, so city-level names are also displayed.
        # (Jung-gu, Seo-gu)
        if len(row['ID'].split())==2:
            dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])
        elif row['ID'][:2]=='고성':
            dispname = '고성'
        else:
            dispname = row['ID']

        # If the name has more than 3 characters, such as Seodaemun-gu or Seogwipo-si, it is displayed in small letters.
        if len(dispname.splitlines()[-1]) >= 3:
            fontsize, linespacing = 10.0, 1.1
        else:
            fontsize, linespacing = 11, 1.

        annocolor = 'white' if np.abs(row[targetData]) > whitelabelmin else 'black'
        plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                     fontsize=fontsize, ha='center', va='center', color=annocolor,
                     linespacing=linespacing)

    # Draw city and city boundaries.
    for path in BORDER_LINES:
        ys, xs = zip(*path)
        plt.plot(xs, ys, c='black', lw=2)

    plt.gca().invert_yaxis()

    plt.axis('off')

    cb = plt.colorbar(shrink=.1, aspect=10)
    cb.set_label(datalabel)

    plt.tight_layout()
    plt.show()


#%%


pop.head()


#%%

pop['여성비'] = (pop['인구수여자']/pop['인구수합계'] - 0.5)*100
drawKorea('여성비', pop, 'RdBu')


#%%

pop['2030여성비'] = (pop['20-39세여자']/pop['20-39세합계'] - 0.5)*100
drawKorea('2030여성비', pop, 'RdBu')


#%%

# ## Representing areas at risk of depopulation in Folium

#%%

pop_folium = pop.set_index('ID')
pop_folium.head()


#%%


import folium
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#%%

geo_path = '../dataset/05. skorea_municipalities_geo_simple.json'
geo_str = json.load(open(geo_path, encoding='utf-8'))

#%%

"""
map = folium.Map(location=[36.2002, 127.054], zoom_start=7)
map.choropleth(geo_data = geo_str,
               data = pop_folium['인구수합계'],
               columns = [pop_folium.index, pop_folium['인구수합계']],
               fill_color = 'YlGnBu', #PuRd, YlGnBu
               key_on = 'feature.id')
map
"""

#%%

from folium import Choropleth

map = folium.Map(location=[36.2002, 127.054], zoom_start=7)

choropleth = Choropleth(geo_data = geo_str,
               data = pop_folium['인구수합계'],
               columns = [pop_folium.index, pop_folium['인구수합계']],
               fill_color = 'YlGnBu', #PuRd, YlGnBu
               key_on = 'feature.id').add_to(map)
map

#%%

"""
map = folium.Map(location=[36.2002, 127.054], zoom_start=7)
map.choropleth(geo_data = geo_str,
               data = pop_folium['소멸위기지역'],
               columns = [pop_folium.index, pop_folium['소멸위기지역']],
               fill_color = 'PuRd', #PuRd, YlGnBu
               key_on = 'feature.id')

map
"""

#%%

map = folium.Map(location=[36.2002, 127.054], zoom_start=7)
choropleth = Choropleth(geo_data = geo_str,
               data = pop_folium['소멸위기지역'],
               columns = [pop_folium.index, pop_folium['소멸위기지역']],
               fill_color = 'PuRd', #PuRd, YlGnBu
               key_on = 'feature.id').add_to(map)

map


#%%

draw_korea.to_csv("../dataset/05. draw_korea.csv", encoding='utf-8', sep=',')

#%%

# THE END



