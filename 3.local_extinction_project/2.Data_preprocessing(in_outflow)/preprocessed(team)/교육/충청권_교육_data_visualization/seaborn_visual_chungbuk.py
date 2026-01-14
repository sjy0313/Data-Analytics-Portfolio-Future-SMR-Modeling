# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:26:13 2024

@author: Shin
"""

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import seaborn as sns


# Font settings
font_path = "c:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
#%%
#city = ['result_chungbuk', 'chungbuk_pop']

file_path = f"D:/WORKSPACE/github/MYSELF24/Python/Final_project/chungbuk_data/result_chungbuk.csv"
edu = pd.read_csv(file_path, header=None)
edu = edu.iloc[:, 1]
# edu.columns = ['Education Infrastructure (Wed)']
edu = edu.rename('교육인프라(수)')

#%% 

file_path = f"D:/WORKSPACE/github/MYSELF24/Python/Final_project/chungbuk_data/chungbuk_pop.csv"
pop = pd.read_csv(file_path)
pop = pop.iloc[:, 1]
# pop.columns = ['Population(Number)']
pop = pop.rename('인구(수)')
#%%
chungbuk = pd.concat([edu, pop],axis=1)
#%%

fig = plt.figure(figsize=(10,5))
#ax1 = fig.add_subplot(1,2,1)
#ax2 = fig.add_subplot(1,2,2)

sns.regplot(x='인구(수)', y='교육인프라(수)', data=chungbuk) #ax= ax1
plt.title('충북 : 교육인프라와 인구관계도')

plt.savefig(f"D:/WORKSPACE/github/MYSELF24/Python/Final_project/Visual_data/충북/pop_edu_reg.png")
plt.show()
# sns.regplot(x='Population (number)', y='Education Infrastructure (number)', data=chungbuk, ax= ax2, fit_reg=False)


# sns.jointplot(x='Population (number)', y='Education Infrastructure (number)', kind='reg', data=chungbuk,)
#sns.pairplot(chungbuk)

#%%
fig = plt.figure(figsize=(10,5))
#ax1 = fig.add_subplot(1,2,1)
#ax2 = fig.add_subplot(1,2,2)

sns.regplot(x='교육인프라(수)', y='인구(수)', data=chungbuk) #ax= ax1
plt.title('충북 : 교육인프라와 인구관계도')

plt.savefig(f"D:/WORKSPACE/github/MYSELF24/Python/Final_project/Visual_data/충북/edu_pop_reg.png")
plt.show()




