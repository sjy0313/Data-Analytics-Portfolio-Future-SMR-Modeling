# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:24:01 2024

@author: Shin
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# extinction level
file_path = 'C:/Users/Shin/Documents/Final_Project/Data/교육_전국/교육_연도별_전국통합/소멸등급(2015~2021).csv'
parameter_integrate_p = pd.read_csv(file_path)

#%%
file_paths = ['C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/교육종합_v0.2.xlsx',
            'C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/보건.xlsx',
           'C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/사회보호.xlsx',
           'C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/교통_2015~2021_전국_종합_v1.xlsx',
           'C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/주거_2015~2021_전국_종합_v1.xlsx',
          'C:/Users/Shin/Documents/Final_Project/Mysql/data/excel/행정_종합_2015~2021_v1.xlsx']


def merge_dataframes(file_paths):
    dataframes = []
    for path in file_paths:
        parameter_integrate = pd.read_excel(path)
        dataframes.append(parameter_integrate.iloc[:, 1:])  

    result = pd.concat(dataframes, axis=1)  # Merge column-wise
    return result

parameter_integrate = merge_dataframes(file_paths)
print(parameter_integrate)

# Analysis of the relationship between extinction risk level and integrated variables
model = sm.OLS(parameter_integrate_p['소멸위험등급'], parameter_integrate[:] )
results = model.fit()
print(results.summary())  
'''
OLS Regression Results 회귀분석 결과                            
=======================================================================================
Dep. Variable:                 소멸위험등급   R-squared (uncentered):             0.891
Model:                            OLS   Adj. R-squared (uncentered):              0.889
Method:                 Least Squares   F-statistic:                              312.9 F-statistic은 모델의 설명력이 통계적으로 유의미한지 검정
Date:                Wed, 31 Jul 2024   Prob (F-statistic):                        0.00 매우 낮은 p-value으로 모델이 유의미하다는 것을 강하게 시사
Time:                        09:31:13   Log-Likelihood:                         -2109.6
No. Observations:                1603   AIC:                                      4301.
parameter_integrate Residuals:                    1562   BIC:                                      4522.
parameter_integrate Model:                          41                                                  
Covariance Type:            nonrobust                                                  
=====================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------
교원_1인당_학생수_유치원        0.0260      0.021      1.247      0.213      -0.015       0.067
교원_1인당_학생수_초등학교       0.0399      0.043      0.935      0.350      -0.044       0.124
교원_1인당_학생수_중학교        0.1536      0.036      4.279      0.000       0.083       0.224
교원_1인당_학생수_고등학교       0.1153      0.030      3.901      0.000       0.057       0.173
유치원_학급당 학생 수 (명)      0.0423      0.014      2.993      0.003       0.015       0.070
초등학교_학급당 학생 수 (명)    -0.0796      0.031     -2.577      0.010      -0.140      -0.019
중학교_학급당 학생 수 (명)     -0.0776      0.019     -3.982      0.000      -0.116      -0.039
고등학교_학급당 학생 수 (명)    -0.0222      0.016     -1.431      0.153      -0.053       0.008
학교교과 교습학원 (개)        -0.0005      0.000     -5.155      0.000      -0.001      -0.000
평생직업 교육학원 (개)         0.0035      0.001      3.498      0.000       0.002       0.006
사설학원당 학생수 (명)        -0.0001      0.000     -0.744      0.457      -0.001       0.000
유치원생 수            -3.901e-06      3e-05     -0.130      0.897   -6.28e-05     5.5e-05
초등학생 수             9.568e-07   9.49e-06      0.101      0.920   -1.76e-05    1.96e-05
종합병원                 -0.0689      0.025     -2.783      0.005      -0.117      -0.020
병원                   -0.0387      0.008     -4.661      0.000      -0.055      -0.022
의원                   -0.0005      0.001     -1.018      0.309      -0.002       0.000
치과병(의)원               0.0011      0.001      0.774      0.439      -0.002       0.004
한방병원                  0.0119      0.003      3.486      0.001       0.005       0.019
한의원                   0.0035      0.002      1.951      0.051   -1.87e-05       0.007
인구 천명당 의료기관병상수(개)    -0.0038      0.003     -1.158      0.247      -0.010       0.003
총병상수 (개)           5.532e-05   2.14e-05      2.584      0.010    1.33e-05    9.73e-05
고위험음주율                0.0092      0.009      1.071      0.284      -0.008       0.026
비만율                   0.0019      0.008      0.233      0.816      -0.014       0.018
EQ.5D(건강상태 표준화)      -2.1779      0.706     -3.086      0.002      -3.562      -0.794
주관적건강수준인지율            0.0162      0.004      4.558      0.000       0.009       0.023
건강보험 적용인구 현황      -2.363e-05   4.17e-06     -5.667      0.000   -3.18e-05   -1.54e-05
운전행태영역                0.0040      0.003      1.145      0.252      -0.003       0.011
교통안전영역               -0.0025      0.004     -0.559      0.576      -0.011       0.006
보행행태영역               -0.0133      0.012     -1.079      0.281      -0.037       0.011
1인당 자동차등록대수          -0.7825      0.181     -4.315      0.000      -1.138      -0.427
도시지역면적             -6.21e-10   3.63e-10     -1.709      0.088   -1.33e-09    9.19e-11
주택 수              -1.651e-05   3.15e-06     -5.239      0.000   -2.27e-05   -1.03e-05
출생아수              -9.623e-05   7.13e-05     -1.349      0.177      -0.000    4.37e-05
합계출산율                -0.5015      0.123     -4.086      0.000      -0.742      -0.261
남녀성비                  0.0529      0.006      9.562      0.000       0.042       0.064
인구증가율                 0.0503      0.011      4.543      0.000       0.029       0.072
주민등록인구             3.284e-05   4.09e-06      8.034      0.000    2.48e-05    4.09e-05
가구수                -9.34e-06   3.48e-06     -2.686      0.007   -1.62e-05   -2.52e-06
일반공공행정예산비중           -0.0052      0.007     -0.772      0.440      -0.019       0.008
하수도보급률                0.0056      0.002      2.537      0.011       0.001       0.010
상수도보급률               -0.0128      0.003     -4.474      0.000      -0.018      -0.007
==============================================================================
Omnibus:                       34.036   Durbin-Watson:                   1.  자기상관을 측정 값이 2에 가까울수록 자기상관이 없음을 의미
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               21.556
Skew:                           0.140   Prob(JB):                     2.09e-05
Kurtosis:                       2.506   Cond. No.  다중공선성         3.72e+09 
==============================================================================
'''
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_train = parameter_integrate[:]
def feature_engineering_XbyVIF(X_train):
    vif = pd.DataFrame()
    vif['VIF_Factor'] = [variance_inflation_factor(X_train.values, i)
                         for i in range(X_train.shape[1])]
    vif['Feature'] = X_train.columns
    return vif
vif = feature_engineering_XbyVIF(X_train)
print(vif)   
'''
 VIF_Factor            Feature
0     95.186268     교원_1인당_학생수_유치원
1    521.218827    교원_1인당_학생수_초등학교
2    270.375424     교원_1인당_학생수_중학교
3    176.494232    교원_1인당_학생수_고등학교
4    100.489546   유치원_학급당 학생 수 (명)
5    664.625147  초등학교_학급당 학생 수 (명)
6    386.866686   중학교_학급당 학생 수 (명)
7    281.130457  고등학교_학급당 학생 수 (명)
8     22.096939      학교교과 교습학원 (개)
9     23.485774      평생직업 교육학원 (개)
10     2.581102      사설학원당 학생수 (명)
11    33.688831             유치원생 수
12    51.885965             초등학생 수
13     6.243579               종합병원
14    11.693642                 병원
15    25.374645                 의원
16    51.154111            치과병(의)원
17     2.429785               한방병원
18    52.298010                한의원
19     6.639114  인구 천명당 의료기관병상수(개)
20    15.833686           총병상수 (개)
21    28.307544             고위험음주율
22   115.148701                비만율
23   874.596279    EQ.5D(건강상태 표준화)
24    54.859069         주관적건강수준인지율
25  3276.359825       건강보험 적용인구 현황
26    59.081145             운전행태영역
27    18.795760             교통안전영역
28    80.049329             보행행태영역
29    16.814881        1인당 자동차등록대수
30     3.662277             도시지역면적
31   202.880941               주택 수
32    47.237937               출생아수
33    38.297247              합계출산율
34   595.593724               남녀성비
35     1.513586              인구증가율
36  3195.760674             주민등록인구
37   357.916407                가구수
38     5.860225         일반공공행정예산비중
39    70.495282             하수도보급률
40   134.624732             상수도보급률'''


#%% 
# Multiple regression analysis results
# As a result of checking the coef (regression coefficient), if the regression coefficient is positive, the dependent variable, the local extinction risk grade, is lowered (grade increase 1->4), that is,
# It is judged to be a factor that prevents the risk of fat loss.
# P>|t| : If the P-value is 0.05 or less, it is judged that the independent variable has a significant effect on the dependent variable.


# Filter variables with a positive coef value and a p value of 0.05 or less.

'''
                                      Variable  Coefficient       P-value
교원_1인당_학생수_중학교      교원_1인당_학생수_중학교     0.153609  1.991120e-05
교원_1인당_학생수_고등학교    교원_1인당_학생수_고등학교     0.115309  9.986617e-05
유치원_학급당 학생 수 (명)  유치원_학급당 학생 수 (명)     0.042300  2.807628e-03
평생직업 교육학원 (개)        평생직업 교육학원 (개)     0.003539  4.819803e-04
한방병원                          한방병원     0.011943  5.042306e-04
총병상수 (개)                  총병상수 (개)     0.000055  9.866238e-03
주관적건강수준인지율              주관적건강수준인지율     0.016248  5.578230e-06
남녀성비                          남녀성비     0.052923  4.315324e-21
인구증가율                        인구증가율     0.050320  5.976054e-06
주민등록인구                      주민등록인구     0.000033  1.845772e-15
하수도보급률                      하수도보급률     0.005586  1.129089e-02'''
#%%

best_model = sm.OLS(parameter_integrate_p['소멸위험등급'], parameter_integrate[['교원_1인당_학생수_중학교', '교원_1인당_학생수_고등학교','유치원_학급당 학생 수 (명)',
                                                    '평생직업 교육학원 (개)','한방병원', '총병상수 (개)', '주관적건강수준인지율', '남녀성비', '인구증가율', '주민등록인구', '하수도보급률']])
best_results = best_model.fit()
print(best_results.summary())  

'''
 OLS Regression Results 회귀분석 결과                           
=======================================================================================
Dep. Variable:                 소멸위험등급   R-squared (uncentered):             0.857
Model:                            OLS   Adj. R-squared (uncentered):              0.856
Method:                 Least Squares   F-statistic:                              866.9
Date:                Wed, 31 Jul 2024   Prob (F-statistic):                        0.00
Time:                        10:27:17   Log-Likelihood:                         -2330.9
No. Observations:                1603   AIC:                                      4684.
parameter_integrate Residuals:                    1592   BIC:                                      4743.
parameter_integrate Model:                          11                                                  
Covariance Type:            nonrobust                                                  
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
교원_1인당_학생수_중학교       0.0030      0.015      0.204      0.838      -0.026       0.032
교원_1인당_학생수_고등학교      0.0452      0.015      3.048      0.002       0.016       0.074
유치원_학급당 학생 수 (명)     0.0629      0.010      6.131      0.000       0.043       0.083
평생직업 교육학원 (개)       -0.0013      0.000     -4.504      0.000      -0.002      -0.001
한방병원                 0.0092      0.003      3.411      0.001       0.004       0.014
총병상수 (개)         -3.671e-05   1.48e-05     -2.488      0.013   -6.57e-05   -7.77e-06
주관적건강수준인지율           0.0134      0.003      4.068      0.000       0.007       0.020
남녀성비                 0.0033      0.002      1.675      0.094      -0.001       0.007
인구증가율                0.0109      0.011      1.004      0.316      -0.010       0.032
주민등록인구            -5.07e-08   2.17e-07     -0.233      0.816   -4.77e-07    3.75e-07
하수도보급률               0.0029      0.002      1.392      0.164      -0.001       0.007
==============================================================================
Omnibus:                      552.264   Durbin-Watson:                   1.375
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               78.622
Skew:                           0.046   Prob(JB):                     8.46e-18
Kurtosis:                       1.919   Cond. No.                     2.15e+05
=============================================================================='''

from statsmodels.stats.outliers_influence import variance_inflation_factor
X_train = parameter_integrate[['교원_1인당_학생수_중학교', '교원_1인당_학생수_고등학교','유치원_학급당 학생 수 (명)',
                                                    '평생직업 교육학원 (개)','한방병원', '총병상수 (개)', '주관적건강수준인지율', '남녀성비', '인구증가율', '주민등록인구', '하수도보급률']]
def feature_engineering_XbyVIF(X_train):
    vif = pd.DataFrame()
    vif['VIF_Factor'] = [variance_inflation_factor(X_train.values, i)
                         for i in range(X_train.shape[1])]
    vif['Feature'] = X_train.columns
    return vif
vif = feature_engineering_XbyVIF(X_train)
print(vif)   

'''
 VIF_Factor           Feature
0    34.713279    교원_1인당_학생수_중학교
1    34.296144   교원_1인당_학생수_고등학교
2    41.001107  유치원_학급당 학생 수 (명)
3     1.544942     평생직업 교육학원 (개)
4     1.153632              한방병원
5     5.815077          총병상수 (개)
6    36.363790        주관적건강수준인지율
7    57.878162              남녀성비
8     1.125742             인구증가율
9     6.978825            주민등록인구
10   49.491858            하수도보급률'''




#%%
# Autocorrelation:
# The residuals should not be autocorrelated with time order or other independent variables. You can evaluate autocorrelation using the Durbin-Watson statistic.
#from statsmodels.stats.stattools import durbin_watson
# dw = durbin_watson(parameter_integrate['잔차'])
print(f'Durbin-Watson 통계량: {dw}')



#%%
'''
# Through residual analysis, you can build a better prediction model by assessing the suitability of the model and finding ways to improve it.
# Residuals are defined as the difference between the model's predicted values ​​and the actual observed values.
# It is common to evaluate agreement with the theoretical distribution with a selected sample rather than the population.
# Font settings
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Korean font settings
font_path = "c:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# residual data
residuals = best_results.resid

# Q-Q (Quantile-Quantile Plot)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
sm.qqplot(residuals, line='s', ax=ax)


plt.xlim(-3, 3)
plt.ylim(-3, 3)

plt.title('실제값과 오차비교(Q-Q Plot)')
plt.xlabel('이론적 분위수')  # x-axis labels
plt.ylabel('실제 데이터 분위수')  # y-axis labels

# Save plot
plt.savefig('C:/Users/Shin/Documents/Final_Project/Photo/qqplot결과.png')

# Show on plot screen
plt.show()
'''

#%%
# sapiro black
# Shapiro test before applying log transformation to independent variables
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

residuals = best_results.resid

# Shapiro-Wilk test
shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic}")
print(f"p-value: {shapiro_test.pvalue}")

# Histogram and normal distribution curve of residuals
plt.hist(residuals, bins=30, density=True, alpha=0.6, color='g')

mu, std = np.mean(residuals), np.std(residuals)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.parameter_integrate(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

plt.title('예측값과 실제값 차이 히스토그램(로그변환이전)')
plt.xlabel('잔차')
plt.ylabel('밀도')
plt.savefig('C:/Users/Shin/Documents/Final_Project/Photo/잔차결과(로그변환이전).png')
plt.show()

# Normality Test
#Shapiro-Wilk test statistic: 0.9684673915431092
# If the number of data is very small, the normality test is easily passed, so you should not blindly trust the results.
# The closer the test statistic is to 1, the closer the residual distribution is to a normal distribution.
#p-value: 3.105358961597644e-18
# The P-value is much smaller than 0.05, meaning that the residuals do not follow a normal distribution.
# If there are many variables, overfitting or multicollinearity may occur.
'''
# Variables that affect fat loss
Shapiro-Wilk test statistic: 0.9684673915431092
p-value: 3.105358961597644e-18
# All variables
Shapiro-Wilk test statistic: 0.9919487552757812
p-value: 1.0836112418254231e-07'''
#%%

# Shapiro-Wilk test result output
print(f"Shapiro-Wilk test statistic: {shapiro_test.statistic}")
print(f"p-value: {shapiro_test.pvalue}")



#%%
# Kolmogorov-Smirnov test before log transformation
from scipy import stats
import numpy as np

# residual data
residuals = best_results.resid

# Kolmogorov-Smirnov test
mu, sigma = np.mean(residuals), np.std(residuals)
statistic, p_value = stats.kstest(residuals, 'norm', args=(mu, sigma))

print(f'K-S Statistic: {statistic}')
print(f'P-value: {p_value}')
'''
K-S Statistic: 0.0660961497316579
P-value: 1.5632972948507028e-06'''


#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro

# Data preparation (using example dataframe parameter_integrate)
# parameter_integrate = pd.read_csv('your_data.csv')

# Apply log transformation to independent variables
parameter_integrate['log_교원_1인당_학생수_중학교'] = np.log(parameter_integrate['교원_1인당_학생수_중학교'] + 1)  # +1 prevents 0 values
parameter_integrate['log_교원_1인당_학생수_고등학교'] = np.log(parameter_integrate['교원_1인당_학생수_고등학교'] + 1)
parameter_integrate['log_유치원_학급당 학생 수 (명)'] = np.log(parameter_integrate['유치원_학급당 학생 수 (명)'] + 1)
parameter_integrate['log_평생직업 교육학원 (개)'] = np.log(parameter_integrate['평생직업 교육학원 (개)'] + 1)
parameter_integrate['log_한방병원'] = np.log(parameter_integrate['한방병원'] + 1)
parameter_integrate['log_총병상수 (개)'] = np.log(parameter_integrate['총병상수 (개)'] + 1)
parameter_integrate['log_남녀성비'] = np.log(parameter_integrate['남녀성비'] + 1)
parameter_integrate['log_ 주관적건강수준인지율'] = np.log(parameter_integrate['주관적건강수준인지율'] + 1)
parameter_integrate['log_ 인구증가율'] = np.log(parameter_integrate['인구증가율'] + 1)
parameter_integrate['log_주민등록인구'] = np.log(parameter_integrate['주민등록인구'] + 1)
parameter_integrate['log_하수도보급률'] = np.log(parameter_integrate['하수도보급률'] + 1)

parameter_integrate = parameter_integrate.fillna(0)
# Prepare independent variables and constant terms
X = parameter_integrate[['log_교원_1인당_학생수_중학교', 'log_교원_1인당_학생수_고등학교', 
       'log_유치원_학급당 학생 수 (명)', 'log_평생직업 교육학원 (개)', 
       'log_한방병원', 'log_총병상수 (개)','남녀성비', '주관적건강수준인지율', '인구증가율', '주민등록인구', '하수도보급률']]
X = sm.add_constant(X)  # Add constant term

# dependent variable
y = parameter_integrate_p['소멸위험등급']

# OLS model fit
model = sm.OLS(y, X)
log_results = model.fit()

# Result output
print(log_results.summary())
'''
OLS Regression Results                            
==============================================================================
Dep. Variable:                 소멸위험등급   R-squared:                 0.154
Model:                            OLS   Adj. R-squared:                  0.149
Method:                 Least Squares   F-statistic:                     26.40
Date:                Wed, 31 Jul 2024   Prob (F-statistic):           5.92e-51
Time:                        14:43:09   Log-Likelihood:                -2319.4
No. Observations:                1603   AIC:                             4663.
Df Residuals:                    1591   BIC:                             4727.
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
const                   -5.5211      0.673     -8.198      0.000      -6.842      -4.200
log_교원_1인당_학생수_중학교       0.0437      0.140      0.312      0.755      -0.231       0.318
log_교원_1인당_학생수_고등학교      0.5140      0.139      3.707      0.000       0.242       0.786
log_유치원_학급당 학생 수 (명)     1.0455      0.152      6.893      0.000       0.748       1.343
log_평생직업 교육학원 (개)       -0.1173      0.020     -5.958      0.000      -0.156      -0.079
log_한방병원                 0.0773      0.031      2.506      0.012       0.017       0.138
log_총병상수 (개)             0.0103      0.018      0.579      0.562      -0.025       0.045
남녀성비                     0.0256      0.005      5.069      0.000       0.016       0.035
주관적건강수준인지율               0.0190      0.004      5.420      0.000       0.012       0.026
인구증가율                    0.0056      0.011      0.518      0.605      -0.016       0.027
주민등록인구               -3.858e-07   1.65e-07     -2.340      0.019   -7.09e-07   -6.24e-08
하수도보급률                   0.0075      0.002      3.318      0.001       0.003       0.012
==============================================================================
Omnibus:                      342.350   Durbin-Watson:                   1.406
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               66.258
Skew:                          -0.024   Prob(JB):                     4.10e-15
Kurtosis:                       2.005   Cond. No.                     8.28e+06
==============================================================================
'''
#%%
# Residual analysis
residuals = log_results.resid
# Shapiro-Wilk test
shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic}")
print(f"p-value: {shapiro_test.pvalue}")

# Histogram and normal distribution curve of residuals
plt.hist(residuals, bins=30, density=True, alpha=0.6, color='g')

# Drawing a normal distribution curve
mu, std = np.mean(residuals), np.std(residuals)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

plt.title('예측값과 실제값 차이 히스토그램 (로그 변환 이후)')
plt.xlabel('잔차')
plt.ylabel('밀도')

# save graph
plt.savefig('C:/Users/Shin/Documents/Final_Project/Photo/잔차결과(로그변환이후).png')

# Displayed on graph screen
plt.show()

#%%
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# residual data
residuals = best_results.resid

# Shapiro-Wilk test
shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic}")
print(f"p-value: {shapiro_test.pvalue}")

# Histogram and normal distribution curve of residuals
plt.hist(residuals, bins=30, density=True, alpha=0.6, color='g')

# Drawing a normal distribution curve
mu, std = np.mean(residuals), np.std(residuals)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

plt.title('예측값과 실제값 차이 히스토그램 (로그 변환 이전)')
plt.xlabel('잔차')
plt.ylabel('밀도')

# save graph
plt.savefig('C:/Users/Shin/Documents/Final_Project/Photo/잔차결과(로그변환이전).png')

# Displayed on graph screen
plt.show()

#%%
# Kolmogorov-Smirnov test after log transformation
from scipy import stats
import numpy as np

# residual data
residuals = log_results.resid

# Kolmogorov-Smirnov test
mu, sigma = np.mean(residuals), np.std(residuals)
statistic, p_value = stats.kstest(residuals, 'norm', args=(mu, sigma))

print(f'K-S Statistic: {statistic}')
print(f'P-value: {p_value}')
#K-S Statistic: 0.05975288134346379
#P-value: 2.037984247307559e-05
#%%
# conclusion :
# Considering that the K-S statistic decreased and the p-value increased after log transformation, it can be said that log transformation contributed to some extent to improving the normality of the residuals.
# However, in both cases the p-value is much lower than 0.05, allowing us to conclude that the residuals are still not normally distributed.
# This suggests that additional transformations or other methods of modeling may be needed.
#%%

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Load data (using example data)
# parameter_integrate_p: Dependent variable data frame (e.g. including extinction risk level)
# parameter_integrate: independent variable data frame

# Setting dependent variables
y = parameter_integrate_p['소멸위험등급']

# Independent variable data frame (bias added)
X = parameter_integrate.values

# 1. Standard Scaler
scaler_standard = StandardScaler()
X_standard_scaled = scaler_standard.fit_transform(X)
X_standard_scaled = sm.add_constant(X_standard_scaled)  # Add bias

model_standard = sm.OLS(y, X_standard_scaled)
results_standard = model_standard.fit()
print("Standard Scaler:")
print(results_standard.summary())

# 2. MinMax Scaler
scaler_minmax = MinMaxScaler()
X_minmax_scaled = scaler_minmax.fit_transform(X)
X_minmax_scaled = sm.add_constant(X_minmax_scaled)  # Add bias

model_minmax = sm.OLS(y, X_minmax_scaled)
results_minmax = model_minmax.fit()
print("\nMinMax Scaler:")
print(results_minmax.summary())

# 3. Robust Scaler
scaler_robust = RobustScaler()
X_robust_scaled = scaler_robust.fit_transform(X)
X_robust_scaled = sm.add_constant(X_robust_scaled)  # Add bias

model_robust = sm.OLS(y, X_robust_scaled)
results_robust = model_robust.fit()
print("\nRobust Scaler:")
print(results_robust.summary())
'''
Standard Scaler:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 소멸위험등급   R-squared:                 0.346
Model:                            OLS   Adj. R-squared:                  0.329
Method:                 Least Squares   F-statistic:                     20.18
Date:                Thu, 01 Aug 2024   Prob (F-statistic):          4.34e-115
Time:                        15:21:16   Log-Likelihood:                -2112.9
No. Observations:                1603   AIC:                             4310.
Df Residuals:                    1561   BIC:                             4536.
Df Model:                          41                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.4997      0.023    109.242      0.000       2.455       2.545
x1             0.0814      0.056      1.456      0.146      -0.028       0.191
x2             0.2017      0.157      1.285      0.199      -0.106       0.510
x3             0.4873      0.117      4.166      0.000       0.258       0.717
x4             0.2949      0.077      3.845      0.000       0.144       0.445
x5             0.1529      0.057      2.671      0.008       0.041       0.265
x6            -0.4466      0.155     -2.876      0.004      -0.751      -0.142
x7            -0.3884      0.100     -3.874      0.000      -0.585      -0.192
x8            -0.0993      0.076     -1.313      0.189      -0.248       0.049
x9            -0.4006      0.088     -4.573      0.000      -0.572      -0.229
x10            0.2636      0.092      2.873      0.004       0.084       0.444
x11           -0.0073      0.032     -0.225      0.822      -0.071       0.056
x12            0.0237      0.101      0.234      0.815      -0.175       0.222
x13            0.0603      0.121      0.500      0.617      -0.176       0.297
x14           -0.1394      0.042     -3.309      0.001      -0.222      -0.057
x15           -0.3058      0.058     -5.267      0.000      -0.420      -0.192
x16           -0.0342      0.089     -0.386      0.699      -0.208       0.139
x17            0.1089      0.123      0.887      0.375      -0.132       0.350
x18            0.1160      0.034      3.395      0.001       0.049       0.183
x19            0.2305      0.124      1.859      0.063      -0.013       0.474
x20           -0.0402      0.032     -1.241      0.215      -0.104       0.023
x21            0.1901      0.067      2.854      0.004       0.059       0.321
x22            0.0375      0.029      1.309      0.191      -0.019       0.094
x23            0.0062      0.033      0.188      0.851      -0.059       0.071
x24           -0.0190      0.026     -0.743      0.457      -0.069       0.031
x25            0.1202      0.027      4.400      0.000       0.067       0.174
x26           -6.0034      0.871     -6.893      0.000      -7.712      -4.295
x27            0.0481      0.045      1.075      0.283      -0.040       0.136
x28           -0.0182      0.036     -0.502      0.616      -0.089       0.053
x29           -0.0398      0.042     -0.943      0.346      -0.123       0.043
x30           -0.1148      0.028     -4.102      0.000      -0.170      -0.060
x31            0.0919      0.038      2.430      0.015       0.018       0.166
x32           -0.1516      0.034     -4.458      0.000      -0.218      -0.085
x33           -0.0204      0.024     -0.856      0.392      -0.067       0.026
x34           -0.0543      0.034     -1.605      0.109      -0.121       0.012
x35           -1.2348      0.216     -5.723      0.000      -1.658      -0.812
x36           -0.0983      0.115     -0.857      0.392      -0.323       0.127
x37           -0.1445      0.037     -3.940      0.000      -0.216      -0.073
x38            0.2804      0.031      9.073      0.000       0.220       0.341
x39            0.1330      0.028      4.754      0.000       0.078       0.188
x40            7.1802      0.903      7.950      0.000       5.409       8.952
x41           -0.0203      0.025     -0.809      0.419      -0.070       0.029
==============================================================================
Omnibus:                       41.333   Durbin-Watson:                   1.700
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               23.784
Skew:                           0.127   Prob(JB):                     6.84e-06
Kurtosis:                       2.460   Cond. No.                         212.
==============================================================================
'''
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
'''
MinMax Scaler:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 소멸위험등급   R-squared:                 0.346
Model:                            OLS   Adj. R-squared:                  0.329
Method:                 Least Squares   F-statistic:                     20.18
Date:                Thu, 01 Aug 2024   Prob (F-statistic):          4.34e-115
Time:                        15:21:16   Log-Likelihood:                -2112.9
No. Observations:                1603   AIC:                             4310.
Df Residuals:                    1561   BIC:                             4536.
Df Model:                          41                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.8132      0.376      4.821      0.000       1.075       2.551
x1             0.4250      0.292      1.456      0.146      -0.148       0.998
x2             0.8176      0.637      1.285      0.199      -0.431       2.066
x3             2.1017      0.505      4.166      0.000       1.112       3.091
x4             1.8255      0.475      3.845      0.000       0.894       2.757
x5             0.7529      0.282      2.671      0.008       0.200       1.306
x6            -1.7703      0.616     -2.876      0.004      -2.978      -0.563
x7            -2.2006      0.568     -3.874      0.000      -3.315      -1.086
x8            -0.6755      0.514     -1.313      0.189      -1.685       0.334
x9            -1.6695      0.365     -4.573      0.000      -2.386      -0.953
x10            1.4313      0.498      2.873      0.004       0.454       2.409
x11           -0.1131      0.502     -0.225      0.822      -1.098       0.872
x12            0.1486      0.635      0.234      0.815      -1.097       1.394
x13            0.3284      0.657      0.500      0.617      -0.961       1.618
x14           -0.8941      0.270     -3.309      0.001      -1.424      -0.364
x15           -1.8553      0.352     -5.267      0.000      -2.546      -1.164
x16           -0.3520      0.911     -0.386      0.699      -2.139       1.435
x17            0.7317      0.825      0.887      0.375      -0.886       2.349
x18            1.4812      0.436      3.395      0.001       0.625       2.337
x19            1.2976      0.698      1.859      0.063      -0.071       2.667
x20           -0.2901      0.234     -1.241      0.215      -0.749       0.168
x21            1.2652      0.443      2.854      0.004       0.396       2.135
x22            0.2594      0.198      1.309      0.191      -0.129       0.648
x23            0.0424      0.225      0.188      0.851      -0.399       0.484
x24           -0.1160      0.156     -0.743      0.457      -0.422       0.190
x25            0.7956      0.181      4.400      0.000       0.441       1.150
x26          -32.8079      4.760     -6.893      0.000     -42.144     -23.472
x27            0.3188      0.297      1.075      0.283      -0.263       0.900
x28           -0.0801      0.160     -0.502      0.616      -0.393       0.233
x29           -0.2891      0.307     -0.943      0.346      -0.891       0.313
x30           -1.9316      0.471     -4.102      0.000      -2.855      -1.008
x31            0.5299      0.218      2.430      0.015       0.102       0.958
x32           -0.9571      0.215     -4.458      0.000      -1.378      -0.536
x33           -0.2443      0.285     -0.856      0.392      -0.804       0.316
x34           -0.3489      0.217     -1.605      0.109      -0.775       0.077
x35           -6.8086      1.190     -5.723      0.000      -9.142      -4.475
x36           -0.7233      0.844     -0.857      0.392      -2.379       0.933
x37           -1.0437      0.265     -3.940      0.000      -1.563      -0.524
x38            2.2144      0.244      9.073      0.000       1.736       2.693
x39            2.3583      0.496      4.754      0.000       1.385       3.331
x40           38.8799      4.890      7.950      0.000      29.287      48.472
x41           -0.2191      0.271     -0.809      0.419      -0.750       0.312
==============================================================================
Omnibus:                       41.333   Durbin-Watson:                   1.700
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               23.784
Skew:                           0.127   Prob(JB):                     6.84e-06
Kurtosis:                       2.460   Cond. No.                         805.
==============================================================================
'''
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
'''
Robust Scaler:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 소멸위험등급   R-squared:                 0.346
Model:                            OLS   Adj. R-squared:                  0.329
Method:                 Least Squares   F-statistic:                     20.18
Date:                Thu, 01 Aug 2024   Prob (F-statistic):          4.34e-115
Time:                        15:21:16   Log-Likelihood:                -2112.9
No. Observations:                1603   AIC:                             4310.
Df Residuals:                    1561   BIC:                             4536.
Df Model:                          41                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.5092      0.052     47.943      0.000       2.407       2.612
x1             0.0911      0.063      1.456      0.146      -0.032       0.214
x2             0.3816      0.297      1.285      0.199      -0.201       0.964
x3             0.7506      0.180      4.166      0.000       0.397       1.104
x4             0.4564      0.119      3.845      0.000       0.224       0.689
x5             0.2259      0.085      2.671      0.008       0.060       0.392
x6            -0.7081      0.246     -2.876      0.004      -1.191      -0.225
x7            -0.5312      0.137     -3.874      0.000      -0.800      -0.262
x8            -0.1330      0.101     -1.313      0.189      -0.332       0.066
x9            -0.2459      0.054     -4.573      0.000      -0.351      -0.140
x10            0.1493      0.052      2.873      0.004       0.047       0.251
x11           -0.0013      0.006     -0.225      0.822      -0.013       0.010
x12            0.0256      0.109      0.234      0.815      -0.189       0.240
x13            0.0675      0.135      0.500      0.617      -0.197       0.332
x14           -0.1626      0.049     -3.309      0.001      -0.259      -0.066
x15           -0.3452      0.066     -5.267      0.000      -0.474      -0.217
x16           -0.0352      0.091     -0.386      0.699      -0.214       0.143
x17            0.1297      0.146      0.887      0.375      -0.157       0.416
x18            0.0233      0.007      3.395      0.001       0.010       0.037
x19            0.2558      0.138      1.859      0.063      -0.014       0.526
x20           -0.0464      0.037     -1.241      0.215      -0.120       0.027
x21            0.2261      0.079      2.854      0.004       0.071       0.381
x22            0.0503      0.038      1.309      0.191      -0.025       0.126
x23            0.0084      0.045      0.188      0.851      -0.080       0.096
x24           -0.0256      0.034     -0.743      0.457      -0.093       0.042
x25            0.1562      0.036      4.400      0.000       0.087       0.226
x26           -7.7056      1.118     -6.893      0.000      -9.898      -5.513
x27            0.0187      0.017      1.075      0.283      -0.015       0.053
x28           -0.0273      0.054     -0.502      0.616      -0.134       0.079
x29           -0.0396      0.042     -0.943      0.346      -0.122       0.043
x30           -0.0743      0.018     -4.102      0.000      -0.110      -0.039
x31            0.1319      0.054      2.430      0.015       0.025       0.238
x32           -0.1505      0.034     -4.458      0.000      -0.217      -0.084
x33           -0.0059      0.007     -0.856      0.392      -0.020       0.008
x34           -0.0413      0.026     -1.605      0.109      -0.092       0.009
x35           -1.5634      0.273     -5.723      0.000      -2.099      -1.028
x36           -0.1170      0.137     -0.857      0.392      -0.385       0.151
x37           -0.1906      0.048     -3.940      0.000      -0.286      -0.096
x38            0.3033      0.033      9.073      0.000       0.238       0.369
x39            0.1052      0.022      4.754      0.000       0.062       0.149
x40            9.3599      1.177      7.950      0.000       7.051      11.669
x41           -0.0193      0.024     -0.809      0.419      -0.066       0.027
==============================================================================
Omnibus:                       41.333   Durbin-Watson:                   1.700
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               23.784
Skew:                           0.127   Prob(JB):                     6.84e-06
Kurtosis:                       2.460   Cond. No.                         385.
==============================================================================
'''
#%%
from matplotlib import font_manager, rc
font_path = "c:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

import matplotlib.pyplot as plt

# data
categories = ['GBM', 'LGBM']
values = [89.2, 91.3]

# Create a bar chart
plt.bar(categories, values)

# Set graph title and axis labels
plt.title('분류모델 성능비교')
plt.xlabel('모델종류')
plt.ylabel('분류정확도(%)')
plt.ylim(85, 93)
plt.savefig('C:/Users/Shin/Documents/Final_Project/Mysql/data/성능비교.png')
# graph display
plt.show()





























