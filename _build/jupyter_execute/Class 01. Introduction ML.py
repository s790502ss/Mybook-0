#!/usr/bin/env python
# coding: utf-8

# # Class 01. Introduction ML 簡介

# In[1]:


import warnings # 忽略警告訊息 
warnings.filterwarnings("ignore")


# # 1. 世界人口

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Datasets
year=[1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965,
      1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981,
      1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997,
      1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
      2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029,
      2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045,
      2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061,
      2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077,
      2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093,
      2094, 2095, 2096, 2097, 2098, 2099, 2100]
pop=[2.53, 2.57, 2.62, 2.67, 2.71, 2.76, 2.81, 2.86, 2.92, 2.97, 3.03, 3.08, 3.14, 3.2, 3.26, 3.33, 3.4,
     3.47, 3.54, 3.62, 3.69, 3.77, 3.84, 3.92, 4.0, 4.07, 4.15, 4.22, 4.3, 4.37, 4.45, 4.53, 4.61, 4.69,
     4.78, 4.86, 4.95, 5.05, 5.14, 5.23, 5.32, 5.41, 5.49, 5.58, 5.66, 5.74, 5.82, 5.9, 5.98, 6.05, 6.13,
     6.2, 6.28, 6.36, 6.44, 6.51, 6.59, 6.67, 6.75, 6.83, 6.92, 7.0, 7.08, 7.16, 7.24, 7.32, 7.4, 7.48,
     7.56, 7.64, 7.72, 7.79, 7.87, 7.94, 8.01, 8.08, 8.15, 8.22, 8.29, 8.36, 8.42, 8.49, 8.56, 8.62, 8.68,
     8.74, 8.8, 8.86, 8.92, 8.98, 9.04, 9.09, 9.15, 9.2, 9.26, 9.31, 9.36, 9.41, 9.46, 9.5, 9.55, 9.6, 9.64,
     9.68, 9.73, 9.77, 9.81, 9.85, 9.88, 9.92, 9.96, 9.99, 10.03, 10.06, 10.09, 10.13, 10.16, 10.19, 10.22,
     10.25, 10.28, 10.31, 10.33, 10.36, 10.38, 10.41, 10.43, 10.46, 10.48, 10.5, 10.52, 10.55, 10.57, 10.59,
     10.61, 10.63, 10.65, 10.66, 10.68, 10.7, 10.72, 10.73, 10.75, 10.77, 10.78, 10.79, 10.81, 10.82, 10.83,
     10.84, 10.85]
df = pd.DataFrame({'year' : year, 'pop' : pop})

x=year
y=pop

# 2. 互動查詢
# 迴歸次方，越高則模型會越擬合
in_squa = 30 # int(input('Please input regress square(1~50):')) *上傳 notebook 不能有 input 
in_year = 2100 # int(input('Please input year(1950~2100) to calculation:')) *上傳 notebook 不能有 input 

# 線性迴歸係數解
fit1 = np.polyfit(x, y, in_squa)

if (2100>=in_year>= 1950) & (50>=in_squa>=1):
    print('The actual pop is:', y[in_year-1950])
    
    # np.poly1d(): 把線性迴歸係數解代入 in_year
    print(f'Predict pop is: {(np.poly1d(fit1)(in_year)):.2f}')
    y1 = fit1[0]*np.array(x) + fit1[1]
    
    # MSE
    print('MSE is:', f'{((y - y1)**2).mean():.2f}')
elif 50<in_squa or in_squa<1:
    print('Wrong square!')
else:
    print('Wrong year!')


# In[3]:


# 3. 觀察三種迴歸次方作圖
def ppf(x, y, order):
    # polyfit: 線性迴歸係數解
    fit = np.polyfit(x, y, order)
    
    # poly1d: 將 polyfit 迴歸解代入
    p = np.poly1d(fit)
    t = np.linspace(1950, 2100, 2000)
    plt.plot(x, y, 'ro', t, p(t), 'b--')

# 作圖
plt.figure(figsize=(18, 4))
titles = ['fitting with 1', 'fitting with 3', 'fitting with 50']

for i, o in enumerate([1, 3, 50]):
    plt.subplot(1, 3, i+1)
    
    # 呼叫函式
    ppf(year, pop, o)
    plt.title(titles[i], fontsize=20)
plt.show()


# # 2. 鳶尾花

# In[4]:


import pandas as pd
import numpy as np
from sklearn import datasets     # 引用 Scikit-Learn 中的 套件 datasets

ds = datasets.load_iris()        # dataset: 引用 datasets 中的函數 load_iris
print(ds.DESCR)                  # DESCR: description，描述載入內容

# 建立 data 關係 (ds.data: 值, ds.feature_names: 項目名)
X =pd.DataFrame(ds.data, columns=ds.feature_names)
y = ds.target

# print(X.head(10))
# print('------------------------------------------')
# print(y)
# print('------------------------------------------')
# print(ds.target_names)                                # 列出 y 名稱
# name = ['setosa' 'versicolor' 'virginica']
# print('------------------------------------------')
# print(X.info())
# print(X.describe())                                   # 含 count, mean, std, min, 25%, 50%, 75%, max
# print('------------------------------------------')

# # 2. Data clean (missing value check)
# print(X.isna().sum())
# print('------------------------------------------')

# 3. Feature Engineering
# 如：黃豆長/寬/高 → 黃豆體積
# 此範例沒有，故不需要整理 data。

# 4. Split
from sklearn.model_selection import train_test_split    

# test_size=0.2 : 測試用資料為 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train.shape, y_train.shape)
# 得到 (120, 4) (120,)

# 5. Define and train the KNN model
from sklearn.neighbors import KNeighborsClassifier

# n_neighbors=: 超參數 (hyperparameter)
clf = KNeighborsClassifier(n_neighbors = 3)
# 適配(訓練)，迴歸/分類/降維...皆用 fit(x_train, y_train)
clf.fit(X_train, y_train)

# algorithm.score: 使用 test 資料，並根據結果評分
print(f'score={clf.score(X_test, y_test)}')

# 驗證答案
print(' '.join(y_test.astype(str)))
print(' '.join(clf.predict(X_test).astype(str)))

# 查看預測的機率
print(clf.predict_proba(X_test))

# print(clf.predict([[5.1,3.5,1.4,0.2], [3.1,2.5,1.4,0.2]]))
# print(clf.predict_proba([[5.1,3.5,1.4,0.2], [3.1,2.5,1.4,0.2]]))


# # 3. 乳癌 KNN

# In[5]:


import pandas as pd
import numpy as np
from sklearn import datasets     # 引用 Scikit-Learn 中的 套件 datasets
# 1. Dataset
ds = datasets.load_breast_cancer()
# print(ds.DESCR) # 描述載入內容

# pd.set_option('display.max_columns', None) # 將 max_columns 全秀出來，不會產生...，同理 max_rows。
X =pd.DataFrame(ds.data, columns=ds.feature_names)
y = ds.target

# 2. Data clean (missing value check)
print(X.isna().sum())


# In[6]:


# 3. Feature Engineering

# 4. Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train.shape, y_train.shape
# 得到 (455, 30) (455,)


# In[7]:


# 5. Define and train the KNN model
from sklearn.neighbors import KNeighborsClassifier

# n_neighbors=: 超參數 (hyperparameter)
clf = KNeighborsClassifier(n_neighbors = 3)

# 適配(訓練)，迴歸/分類/降維...皆用 fit(x_train, y_train)
clf.fit(X_train, y_train)

# algorithm.score: 使用 test 資料，並根據結果評分
clf.score(X_test, y_test)


# In[8]:


# 驗證答案
print(' '.join(y_test.astype(str)))
print(' '.join(clf.predict(X_test).astype(str)))

# 查看預測的機率
print(clf.predict_proba(X_test))

# # print(clf.predict([[5.1,3.5,1.4,0.2], [3.1,2.5,1.4,0.2]]))
# # print(clf.predict_proba([[5.1,3.5,1.4,0.2], [3.1,2.5,1.4,0.2]]))


# # 4. load_boston()

# In[9]:


import pandas as pd
import numpy as np
from sklearn import datasets     # 引用 Scikit-Learn 中的 套件 datasets

ds = datasets.load_boston()
print(ds.DESCR)

# 1. Dataset
# pd.set_option('display.max_columns', None)        # 將 max_columns 全秀出來，不會產生...，同理 max_rows。
X =pd.DataFrame(ds.data, columns=ds.feature_names)
y = ds.target


# In[10]:


X.head(10)


# In[11]:


X.info()


# In[12]:


# 含 count, mean, std, min, 25%, 50%, 75%, max
X.describe()


# In[13]:


# 2. Data clean (missing value check)
X.isna().sum()


# In[14]:


# 3. Feature Engineering

# 4. Split
from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train.shape, y_train.shape)


# In[15]:


# 5. Define and train the LinearRegression model
from sklearn.linear_model import LinearRegression
clf = LinearRegression()

# 適配(訓練)，迴歸/分類/降維...皆用 fit(x_train, y_train)
clf.fit(X_train, y_train)

# algorithm.score: 使用 test 資料，並根據結果評分
clf.score(X_test, y_test)

