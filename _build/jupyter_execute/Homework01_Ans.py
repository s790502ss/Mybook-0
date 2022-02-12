#!/usr/bin/env python
# coding: utf-8

# # Homework01_Ans

# In[1]:


import warnings # 忽略警告訊息 
warnings.filterwarnings("ignore")


# # 1. load_wine()

# In[2]:


import pandas as pd
import numpy as np
from sklearn import datasets     # 引用 Scikit-Learn 中的 套件 datasets

ds = datasets.load_wine()
print(ds.DESCR)                  # DESCR: description，描述載入內容

# 1. Dataset
X =pd.DataFrame(ds.data, columns=ds.feature_names)
# print(X.head())
y = ds.target
# print(y)

# # 2. Data clean
# print(X.isna().sum())

# # 3. Date Feturing
# # None

# 4. Split
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.1)
print(X_train.shape, y_train.shape)
# 得到 (160, 13) (160,)
# print('------------------------------------------')

# 5-1. Define and train the KNN model
from sklearn.neighbors import KNeighborsClassifier as KNN
clf = KNN(n_neighbors=3)

# 訓練
clf.fit(X_train, y_train)

# 打分數
print(clf.score(X_test, y_test))
# score = 0.72

# 驗證答案
print(list(y_test))
print(list(clf.predict(X_test)))
# [0, 0, 1, 1, 1, 0, 2, 0, 2, 1, 0, 0, 0, 1, 0, 2, 2, 1]
# [0, 0, 1, 1, 1, 0, 2, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 2]
# 錯 5 個

# 5-2. Define and train the LogisticRegression model
from sklearn.linear_model import LogisticRegression as lr
clf2 = lr(solver='liblinear')
# 訓練
clf2.fit(X_train, y_train)

# 打分數
print(clf2.score(X_test, y_test))
# score = 1.0

# 驗證答案
print(list(y_test))
print(list(clf.predict(X_test)))
# [2, 2, 1, 2, 0, 0, 0, 0, 1, 2, 0, 2, 1, 2, 0, 2, 1, 1]
# [2, 2, 1, 2, 0, 0, 0, 0, 1, 2, 0, 2, 1, 2, 0, 2, 1, 1]
# 全對!

X_test.to_csv('wine_test.csv', index=False)

# 補充：存取 & 取用模型
import joblib as jb
# 存取
jb.dump(clf2, 'wine.joblib')

# 讀取
X = pd.read_csv('wine_test.csv')
print('存取/取用:\n', list(y_test))
print(list(clf2.predict(X)))


# # 2. load_diabetes()

# In[3]:


import pandas as pd
import numpy as np
from sklearn import datasets, neighbors     # 引用 Scikit-Learn 中的 套件 datasets

ds = datasets.load_diabetes()
print(ds.DESCR)                  # DESCR: description，描述載入內容

# 1. Dataset
X = pd.DataFrame(ds.data, columns=ds.feature_names)
y = ds.target
print(X)    # 經過標準化: (X-m)/sigma，平均為 0 / 標準差為 1
print(y)

# 2. Data clean
print(X.isna().sum())

# 3. Date Feturing
# None

# 4. Split
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
# 得到 (353, 10) (353,)

# 5. Define and train the LinearRegression model
from sklearn.linear_model import LinearRegression as lr
clf = lr()

# 訓練
clf.fit(X_train, y_train)

# 打分數
print(f'{clf.score(X_test, y_test):.2}')
# score = 0.50

from sklearn.metrics import mean_squared_error, r2_score
y_pred = clf.predict(X_test)

# Coefficients (一次項式係數)
# y = w1*x1 + w2*x2 + w3*x3 ... w10*x10 + b
print('Coefficients: ', clf.coef_)
print('Intercept: ', clf.intercept_)

# MSE (均方誤差)
# 1/n * sum(y_pred-y_test)
print(f'Mean squared error: {mean_squared_error(y_test, y_pred)}')

# Coefficient of determination (判定係數)
# 越接近 1 越好
print(f'Coefficient of determination: {r2_score(y_test, y_pred)}')

# # 驗證答案
# print(list(y_test))
# print(list(clf.predict(X_test)))

# # # 補充：存取 & 取用模型
# # import joblib as jb
# # # 存取
# # jb.dump(clf2, 'wine.joblib')

# # # 讀取
# # X = pd.read_csv('wine_test.csv')
# # print('存取/取用:\n', list(y_test))
# # print(list(clf2.predict(X)))


# # 3. Tips

# In[4]:


import pandas as pd
import numpy as np
from sklearn import datasets     # 引用 Scikit-Learn 中的 套件 datasets
pd.set_option('display.float_format', lambda x: f'{x:.3}')           # 轉換 '科學記號' to '浮點數'


df = pd.read_csv('tips.csv')
print(df)

# 1. Dataset
X = df.drop('tip', axis=1)
# print(X.head())
y = df['tip']
# print(y.head())

# 2. Data clean
# 2-1. isna
print(df.isna().sum())

# 2-2. coding coulumn item
print(X['day'].unique())

gb = df.groupby(['day'])['tip'].mean()
print(gb)
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(gb.index, gb.values)
plt.savefig('pic_18.png')
plt.show()

X['sex'].replace({'Female' : 0, 'Male' : 1}, inplace=True)
X['smoker'].replace({'Yes' : 0, 'No' : 1}, inplace=True)
X['day'].replace({'Thur' : 0, 'Fri' : 0, 'Sat' : 2, 'Sun' : 3}, inplace=True)
X['time'].replace({'Lunch' : 0, 'Dinner' : 1}, inplace=True)
# print(X)
# print(y)

# 3. Date Feturing

# 4. Split
from sklearn.model_selection import train_test_split as tts

# test_size=0.2 : 測試用資料為 20%
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2)
print(X_train.shape, y_train.shape)
# 得到 (195, 6) (195,)
print('------------------------------------------')


# 5. Define and train the LinearRegression model
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(X_train, y_train)

print(f'score = {clf.score(X_test, y_test):.2}')

# 驗證答案
print(list(y_test))
b = [float(f'{i:.2}') for i in clf.predict(X_test)]
print(b)


# In[ ]:




