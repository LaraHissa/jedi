#!/usr/bin/env python
# coding: utf-8

# In[159]:


import numpy as np
from numpy import array
from numpy import corrcoef
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys


# In[160]:


def measure_correlation(v1,v2):
    return corrcoef(v1,v2)[0,1]


from sklearn.preprocessing import MinMaxScaler




dataset = pd.DataFrame()

#Read the csv file
path = (os.path.abspath(".")) + "/../../../dataset/wine/"
filename = "wine.csv"
file = path + str(filename)
dataset = pd.read_csv(file, delimiter=",", header=None)


from sklearn.preprocessing import StandardScaler
print(dataset.columns)

features = [1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13]
# Separating out the features
x = dataset.loc[:, features].values
print(x)
# Separating out the target
y = dataset.loc[:,[0]].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, dataset[[0]]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [1, 2, 3]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf[0] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()



columns=[0]
dataset.drop(columns, inplace=True, axis=1)
print(dataset.head(5))
print(dataset.columns)

dataset.describe()

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(dataset)
dataset.loc[:,:] = scaled_values

#print(dataset.corr())

plt.imshow(dataset.T.corr(), cmap=plt.cm.Reds, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(dataset.columns))]
plt.xticks(tick_marks, dataset.columns, rotation='vertical')
plt.yticks(tick_marks, dataset.columns)


print(dataset.T.corr())
plt.imshow(dataset.T.corr())
plt.show()










