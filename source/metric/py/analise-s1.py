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


# In[161]:


from sklearn.preprocessing import MinMaxScaler


# In[162]:


dataset = pd.DataFrame() 
#Read the csv file
path = (os.path.abspath(".")) + "/../../../dataset/super_trunfo/s1/"
filename = "s1.csv"
file = path + str(filename)
dataset = pd.read_csv(file, delimiter=";", dtype={
    'id': str, 'gp_disp':float, 'titulos':float, 'vitorias':float, 
    'pole_positions':float, 'voltas_rapidas':float
   })


# In[163]:


dataset


# In[164]:


dataset.columns


# In[165]:


dataset.describe()


# In[166]:


df = dataset


# In[167]:


columns = ['id']
df.drop(columns, inplace=True, axis=1)
df.head(50)


# In[168]:


df.describe()


# In[183]:


scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(df) 
df.loc[:,:] = scaled_values


# In[184]:


df


# In[185]:


print(df.corr())
plt.imshow(df.corr(), cmap=plt.cm.Reds, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df.columns))]
plt.xticks(tick_marks, df.columns, rotation='vertical')
plt.yticks(tick_marks, df.columns)
plt.show()


# In[186]:


print(df.T.corr())
plt.imshow(df.T.corr())
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




