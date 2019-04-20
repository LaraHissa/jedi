#!/usr/bin/env python
# coding: utf-8

# In[233]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys


# In[234]:


from sklearn.preprocessing import MinMaxScaler


# In[235]:


dataset = pd.DataFrame()
#Read the csv file
path = (os.path.abspath(".")) + "/../../../dataset/super_trunfo/s4/"
filename = "s4.csv"
file = path + str(filename)
dataset = pd.read_csv(file, delimiter=";", dtype={
    'type': str, 'name': str, 'speed':float, 'hp':float, 'kW':float,
    't100':float, 'cc':float, 'cil':float, 'weight':float
   })


# In[236]:


dataset.head(10)


# In[237]:


dataset.columns


# In[238]:


dataset = dataset.sort_values(by = ['speed'], ascending=False)
dataset.head(10)


# In[239]:


dataset[['speed', 'hp']]


# In[240]:


dataset = dataset.sort_values(dataset.columns[3], ascending=False)
dataset.head(10)


# In[241]:


dataset.describe()


# In[242]:


df = dataset


# In[243]:


df.columns


# In[244]:


columns = ['type', 'name']
df.drop(columns, inplace=True, axis=1)
df.head(5)


# In[245]:


df.describe()


# In[246]:


scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df)
df.loc[:,:] = scaled_values


# In[247]:


df.head(5)


# In[248]:


df.tail(5)


# In[249]:


df.describe()


# In[250]:


df = df.sort_values(df.columns[0])


# In[251]:


df = df.sort_values(by = ['speed'])
print(df)

# In[252]:


df


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





# In[ ]:





# In[ ]:
