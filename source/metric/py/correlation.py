#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


import os
import sys
import math

from datetime import datetime
from datetime import timedelta
import datetime
from scipy import spatial

from numpy import array
from numpy import corrcoef

from scipy.spatial import distance


# In[65]:


def dist_euclidian(v1,v2):
    return distance.euclidean(v1, v2)

def measure_cossine(v1,v2):
    return (1 - spatial.distance.cosine(v1, v2))

def measure_correlation(v1,v2):
    return corrcoef(v1,v2)[0,1]

def dist_correlation(dc):
    return math.sqrt(2 *(1 - dc))

def measure_angle(arcc):
    return math.degrees(np.arccos(arcc))


# In[66]:


s1 = np.array([[1,0.92,0.95,1],
               [0.92,0.52,0.65,0.74],
               [0.82,0.44,0.65,0.74],
               [0.74,0.32,0.65,0.75]
              ], np.float)


# In[67]:


m=np.zeros((4,4))

for x in range(4):
    line = ""
    for y in range(4):
        m[x,y]=measure_correlation(s1[x],s1[y])
        value = measure_correlation(s1[x],s1[y])
        value = round(value,3)
        line = line + str(value) + " "

print (m)


# In[68]:

cmap = colors.ListedColormap(['white', 'red'])

plt.imshow(m, cmap=cmap)
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
