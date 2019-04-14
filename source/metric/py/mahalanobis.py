#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# In[50]:


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


# In[51]:


s1 = np.array([[1, 2, 3, 4, 5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 4, 1], 
               [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 5, 1]
              ], np.float)
s1_desc = "| [correlacao cresc]"

s2 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                4, 1
               ], 
               
               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                5, 1
                ]
              ], np.float)
s2_desc = "| [correlacao nula]"

s3 = np.array([[1, 2, 3, 4, 5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 4, 1], 
               [28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 5, 1]
              ], np.float)
s3_desc = "| [correlacao desc"

s4 = np.array([[1, 2, 3, 4, 5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 4, 1], 
              [2800, 2600, 2400, 2200, 2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 200, 5, 1]
              ], np.float)
s4_desc = "| [correlacao  desc + "


# In[52]:


plt.plot(s1[0], s1[1], 'ro')   
x1, y1 = [4, 1], [5, 1]
plt.plot(x1, y1, marker = 'o')
plt.show()


# In[53]:


plt.plot(s2[0], s2[1], 'ro')   
x1, y1 = [4, 1], [5, 1]
plt.plot(x1, y1, marker = 'o')
plt.show()


# In[54]:


plt.plot(s3[0], s3[1], 'ro')
x1, y1 = [4, 1], [5, 1]
plt.plot(x1, y1, marker = 'o')
plt.show()


# In[55]:


plt.plot(s4[0], s4[1], 'ro')  
x1, y1 = [4, 1], [5, 1]
plt.plot(x1, y1, marker = 'o')
plt.show()


# In[56]:


df1 = pd.DataFrame(np.transpose(s1))
df1.corr()
np.linalg.det(df1.corr())


# In[57]:


df2 = pd.DataFrame(np.transpose(s2))
df2.corr()
np.linalg.det(df2.corr())


# In[58]:


df3 = pd.DataFrame(np.transpose(s3))
df3.corr()
np.linalg.det(df3.corr())


# In[59]:


df4 = pd.DataFrame(np.transpose(s4))
df4.corr()
np.linalg.det(df4.corr())


# In[60]:


print((measure_correlation(s1[0],s1[1])))
print((measure_correlation(s2[0],s2[1])))
print((measure_correlation(s3[0],s3[1])))


# In[61]:


print(dist_euclidian(s1[0][14],s1[1][14]))
print(dist_euclidian(s2[0][16],s2[1][16]))
print(dist_euclidian(s3[0][14],s3[1][14]))


# In[62]:


s1[0]


# In[63]:


s1[1]


# In[64]:


S1 = df1.corr()
S2 = df2.corr()
S3 = df3.corr()
S4 = df4.corr()


# In[65]:


p1 = (s1[0][14],s1[1][14])
p1


# In[66]:


p2 = (s1[0][15],s1[1][15])
p2


# In[67]:


de = dist_euclidian(p1,p2)
de_desc = "| [distancia]"
print(dist_euclidian(p1,p2))


# In[68]:


dm1 = distance.mahalanobis(p1, p2, S1)
dm2 = distance.mahalanobis(p1, p2, S2)
dm3 = distance.mahalanobis(p1, p2, S3)
dm4 = distance.mahalanobis(p1, p2, S4)

print("distance euclidiana")
print(de, de_desc)
print("distance mahalanobis")
print(distance.mahalanobis(p1, p2, S1), s1_desc)
print(distance.mahalanobis(p1, p2, S2), s2_desc)
print(distance.mahalanobis(p1, p2, S3), s3_desc)
print(distance.mahalanobis(p1, p2, S4), s4_desc)


# In[69]:


print(abs(1 -  dm1/de) * 100)
print(abs(1  - dm2/de) * 100)
print(abs(1  - dm3/de) * 100)
print(abs(1  - dm4/de) * 100)


# In[ ]:




