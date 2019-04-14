#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[7]:


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


# In[8]:


s1 = np.array([[1,0.92,0.95,1], 
               [0.92,0.52,0.65,0.74],
               [0.82,0.44,0.65,0.74],
               [0.74,0.32,0.65,0.75]
              ], np.float)


# In[36]:


for x in range(4):
    for y in range(4):
        print("cor(",x,",", y, ")", (measure_correlation(s1[x],s1[y])))
    print("------")


# In[55]:


m=np.zeros((4,4))

for x in range(4):
    line = ""
    for y in range(4):
        m[x,y]=measure_correlation(s1[x],s1[y])
        value = measure_correlation(s1[x],s1[y])
        value = round(value,3)
        line = line + str(value) + " "
        
    #print(line)
#print (m)
plt.imshow(m)
plt.show()


# In[43]:


x = round(5.76543, 2)
print(x)


# In[31]:


print((measure_correlation(s1[0],s1[1])))
print((measure_cossine(s1[0],s1[1])))
plt.plot(s1[0], s1[1], 'ro')   
plt.show()


# In[ ]:





# In[16]:


print((measure_correlation(s1[0],s1[2])))
print((measure_cossine(s1[0],s1[2])))
plt.plot(s1[0], s1[2], 'ro')   
plt.show()


# In[ ]:





# In[ ]:





# In[4]:


plt.plot(s1[0], s1[1], 'ro')   
x1, y1 = [1, 4], [5, 1]
plt.plot(x1, y1, marker = 'o')
plt.show()


# In[5]:


plt.plot(s2[0], s2[1], 'ro')   
x1, y1 = [1, 4], [5, 1]
plt.plot(x1, y1, marker = 'o')
plt.show()


# In[6]:


plt.plot(s3[0], s3[1], 'ro')
x1, y1 = [1, 4], [5, 1]
plt.plot(x1, y1, marker = 'o')
plt.show()


# In[7]:


plt.plot(s4[0], s4[1], 'ro')  
x1, y1 = [1, 4], [5, 1]
plt.plot(x1, y1, marker = 'o')
plt.show()


# In[8]:


df1 = pd.DataFrame(np.transpose(s1))
df1.corr()
np.linalg.det(df1.corr())


# In[9]:


df2 = pd.DataFrame(np.transpose(s2))
df2.corr()
np.linalg.det(df2.corr())


# In[10]:


df3 = pd.DataFrame(np.transpose(s3))
df3.corr()
np.linalg.det(df3.corr())


# In[11]:


df4 = pd.DataFrame(np.transpose(s4))
df4.corr()
np.linalg.det(df4.corr())


# In[12]:


print((measure_correlation(s1[0],s1[1])))
print((measure_correlation(s2[0],s2[1])))
print((measure_correlation(s3[0],s3[1])))


# In[13]:


print(dist_euclidian(s1[0][14],s1[1][14]))
print(dist_euclidian(s2[0][16],s2[1][16]))
print(dist_euclidian(s3[0][14],s3[1][14]))


# In[14]:


s1[0]


# In[15]:


s1[1]


# In[16]:


S1 = df1.corr()
S2 = df2.corr()
S3 = df3.corr()
S4 = df4.corr()


# In[17]:


p1 = (s1[0][14],s1[1][14])
p1


# In[18]:


p2 = (s1[0][15],s1[1][15])
p2


# In[19]:


de = dist_euclidian(p1,p2)
de_desc = "| [distancia]"
print(dist_euclidian(p1,p2))


# In[20]:


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


# In[21]:


print(abs(1 -  dm1/de) * 100)
print(abs(1  - dm2/de) * 100)
print(abs(1  - dm3/de) * 100)
print(abs(1  - dm4/de) * 100)


# In[ ]:




