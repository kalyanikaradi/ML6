#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sns.get_dataset_names()


# In[3]:


df=sns.load_dataset('iris')


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


import seaborn as sns
df=data


# In[8]:


sns.pairplot(data=df, hue='species')
plt.show()


# In[9]:


X=df.iloc[:,:-1]
X.head()


# In[10]:


#import KMeans algo 
from sklearn.cluster import KMeans
#initial model
model = KMeans(n_clusters=3, random_state=10)

model.fit(X)


# In[11]:


X.shape


# In[12]:


#labels of clustersmodel.labels_


# In[13]:


#centroids of cluster
model.cluster_centers_


# In[14]:


#set colours to clusters to diff(Not required)
color_scheme = np.array(['red','blue','green','yellow','pink','cyan'])
color_scheme


# In[15]:


df.species = df.species.map({'setosa':0,'versicolor':1,'virginica':2})
df


# In[16]:


#visualize cluster
plt.scatter(df.petal_length,df.petal_width,color=color_scheme[df.species])


# In[17]:


#cluster model formed
plt.scatter(X.petal_length,X.petal_width,color=color_scheme[model.labels_]);


# In[18]:


model.labels_


# In[19]:


#dataset
X['Group']=pd.DataFrame(model.labels_)
X.head()


# In[20]:


new_point=[[4.3,3.5,1.5,0.4]]
model.predict(new_point)


# In[21]:


type(new_point)


# In[22]:


#elbow method(imp)(The WCSS is the sum of the squared distances between each data point and the centroid of its assigned cluster.)
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[23]:


#evaluation metrics
label=model.labels_
label

from sklearn.metrics import silhouette_score
score=silhouette_score(X,label)
score


# In[ ]:





# In[ ]:




