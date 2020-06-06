#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


iris = load_iris()


# In[3]:


X=iris.data
y=iris.target


# In[4]:


iris.target_names


# In[5]:


df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df.loc[df['species'] == 0, 'species'] = 'setosa'
df.loc[df['species'] == 1, 'species'] = 'versicolor'
df.loc[df['species'] == 2, 'species'] = 'virginica'
df.head()


# In[6]:


df.shape


# # Univariate Analysis

# In[7]:


df_setosa = df.loc[df['species'] == 'setosa']
df_versicolor = df.loc[df['species'] == 'versicolor']
df_virginica = df.loc[df['species'] == 'virginica']


# In[8]:


plt.plot(df_setosa['sepal length (cm)'], np.zeros_like(df_setosa['sepal length (cm)']), 'o')
plt.plot(df_virginica['sepal length (cm)'], np.zeros_like(df_virginica['sepal length (cm)']), 'x')
plt.plot(df_versicolor['sepal length (cm)'], np.zeros_like(df_versicolor['sepal length (cm)']), '*')
plt.xlabel('sepal length (cm)')
plt.show()


# # bi-Variate Analysis

# In[9]:


sns.FacetGrid(df,hue='species',size=5).map(plt.scatter, "petal length (cm)", "sepal width (cm)").add_legend()


# # multi Variate Analysis

# In[10]:


sns.pairplot(df,hue='species',size=3)

