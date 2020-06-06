#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import numpy as np


# In[2]:


df = sns.load_dataset('tips')
df.head()


# In[3]:


df['total_bill'] = np.float32(df['total_bill'])
df['tip'] = np.float32(df['tip'])
df['size'] = np.int32(df['size'])
df.info()


# In[4]:


df.corr()


# In[5]:


sns.heatmap(df.corr())


# In[6]:


sns.jointplot(x='tip', y='total_bill', data=df,kind='hex')


# In[7]:


sns.jointplot(x='tip', y='total_bill', data=df,kind='reg')


# In[8]:


sns.pairplot(df)


# In[9]:


sns.pairplot(df, hue='sex')


# In[10]:


sns.distplot(df['tip'])


# In[11]:


sns.distplot(df['tip'], kde=False, bins=10)


# # boxplot VoilionPlot CountPlot BarPlot
# ## Catagorial feature

# In[12]:


sns.countplot('sex',data=df)


# In[13]:


sns.countplot('smoker',data=df)


# In[14]:


sns.countplot('day',data=df)


# In[15]:


sns.countplot(y = 'sex',data=df) #bar on y axis


# In[16]:


sns.barplot(x='total_bill', y='sex', data=df)


# In[17]:


sns.boxplot('smoker','total_bill', data=df)


# In[18]:


sns.boxplot(data=df, orient='v')


# In[19]:


sns.boxplot(x='total_bill', y='day', data=df, hue='smoker')


# In[20]:


sns.violinplot(x='total_bill', y='day', data=df, palette='rainbow')

