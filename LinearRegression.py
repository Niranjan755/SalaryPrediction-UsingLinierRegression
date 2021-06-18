#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# In[3]:


salarydata = pd.read_csv(r"OneDrive\Desktop\Machine Learning\Salary_Data.csv")
salarydata.head()


# In[4]:


salarydata.isnull().any()


# In[5]:


plt.scatter(salarydata["YearsExperience"],salarydata["Salary"])


# In[6]:


x = salarydata.iloc[:,0:1].values
y = salarydata.iloc[:,1].values


# In[7]:


x


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x,y,test_size = 0.2, random_state =0)


# In[9]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# In[10]:


ypred = lr.predict(x_test)


# In[11]:


ypred


# In[12]:


y_test


# In[13]:


plt.scatter(x_train,y_train)
plt.plot(x_train,lr.predict(x_train), color = "red")


# In[14]:


from sklearn.metrics import r2_score
accuracy = r2_score(ypred,y_test)


# In[15]:


accuracy


# In[16]:


yp = lr.predict([[10.5]])


# In[17]:


yp


# In[ ]:




