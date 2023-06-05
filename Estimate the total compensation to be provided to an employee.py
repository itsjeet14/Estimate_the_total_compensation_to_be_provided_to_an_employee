#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
df = pd.read_csv("train_set.csv")
df


# In[21]:


df.info()


# In[22]:


column_names = ['OG', 'DC','Dept','Union','JF','Job','YT']  # Replace with the actual column names in your dataframe

for column in column_names:
    unique_values = df[column].unique()
    mapping = {value: index for index, value in enumerate(unique_values)}
    df[column] = df[column].replace(mapping)

print(df)


# In[23]:


df.describe()


# In[24]:


df.isnull().sum()


# In[25]:


df.drop(df[df['Union'].isnull()].index, inplace=True)


# In[26]:


df.isnull().sum()


# In[27]:


df.drop(df[df['JF'].isnull()].index, inplace=True)


# In[28]:


df.isnull().sum()


# In[29]:


x = df.drop(['Total_Compensation'],axis=1)
y = df['Total_Compensation']


# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[31]:


x_train.shape


# In[32]:


x_test.shape


# In[33]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)


# In[34]:


y_pred = lr.predict(x_train)
y_pred


# In[35]:


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(x_train, y_train)


# In[36]:


kn_pred = knr.predict(x_test)
kn_pred


# In[37]:


from sklearn.metrics import r2_score
r2_score(y_train, y_pred)


# In[38]:


r2_score(y_test, kn_pred)


# In[42]:


import numpy as np
input_data = (1,2,3,4,5,6,7,8,9,10,11,12,13,14)
convert_to_array = np.asarray(input_data)
re_shape = convert_to_array.reshape(-1,1)
prediction = lr.predict(re_shape)
print(prediction)


# In[ ]:




