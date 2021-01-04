#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sklearn
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import plotly as plot
import plotly.express as px
import plotly.graph_objs as go

import cufflinks as cf

import matplotlib.pyplot as plt

import seaborn as sns
import os

import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot


# In[7]:


pyo.init_notebook_mode(connected=True)
cf.go_offline()


# In[15]:


database = pd.read_csv(r'PATH_TO_heart.csv')


# In[16]:


database


# In[17]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

for i in range(len(info)):
    print(database.columns[i]+":\t\t\t"+info[i])


# In[ ]:





# In[18]:


database['target']


# In[22]:


database.groupby('target').size()


# In[23]:


database.shape


# In[25]:


database.size


# In[27]:


database.describe()


# In[ ]:





# In[ ]:





# In[28]:


database.info()


# In[ ]:





# In[29]:


#Visualization


# In[31]:


database.hist(figsize=(14, 14))
plt.show()


# In[ ]:





# In[34]:


sns.barplot(database['sex'], database['target'])
plt.show()


# In[ ]:





# In[37]:


sns.barplot(database['sex'], df['age'], hue = database[
    'target'
])
plt.show()


# In[ ]:





# In[43]:


numeric_columns = ['trestbps', 'chol', 'age', 'oldpeak', 'thalach']


# In[45]:


sns.heatmap(database[numeric_columns].corr(),annot=True,cmap='terrain',linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()


# In[ ]:





# In[50]:


#creat 4 distplots
#primary data analysis
plt.figure(figsize=(12,10))

plt.subplot(221)
sns.distplot(database[database['target'] == 0].age)
plt.title('Age of patients without Heart Disease')

plt.subplot(222)
sns.distplot(database[database['target'] == 1].age)
plt.title('Age of patients with Heart Disease')

plt.subplot(223)
sns.distplot(database[database['target'] == 0].thalach)
plt.title('Maximum Heart Rate of Patients without Heart Disease')

plt.subplot(224)
sns.distplot(database[database['target'] == 1].thalach)
plt.title('Maximum Heart Rate of Patients with Heart Disease')

plt.show()


# In[ ]:





# In[ ]:





# In[53]:


# Data pre-processing


# In[54]:


x, y = database.loc[:, :'thal'], database['target']


# In[55]:


x


# In[56]:


y


# In[ ]:





# In[59]:


x.size
# 70% for training, 30% for testing


# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, test_size = 0.3, shuffle = True)


# In[64]:


x_test.shape


# In[68]:


x_test


# In[ ]:





# In[69]:


#Decision Tree Classifier


# In[74]:


categories = ['You do not have heart disease', 'You have heart disease']


# In[75]:


from sklearn.tree import DecisionTreeClassifier


# In[76]:


dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)


# In[79]:


prediction = dt.predict(x_test)
accuracy_dt = accuracy_score(y_test, prediction) * 100


# In[80]:


accuracy_dt


# In[82]:


print("Accuracy on training set: {:.3f}".format(dt.score(x_train, y_train)))
print("Accuracy on test set:".format(dt.score(x_test, y_test)))


# In[ ]:





# In[83]:


y_test


# In[84]:


prediction


# In[94]:


X_DT = np.array([[57, 0, 0, 140, 241, 0, 1, 123, 1, 0.2, 1, 0, 3]])
X_DT_prediction = dt.predict(X_DT)


# In[95]:


X_DT_prediction[0]


# In[96]:


print(categories[int(X_DT_prediction[0])])


# In[ ]:




