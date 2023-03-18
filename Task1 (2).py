#!/usr/bin/env python
# coding: utf-8

# Name:- Jyoti Suresh Kurade

# Data Science and Buisness Analytics Internship

# TSF GRIPMARCH23

# Task1:- Pridiction Using Supervised ML(Pridict the percentage of student based on the number of study         hours.)
# 
# Algorithm:- Linear Regresion
# 
# 1) Required Libraries

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 2) Import Data using URL

# In[6]:


url="http://bit.ly/w-data"
data=pd.read_csv(url)
data


# 3) Data Cleaning

# In[7]:


print(data.shape)
print("Name of Columns: ",data.columns)


# In[8]:


data.isna().values.any()


# In[9]:


data.duplicated().sum()


# In[10]:


import seaborn as sns
sns.boxplot(data['Hours'])
plt.show


# In[11]:


sns.boxplot(data['Scores'])


# There is no missing values, duplicates and outliers present in the data, so no need to clean the data.We can procced for furthur analysis.

# 4) Dependent and Independent variables

# In[12]:


y=data[['Scores']]
x=data[['Hours']]


# 5) Data Exploration

# In[13]:


data.plot(x='Hours',y='Scores',style='o')
plt.xlabel('no.of study hours')
plt.ylabel('percentage scores')


# From above scatterplot, we clearly see that there is possitive linear relationship between number of study hours and percentage scores.

# 6) Splitting data into train and test data 

# In[14]:



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)


# 7) Training of data

# In[15]:


from sklearn.linear_model import LinearRegression
ls=LinearRegression()


# In[23]:


ls.fit(x_train, y_train)
ls.coef_


# In[24]:


ls.intercept_


# In[25]:


line = ls.coef_*x + ls.intercept_  #plotted best fit line for study hours and precentage score
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# 8) Pridiction using Linear Regression

# In[37]:


print(x_test)
print(y_test)
y_pred = ls.predict(x_test)
y_pred


# In[39]:


df=pd.DataFrame(y_test.values.reshape(-1),y_pred.reshape(-1))
df=df.reset_index()
df.columns=["Actual","Predicted"]
df


# 9) Model Evaluation

# In[27]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[41]:


print("R_square:",r2_score(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))


# 10) Predict the score when study hours are 9.25

# In[42]:


hours=9.25
score=ls.predict([[hours]])
print(f"Number of Hours : {hours}")
print(f"Predicted Score : {score}")


# 11) Conclusion

# 1) Here, R_square is 0.9454 which shows that 94.55% of the variation in dependent variable is explained by independent variable.    That is our model gives 94.55% accurate result.
# 
# 2) Model predicts that if a student studies for 9.25 hrs/day then, he/she will get 93.69% score.

# THANK YOU !
