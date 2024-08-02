#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
print(housing)


# In[3]:


housing['data'].head()


# In[4]:


housing['target'].head()


# In[5]:


df = pd.DataFrame(housing['data'])
df


# In[6]:


df['Price'] = housing['target']
df


# In[7]:


df.info()


# In[8]:


df.isna().sum()


# In[12]:


df.describe().T


# In[13]:


df.hist(figsize=(10,8))


# In[14]:


df.corr()


# In[16]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)


# In[17]:


df.drop(['Latitude','Longitude'],axis=1,inplace=True)


# In[18]:


df


# In[19]:


X = df.drop('Price',axis=1)
y = df['Price']


# In[20]:


X


# In[21]:


y


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)


# In[24]:


df.shape


# In[25]:


X_train


# In[26]:


X_test


# In[27]:


y_train.shape


# In[28]:


y_test.shape


# In[29]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[30]:


y_pred = model.predict(X_test)
y_pred


# In[31]:


from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
print(mean_squared_error(y_pred,y_test))


# In[32]:


print(mean_absolute_error(y_pred,y_test))


# In[33]:


r2_score(y_pred,y_test)


# In[34]:


submit = pd.DataFrame({'Y_original':y_test,'Y_predicted':y_pred})
submit


# In[35]:


df


# In[39]:


df.to_csv('Cleaned_housing_data1.csv')


# In[40]:


new_df = pd.read_csv('Cleaned_housing_data1.csv')
new_df


# In[41]:


import numpy as np


# In[42]:


actual = np.array([10,15,20,25,30])
predicted = np.array([12,18,22,28,30])


# In[43]:


np.mean(actual)


# In[45]:


# Mean Squared Error (MSE)
mse = np.mean((actual - predicted)**2)
print('Mean Squared Error:',mse)


# In[46]:


# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:',rmse)


# In[47]:


abs(-12)


# In[48]:


# Mean Absolute Error (MAE)
mae = np.mean(np.abs(actual - predicted))
print('Mean Absolute Error:',mae)


# In[49]:


# R-squared (R2) Score
mean_actual = np.mean(actual)
r2 = 1 - (mse / np.mean((actual - mean_actual)**2))
print('R-squared (R2) Score:',r2)


# In[52]:


# Adjusted R-squared
n = len(actual)
k = 1 # Number of predictors, adjust this value accordingly
adjusted_r2 = 1 - ((1 - r2) * (n -1) / (n - k - 1))
print('Adjusted R-squared:',adjusted_r2)


# ### Assumption of Linear Regression
# Linear regression relies on several assumptions to be valid. Violation of these assumptions can lead to inaccurate results and unreliable predictions. Here are the key assumptions of linear regression:

# #### 1. Linearity : 
# The relationship between the independent variables (features) and the dependent variable (target) should be approximately linear. This means that the change in the target variable is proportional to changes in the predictor variables.

# #### 2. Independence of Errors:
# The errors (residuals) of the model should be independent of each other. In other words, the error for one data point should not depend on the errors of other data points. This assumption is crucial for the validity of statistical tests and confidence intervals.

# #### 3. Homoscedasticity:
# The variance of the errors should be constant across all levels of the independent variables. In simpler terms, the spread of residuals should be roughly the same for all values of the predictors. Heteroscedasticity (varying spread) can lead to unreliable standard errors and p-values.

# #### 4. Normality of Residuals:
# The residuals should follow a normal distribution. This assumption is important for hypothesis testing and constructing confidence intervals. However, linear regression is often robust to moderate deviations from normality, especially with a sufficient sample size.

# #### 5. No or Little Multicollinearity: 
# Multicollinearity occurs when two or more independent variables in the model are highly correlated with each other. This can make it challenging to determine the individual effect of each predictor on the target variable.

# #### 6. No Outliers or Influential Observations:
# Outliers or influential data points can disproportionately affect the regression model, leading to biased parameter estimates. Identifying and handling outliers is important for model robustness.

# In[ ]:




