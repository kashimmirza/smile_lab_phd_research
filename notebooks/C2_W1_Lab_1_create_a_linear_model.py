#!/usr/bin/env python
# coding: utf-8

# # Create a Linear Model

# ## Linear model using scikit-learn
# 
# Welcome to the first lab of this course!
# 
# You'll practice using a scikit-learn model for linear regression. You will do something similar in this week's assignment (but with a logistic regression model).
# 
# [sklearn.linear_model.LinearRegression()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

# First, import `LinearRegression`, which is a Python 'class'.

# In[1]:


# Import the module 'LinearRegression' from sklearn
from sklearn.linear_model import LinearRegression


# Next, use the class to create an object of type LinearRegression.

# In[2]:


# Create an object of type LinearRegression
model = LinearRegression()
print(model)


# Generate some data by importing a module 'load_data', which is implemented for you.  The features in `X' are: 
# 
# - Age: (years)
# - Systolic_BP: Systolic blood pressure (mmHg)
# - Diastolic_BP: Diastolic blood pressure (mmHg)
# - Cholesterol: (mg/DL)
# 
# The labels in `y` indicate whether the patient has a disease (diabetic retinopathy).
# - y = 1 : patient has retinopathy.
# - y = 0 : patient does not have retinopathy.

# In[3]:


import helper_utils


# In[4]:


# Generate features and labels
X, y = helper_utils.load_data(100)


# Explore the data by viewing the features and the labels

# In[5]:


# View the features
X.head()


# In[6]:


# Plot a histogram of the Age feature
X['Age'].hist();


# In[7]:


# Plot a histogram of the systolic blood pressure feature
X['Systolic_BP'].hist();


# In[8]:


# Plot a histogram of the diastolic blood pressure feature
X['Diastolic_BP'].hist();


# In[9]:


# Plot a histogram of the cholesterol feature
X['Cholesterol'].hist();


# Also take a look at the labels

# In[10]:


# View a few values of the labels
y.head()


# In[11]:


# Plot a histogram of the labels
y.hist();


# Fit the LinearRegression using the features in `X` and the labels in `y`.  To "fit" the model is another way of saying that we are training the model on the data.

# In[12]:


# Fit the linear regression model
model.fit(X, y)
print(model)


# - View the coefficients of the trained model.
# - The coefficients are the 'weights' or $\beta$s associated with each feature
# - You'll use the coefficients for making predictions.
# $$\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... \beta_N x_N$$

# In[13]:


# View the coefficients of the model
print(model.coef_)


# In the assignment, you will do something similar, but using a logistic regression, so that the output of the prediction will be bounded between 0 and 1.

# ### This is the end of this practice section.
# 
# Please continue on with the lecture videos!
# 
# ---
