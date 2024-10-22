#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


#DOWNLOADING CSV FILE FORM HOSTED SERVER ONLINE
import requests

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
r = requests.get(url)

with open('FuelConsumption.csv', 'wb') as f:
    f.write(r.content)


# In[11]:


#READING THE DATA
df = pd.read_csv("FuelConsumption.csv")

df.head()


# In[12]:


#SELECT FEATURES TO USE FOR REGRESSION
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# In[13]:


#PLOTTING EMISSION VALUES WITH RESPECT TO ENGINESIZE TO DETERMINE LINEAR RELATIONSHIP
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[14]:


#SELECTING TRANING AND TESTING DATA.80% FOR TRAINING AND 20% FOR TESTING, MUTUALLY EXCLUSIVE. USE THE RANDOM.RAND() FUNCTION

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[15]:


#PLOTTING THE TRAIN DATA DISTRIBUTION,ONLY ENGINESIZE(ONE VARIABLE IS SIMPLE LINEAR REGRESSION)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[16]:


#MULTIPLE LINEAR REGRESSION

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


# In[17]:


#PREDICTION
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Mean Squared Error (MSE) : %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# In[18]:


#PRACTICE

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
print ('Coefficients: ', regr.coef_)
y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
print('Variance score: %.2f' % regr.score(x, y))


# In[19]:


plt.scatter(y, y_, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Measured vs Predicted CO2 Emissions')
plt.show()


# In[ ]:




