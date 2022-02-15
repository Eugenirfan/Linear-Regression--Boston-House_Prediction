#!/usr/bin/env python
# coding: utf-8

# In[65]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.api as sm
from sklearn.datasets import load_boston


# In[2]:


boston = load_boston()


# In[6]:


boston


# In[5]:


df = pd.DataFrame(boston.data, columns= boston.feature_names)


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.isna().sum()


# In[17]:


df['Price']=boston.target


# In[18]:


df


# In[19]:


fig, ax = plt.subplots(figsize = (20,15))
sns.heatmap(df.corr(), annot = True, linewidths= 2, fmt = '.2f', ax = ax)
plt.show()


# In[ ]:





# In[20]:


df.hist(figsize = (20,15))
plt.show()


# In[21]:


x = df.drop(['Price'], axis = 1)
y = df['Price']


# In[36]:


for predictor in df.columns:
    df.plot.scatter(x=predictor, y='Price', figsize=(10,5), title=predictor+" VS "+ 'Price')


# In[ ]:





# Feature Selection:

# In[42]:


# Calculating correlation matrix
ContinuousCols=['Price','CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX',
                 'PTRATIO', 'B', 'LSTAT']

# Creating the correlation matrix
CorrelationData=df[ContinuousCols].corr()
CorrelationData


# In[44]:


# Filtering only those columns where absolute correlation > 0.5 with Target Variable
# reduce the 0.5 threshold if no variable is selected
CorrelationData['Price'][abs(CorrelationData['Price']) > 0.5 ]


# In[47]:


# Box plots for continuous Target Variable "MEDV" and Categorical predictors
CategoricalColsList=['RAD', 'CHAS']

import matplotlib.pyplot as plt
fig, PlotCanvas=plt.subplots(nrows=1, ncols=len(CategoricalColsList), figsize=(18,5))

# Creating box plots for each continuous predictor against the Target Variable "MEDV"
for PredictorCol , i in zip(CategoricalColsList, range(len(CategoricalColsList))):
    df.boxplot(column='Price', by=PredictorCol, figsize=(5,5), vert=True, ax=PlotCanvas[i])


# In[48]:


# Defining a function to find the statistical relationship with all the categorical variables
def FunctionAnova(inpData, TargetVariable, CategoricalPredictorList):
    from scipy.stats import f_oneway

    # Creating an empty list of final selected predictors
    SelectedPredictors=[]
    
    print('##### ANOVA Results ##### \n')
    for predictor in CategoricalPredictorList:
        CategoryGroupLists=inpData.groupby(predictor)[TargetVariable].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)
        
        # If the ANOVA P-Value is <0.05, that means we reject H0
        if (AnovaResults[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
    
    return(SelectedPredictors)


# In[49]:


# Calling the function to check which categorical variables are correlated with target
CategoricalPredictorList=['RAD', 'CHAS']
FunctionAnova(inpData=df, 
              TargetVariable='Price', 
              CategoricalPredictorList=CategoricalPredictorList)


# In[50]:


SelectedColumns=['RM', 'PTRATIO','LSTAT', 'RAD', 'CHAS']

# Selecting final columns
DataForML=df[SelectedColumns]
DataForML.head()


# In[53]:


# Treating all the nominal variables at once using dummy variables
DataForML_Numeric=pd.get_dummies(DataForML)

# Adding Target Variable to the data
DataForML_Numeric['Price']=df['Price']

# Printing sample rows
DataForML_Numeric.head()


# In[54]:


# Separate Target Variable and Predictor Variables
TargetVariable='Price'
Predictors=['RM', 'PTRATIO', 'LSTAT', 'RAD', 'CHAS']

X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)


# In[70]:


# Multiple Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
RegModel = LinearRegression()

# Printing all the parameters of Linear regression
print(RegModel)

# Creating the model on Training Data
LREG=RegModel.fit(X_train,y_train)

print('Intercept: ',LREG.intercept_)
print('Intercept: ',LREG.coef_)

y_train_prediction = RegModel.predict(X_train)

y_test_prediction = RegModel.predict(X_test)





# In[63]:


LREG.score(X_train,y_train)


# In[72]:


r2_score(y_train,y_train_prediction)


# In[73]:


r2_score(y_test,y_test_prediction)


# In[74]:


import pickle

filename = 'lr_model.pickle'
pickle.dump(RegModel, open(filename, 'wb'))


# In[75]:


loaded_model = pickle.load(open(filename, 'rb'))


# In[76]:


loaded_model


# In[78]:


a=loaded_model.predict([[6.5,15.3,5,1,0]])


# In[79]:


a


# In[ ]:




