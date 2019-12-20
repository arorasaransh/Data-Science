#!/usr/bin/env python
# coding: utf-8

# #### According to a study, companies with accurate sales predictions are 10% more likely to grow their revenue year-over-year and 7.3% more likely to hit quota.

# #### The data scientists at BigMart have collected sales data for 1559 products across 10 stores in different cities for the year 2013. Now each product has certain attributes that sets it apart from other products. Same is the case with each store. The aim is to build a predictive model to find out the sales of each product at a particular store so that it would help the decision makers at BigMart to find out the properties of any product or store, which play a key role in increasing the overall sales.

# In[332]:


#Loading Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from datetime import datetime


# In[232]:


#import os
#os.chdir('/Users/saransharora/Desktop/AnalyticsVidhya')
#os.getcwd()


# In[233]:


#Importing csv files
train = pd.read_csv(r"train_kOBLwZA.csv")
test = pd.read_csv("test_t02dQwI.csv")
submission = pd.read_excel("submission.xlsx")


# In[234]:


#checking the dimention of the dataset

#Training DataSet
shape_train = train.shape
ndim_train = train.ndim
print(shape_train)
print(ndim_train)

shape_test = test.shape
ndim_test = test.ndim
print(shape_test)
print(ndim_test)

#The 12th variable in the training dataset is the target variable. 


# In[235]:


#Glancing over the columns in both training and testing dataset

print(train.columns)
print(test.columns)


# In[236]:


#Knowing the structure of the dataframe - train and test 
#Checking the datatypes

print(train.info())
print(test.info())

#We can notice missing values in both the train and the test dataframes. 


# In[237]:


#Combining both the dataframes to perform modification to data and visualize

df = train.append(test)


# In[238]:


#Checking shape - Number of rows and columns
print(df.shape)


# ###### Exploratory Data Analysis

# In[239]:


#Plotting the target variable: Item_Outlet_Sales

sns.distplot(df['Item_Outlet_Sales'],bins=50,kde=False)

#The data is skewed towards the right


# In[240]:


df.head()


# In[ ]:





# In[241]:


df.columns


# In[242]:


#Plotting the Independent variables to see the distribution


# In[243]:


sns.distplot(df['Item_Visibility'],bins=50,kde=False)

#The data is skewed twowards the right and would need to be looked into. 


# In[244]:


sns.distplot(df['Item_MRP'],bins=50,kde=False)

#The MRP is divided into 4 categories each. 


# In[245]:


sns.distplot(df['Item_Weight'],bins=50,kde=False)

#There is no clear pattern for the item weight distribution. 


# In[246]:


# Plotting categorical values


# In[247]:


sns.countplot(x = df['Item_Fat_Content'])


# In[248]:


#The itemized fat content is not properly labelled. Low Fat, low fat and LF are the same while Reguler and reg are the same. 
#Converting Low Fat, low fat and LF to Low Fat and Regular, reg to Regular. 


# In[249]:


df['Item_Fat_Content'].replace('low fat','Low Fat',inplace=True)
df['Item_Fat_Content'].replace('LF','Low Fat',inplace=True)
df['Item_Fat_Content'].replace('reg','Regular',inplace=True)


# In[250]:


sns.countplot(x = df['Item_Fat_Content'])


# In[251]:


print(df['Item_Type'].value_counts(dropna=False))
df['Item_Type'].value_counts(dropna=False).plot(kind='barh')


# In[ ]:





# In[252]:


print(df['Outlet_Identifier'].value_counts(dropna=False))
df['Outlet_Identifier'].value_counts(dropna=False).plot(kind='barh')


# In[253]:


print(df['Outlet_Location_Type'].value_counts(dropna=False))
df['Outlet_Location_Type'].value_counts(dropna=False).plot(kind='barh')


# In[254]:


print(df['Outlet_Type'].value_counts(dropna=False))
df['Outlet_Type'].value_counts(dropna=False).plot(kind='barh')


# In[255]:


print(df['Outlet_Size'].value_counts(dropna=False))
df['Outlet_Size'].value_counts(dropna=False).plot(kind='barh')

#There are only 4655 values at max for Outlet_Size variable. It means that we have missing values: 4016 Missing Values 


# In[256]:


print(df['Outlet_Establishment_Year'].value_counts(dropna=False))
df['Outlet_Establishment_Year'].value_counts(dropna=False).plot(kind='barh') 

#Inconsistent number of establishments for the year 1998


# ##### Bi-variate relationship between the predictors and the target variable. 

# ###### Target Variable vs Independent Numerical Data 

# In[257]:


sns.scatterplot(x=df['Item_MRP'],y=df['Item_Outlet_Sales'])


# In[260]:


sns.scatterplot(y=df['Item_Visibility'],x=df['Item_MRP'])

#This does not involve the target variable, but defines a relationship between Item Visibility and it's MRP. 


# In[261]:


df.head(2)


# ###### Target Variable vs Independent Categorical Data 

# In[262]:


#Box plot is used to see the distribution of data and outliers.  


# In[263]:


sns.boxplot(y="Item_Fat_Content",x="Item_Outlet_Sales", data=df)


# In[264]:


sns.boxplot(y="Item_Type",x="Item_Outlet_Sales", data=df)


# In[265]:


sns.boxplot(y="Outlet_Identifier",x="Item_Outlet_Sales", data=df)

#OUT010 and OUT019 have distinct data distribution
#OUT027 has many outliers.


# In[266]:


sns.boxplot(y="Outlet_Location_Type",x="Item_Outlet_Sales", data=df)


# In[267]:


sns.boxplot(y="Outlet_Size",x="Item_Outlet_Sales", data=df)


# In[268]:


sns.boxplot(y="Outlet_Type",x="Item_Outlet_Sales", data=df)


# ###### Treating missing values

# In[269]:


print(df['Outlet_Size'].value_counts(dropna=False))


# In[270]:


#Checking the distribution for missing values in Outlet_Size


# In[271]:


missing_value = df['Outlet_Size'].to_list()


# In[272]:


a = pd.DataFrame(missing_value)


# In[273]:


a = a.replace(np.NaN,"MissingValue")


# In[274]:


df['Outlet_Size'] = a


# In[275]:


print(df['Outlet_Size'].value_counts(dropna=False))


# In[276]:


sns.boxplot(x="Outlet_Size", y="Item_Outlet_Sales", data=df)


# In[277]:


#The distribution for missing value is like the distribution for Small outlet size. Hence we can replace the MissingValue with "Small"


# In[278]:


df = df.replace("MissingValue","Small")


# In[279]:


print(df['Outlet_Size'].value_counts(dropna=False))


# In[280]:


sns.boxplot(x="Outlet_Size", y="Item_Outlet_Sales", data=df)


# In[281]:


#Missing Values for Item Weight


# In[282]:


print(df['Item_Weight'].value_counts(dropna=False))


# In[283]:


#Replacing NaN values with mean of Item_Weight
df['Item_Weight'].fillna(df['Item_Weight'].mean(),inplace=True)


# In[284]:


#Let's have a look at the univariate distribution now for Item_Weight
sns.distplot(df['Item_Weight'],bins=50,kde=False)


# In[285]:


#Let's have a look at the visibility. As we can notice below, there are articles which have 0 visibility but are selling. This is not possible. Hence we need to clean this anamoly and fix the 0 values.


# In[286]:


sns.scatterplot(x=df['Item_Visibility'],y=df['Item_Outlet_Sales'])


# In[287]:


df['Item_Visibility'].value_counts()
#879 rows with 0 visibility


# In[288]:


#Let's fill these 0 values with the median of the Visibility Column


# In[289]:


df['Item_Visibility'].replace(0,df['Item_Visibility'].median(), inplace=True)


# In[290]:


#Let's look at the distribution for Visibility again now.
sns.scatterplot(x=df['Item_Visibility'],y=df['Item_Outlet_Sales'])


# ##### Feature Engineering

# In[291]:


df.head()


# In[292]:


# We can introduce more features using the exisiting variables to improve the model's performance. 


# In[306]:


Item_Identifier = df['Item_Identifier'].values


# In[307]:


Item_Identifier = list(Item_Identifier)


# In[308]:


#The initial two letters in the Item Identifier can be used as a new feature. 

Item_ID_Initial = list(map(lambda x:x[1:3], Item_Identifier))
Item_ID_Initial = pd.DataFrame(Item_ID_Initial)
Item_ID_Initial.rename(columns={0:'Item_ID'},inplace=True)
df["Item_ID_Initial"] = Item_ID_Initial


# In[309]:


#Item_Type can also be divided into perishable and non perishable food items. Let's have a look

df['Item_Type'].value_counts()


# In[302]:


def Item_Type(Type):
    if ('Fruits and Vegetables' in Type or 'Dairy' in Type or 'Meat' in Type or 'Breads' in Type or 'Breakfast' in Type or 'Seafood' in Type):
        return('Perishable')
    else:
        return('NonPerishable')
    

df["Item_Type_1"] = df.apply(lambda x: Item_Type(x['Item_Type']), axis=1)


# In[342]:


df.head(5)


# In[311]:


df.columns


# In[318]:


# Dividing the Item_MRP into 4 categories, since the distribution has 4 categories. See below:

sns.scatterplot(x=df['Item_MRP'],y=df['Item_Outlet_Sales'])


# In[328]:


mrp = list(df['Item_MRP'])


# In[325]:


#Identifying values based on the histogram and glancing across the list for Item_MRP
#First: 61
#Second: 130
#Third: 200

def Item_MRP(MRP):
    if MRP<61:
        return('First')
    elif MRP<130:
        return('Second')
    elif MRP<200:
        return('Third')
    else:
        return('Fourth')
    
df["Item_MRP_bucket"] = df.apply(lambda x:Item_MRP(x['Item_MRP']),axis=1)


# In[352]:


#Making a new column for years in establishment
currentyear = datetime.now().strftime('%Y')
established_year = list(df['Outlet_Establishment_Year'])


# In[360]:


established_year_list = list(map(lambda x:int(currentyear)-x, established_year))
df["Established_Years"] = established_year_list


# In[372]:


df.head(10)


# In[368]:


#Label Encoding and One Hot Encoder
#Using Manual Label Encoding for Outlet_Size and Outlet_Location_Type

#Label Encoding Outlet_Size

def outlet_size(size):
    if size == 'Small':
        return 0
    if size == 'Medium':
        return 1
    if size == 'High':
        return 2

df['Outlet_Size_Category'] = df.apply(lambda x:outlet_size(x['Outlet_Size']),axis=1)



#Label Encoding Outlet_Location_type

def outlet_location(loc):
    if loc == 'Tier 1':
        return 0
    if loc == 'Tier 2':
        return 1
    if loc == 'Tier 3':
        return 2
    
df['Outlet_Location_Category'] = df.apply(lambda x:outlet_location(x['Outlet_Location_Type']),axis=1)


# In[373]:


df.columns


# In[374]:


df.head(2)


# In[377]:


from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




