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


# In[80]:


Item_MRP_OHE = pd.get_dummies(pd.Series(list(df['Item_MRP_bucket'])), drop_first=True)


# In[81]:


Item_Type_OHE_1 = pd.get_dummies(pd.Series(list(df['Item_Type_1'])),drop_first=True)


# In[82]:


Item_Fat_OHE = pd.get_dummies(pd.Series(list(df['Item_Fat_Content'])),drop_first=True)


# In[83]:


Item_Type_OHE = pd.get_dummies(pd.Series(list(df['Item_Type'])),drop_first=True)


# In[84]:


Outlet_Type_OHE = pd.get_dummies(pd.Series(list(df['Outlet_Type'])),drop_first=True)


# In[85]:


Outlet_Identifier_OHE = pd.get_dummies(pd.Series(list(df['Outlet_Identifier'])),drop_first=True)


# In[86]:


df_new = df[['Item_MRP','Item_Outlet_Sales','Item_Visibility','Item_Weight','Established_Years','Outlet_Size_Category','Outlet_Location_Category','Item_MRP_weight']]


# In[87]:


Item_MRP_OHE.rename(columns={"Fourth": "Item_MRP_Fourth", "Second": "Item_MRP_Second", "Third": "Item_MRP_Third"},inplace=True)


# In[88]:


Item_Fat_OHE.rename(columns={"Regular": "Fat_Content_Regular"},inplace=True)


# ### Data Preprocess - Removing Skewness and Scaling

# In[89]:


df['Item_MRP_weight']


# In[90]:


df.head(2)


# In[91]:


#Removing Skewness in Visibility and Item_MRP_weight by taking log
df['Item_Visibility'] = np.log(df['Item_Visibility']+1)
df['Item_MRP_weight'] = np.log(df['Item_MRP_weight']+1)


# In[92]:


df.head(2)


# In[93]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Visibility_array = np.array(df['Item_Visibility'])
Visibility_array = Visibility_array.reshape(-1,1)
Visibility_scaled = scaler.fit_transform(Visibility_array)
Visibility_scaled = Visibility_scaled.reshape(-1).tolist()
df['Item_Visibility'] = Visibility_scaled


# In[94]:


scaler = StandardScaler()
Item_MRP_weight_array = np.array(df['Item_MRP_weight']).reshape(-1,1)
Item_MRP_weight_scaled = scaler.fit_transform(Item_MRP_weight_array)
Item_MRP_weight_scaled = Item_MRP_weight_scaled.reshape(-1).tolist()
df['Item_MRP_weight'] = Item_MRP_weight_scaled


# In[95]:


scaler = StandardScaler()
Item_outletsales_array = np.array(df['Item_Outlet_Sales']).reshape(-1,1)
Item_outletsales_scaled = scaler.fit_transform(Item_outletsales_array)
Item_outletsales_scaled = Item_outletsales_scaled.reshape(-1).tolist()
#df['Item_MRP_weight'] = Item_outletsales_scaled


# In[96]:


df.head(1)


# In[97]:


df_final = df[['Item_Identifier','Outlet_Identifier','Item_Visibility','Item_Outlet_Sales','Item_MRP_weight','Established_Years','Outlet_Size_Category','Outlet_Location_Category']]


# In[98]:


df_final.reset_index(drop=True, inplace=True)


# In[99]:


df_final.head(1)


# In[100]:


df_a = (((Item_MRP_OHE.join(Item_Type_OHE_1)).join(Item_Fat_OHE)).join(Outlet_Type_OHE)).join(Outlet_Identifier_OHE)


# In[101]:


df_finally = df_final.join(df_a)


# In[110]:


df_finally


# In[103]:


train_index = df_finally[df_finally['Item_Outlet_Sales'].isnull()].index.tolist()


# In[104]:


df_train = df_finally.drop(train_index)


# In[105]:


df_train.info()


# In[106]:


test_index = df_finally[df_finally['Item_Outlet_Sales'].notnull()].index.tolist()


# In[107]:


df_test = df_finally.drop(test_index)


# In[108]:


df_test.reset_index(drop=True, inplace= True)


# In[109]:


corr = df_train.corr()
corr.style.background_gradient(cmap='coolwarm')


# ##### Strong correlation of Item_Outlet_Sales with other variables:
# Item_MRP_weight, Item_MRP_Fourth, Type3, OUT019 and OUT027

# ##### We will build the following models in the next sections.
# 
# Linear Regression, 
# Lasso Regression, 
# Ridge Regression, 
# RandomForest, 
# XGBoost

# 1. Using Linear Regression with k-fold cross validation

# In[111]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[112]:


df_train.columns


# In[113]:


X_train = df_train.loc[:, df_train.columns != 'Item_Outlet_Sales']
Y_train = df_train.loc[:, df_train.columns == 'Item_Outlet_Sales']
X_test = df_test.loc[:,df_test.columns!='Item_Outlet_Sales']


# In[114]:


X_train.head(2)


# In[115]:


del X_train['Item_Identifier']


# In[116]:


del X_train['Outlet_Identifier']


# In[117]:


regressor = LinearRegression()


# In[118]:


model_regressor = regressor.fit(X_train, Y_train)


# In[119]:


regressor.intercept_


# In[120]:


regressor.coef_


# In[121]:


X_test_2 = X_test[['Item_Identifier','Outlet_Identifier']]


# In[122]:


del X_test['Item_Identifier']


# In[123]:


del X_test['Outlet_Identifier']


# In[124]:


Y_pred = regressor.predict(X_test)


# In[125]:


Y_pred_2 = pd.DataFrame(Y_pred.reshape(-1).tolist())


# In[126]:


X_test_2 = X_test_2.join(Y_pred_2)


# In[127]:


X_test_2.rename(columns = {0:'Item_Outlet_Sales'},inplace=True)


# In[128]:


X_test_2


# In[129]:


from pandas import ExcelWriter


# In[303]:


writer = ExcelWriter('Results.xlsx')
X_test_2.to_excel(writer,index=False)
writer.save()


# In[ ]:


#Submitted the solution to the competition website and got the RMSE: 1424.8637599265


# ##### Using k-fold Cross validation

# In[130]:


from sklearn.model_selection import KFold


# In[131]:


kfold = KFold(n_splits=5)


# In[132]:


df_kfold = df_train.copy(deep=True)


# In[133]:


target_df_kfold = pd.DataFrame(df_kfold['Item_Outlet_Sales'])


# In[134]:


del df_kfold['Item_Identifier']
del df_kfold['Outlet_Identifier']
del df_kfold['Item_Outlet_Sales']


# In[135]:


def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(k_x_train, k_y_train)
    return model.score(k_x_test, k_y_test)


# In[136]:


kfold_scores = []
for kfold_train_index, kfold_test_index in kfold.split(df_kfold):
    print(kfold_train_index)
    print(kfold_test_index)
    #k_x_train, k_x_test, k_y_train, k_y_test = df_kfold[kfold_train_index], df_kfold[kfold_test_index], target_df_kfold[kfold_train_index], target_df_kfold[kfold_test_index]
    #kfold_scores.append(get_score(LinearRegression(), k_x_train, k_x_test, k_y_train, k_y_test))
    
    


# ##### Regularized Regression Models

# ###### Ridge Regularization

# In[137]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.5, normalize=True)

ridge.fit(X_train, Y_train)
ridge_predict = ridge.predict(X_test)


# In[138]:


ridge_predict


# In[139]:


ridge_predict = pd.DataFrame(ridge_predict.reshape(-1).tolist())


# In[140]:


try: 
    del X_test_2['Item_Outlet_Sales']
except:
    pass


# In[141]:


X_test_2 = X_test_2.join(ridge_predict)


# In[142]:


X_test_2.rename(columns = {0:'Item_Outlet_Sales'},inplace=True)


# In[143]:


writer = ExcelWriter('Ridge_Results.xlsx')
X_test_2.to_excel(writer,index=False)
writer.save()


# In[ ]:


#Submitted the solution to the competition website and got the RMSE:1285.605


# ###### Using Lasso Regularization

# In[144]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.05, normalize=True)

lasso.fit(X_train, Y_train)
lasso_predict = lasso.predict(X_test)


# In[145]:


lasso_predict = pd.DataFrame(lasso_predict.reshape(-1).tolist())


# In[146]:


try: 
    del X_test_2['Item_Outlet_Sales']
except:
    pass


# In[147]:


X_test_2 = X_test_2.join(lasso_predict)


# In[148]:


X_test_2.rename(columns = {0:'Item_Outlet_Sales'},inplace=True)


# In[149]:


writer = ExcelWriter('Lasso_Results.xlsx')
X_test_2.to_excel(writer,index=False)
writer.save()


# In[386]:


#Submitted the solution to the competition website and got the RMSE: 1219.6827


# ###### Using Random Forest

# In[152]:


from sklearn.ensemble import RandomForestRegressor


# In[172]:


rf_model = RandomForestRegressor(n_estimators=350, bootstrap=True)
rf_model.fit(X_train, Y_train)
rf_predict = rf_model.predict(X_test)


# In[173]:


rf_predict = pd.DataFrame(rf_predict.reshape(-1).tolist())


# In[174]:


try: 
    del X_test_2['Item_Outlet_Sales']
except:
    pass


# In[175]:


X_test_2 = X_test_2.join(rf_predict)


# In[176]:


X_test_2.rename(columns = {0:'Item_Outlet_Sales'},inplace=True)


# In[177]:


writer = ExcelWriter('Random_Forest_Results.xlsx')
X_test_2.to_excel(writer,index=False)
writer.save()


# In[178]:


#Submitted the solution to the competition website and got the RMSE: 1241.7697

