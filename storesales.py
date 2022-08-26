# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:33:31 2022

@author: Soura
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_train= pd.read_csv("Train.csv")
df_test= pd.read_csv("Test.csv")
df_train.columns
df_test.columns

# Concatenating train and test set to a single dataframe for analysis
df= pd.concat([df_train,df_test])
df.describe()
df.isnull().sum()

## Function for a dataset to get percent nan values
def percent_missing(df):
    percent_nan= 100*df.isnull().sum()/len(df)
    percent_nan= percent_nan[percent_nan>0].sort_values()
    
    return percent_nan

percent_nan= percent_missing(df)
percent_nan
## We have 17% na in Item_weight, 28% in Outlet_Size


df.dtypes
df56= df[(df['Outlet_Type']== 'Small') & (df['Outlet_Size']=='Grocery Store')]

## Data visualization

sns.scatterplot(df_train["Item_Visibility"], df_train["Item_Outlet_Sales"])

plt.figure(figsize=(22,11),dpi=200)
sns.barplot(x=df_train["Item_Type"],y=df_train["Item_Outlet_Sales"])
plt.xticks(rotation=90);

sns.boxplot(x="Item_Type", y="Item_Outlet_Sales", data=df_train)
plt.xticks(rotation=90);

sns.boxplot(x ='Outlet_Size', y = 'Item_Outlet_Sales', data = df)
sns.boxplot(x = 'Outlet_Location_Type', y = 'Item_Outlet_Sales', data = df)
sns.boxplot(x = 'Outlet_Identifier', y = 'Item_Outlet_Sales', data = df)
sns.boxplot(x = 'Item_Fat_Content', y = 'Item_Outlet_Sales', data = df)
## Item_Fat has repeating names 

print(df['Item_Identifier'].value_counts())

'''
def replace(df):
    for i in range(df['Item_Identifier']):
        for i in range (df['Item_Weight']):
'''        

## Since we have 17% values as na in Item_Weights
# so we are filling the na values by the mean of Item_Weights

df.groupby('Item_Type')['Item_Weight'].mean()
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace= True)
df.isnull().sum()

df['Outlet_Size'].value_counts()
df['Item_Outlet_Sales'].mean()

## Looking a the mode values (most frequency) of the outlet size for the values of outlet type
mode= df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

missing= df['Outlet_Size'].isnull()
## Hence filling the nan values of the Outlet_Size by the mode of Outlet_Size 
# for the of Outlet_Type each row has

df.loc[missing, 'Outlet_Size'] = df.loc[missing,'Outlet_Type'].apply(lambda x: mode[x])

df.isnull().sum()
## Hence all null values has been filled up and we did not drop any row




## Replacing redundent values in Item_Fat_Content
df.replace(to_replace=['low fat'],value='Low Fat',inplace=True)
df.replace(to_replace=['LF'],value='Low Fat', inplace= True)
df.replace(to_replace=['Reg'],value='Regular', inplace= True)
df['Item_Fat_Content'].value_counts()



## Changing the Outlet_Establishment_Year into the the number of year established 
df["Outlet_Establishment_Year"].min()
df["Outlet_Establishment_Year"].max()

## Assume that the ongoing year is 2022 so we will count years from today
df["Outlet_Establishment_Year"]= (2022- df["Outlet_Establishment_Year"])
df["Outlet_Establishment_Year"]

df.columns
df.dtypes
len(df['Item_Identifier'].value_counts()) ## 1559 distinct unique item identifiers
df['Item_Type'].value_counts()




## Now changing categorical values from strings into integer type
## We can use Label encoding or one hot encoding..
## So, we will do label encoding for the columns on which it makes sense
# on other columns, we will do one hot encoding by splitting our dataframe



df.to_csv("G:\\Python\\GIT\\Store Sales\\combined_df.csv")




############## PREPROCESSING  ############


## Reading our dataset
df= pd.read_csv("final_df.csv")
len(df)
df.isnull().sum()

## Splitting our dataframe into two parts Numeric features and string features
df_nums = df.select_dtypes(exclude='object')
df_objs = df.select_dtypes(include='object')

df_objs = pd.get_dummies(df_objs,drop_first=True) 
# Drop first =True, to solve dummy variable trap (Multicolinearity)


# Now we have created dumy vars, we will concatenate the splitted df
concat_df = pd.concat([df_nums,df_objs],axis=1)


## Seperating train and test set from concatenated df
df_train = concat_df.iloc[:8523,:] 
## We will divide this into training set, Internal validation set and external validation set

df_test = concat_df.iloc[8523:,:] ## We have to predict tem store sale for this set
df_test= df_test.drop(['Item_Outlet_Sales'],axis=1)


df_train.isnull().sum()
df_test.isnull().sum()



############# Our data is ready for modelling 



X= df_train.drop(['Item_Outlet_Sales'], axis=1)
y= df_train['Item_Outlet_Sales']

from sklearn.model_selection import train_test_split

X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.3, random_state=101)
X_eval, X_test, y_eval, y_test= train_test_split(X_other, y_other, test_size=0.5, random_state=101)
## We have divided our train csv into training, calibration and validation set 



####################### MODELLING ###################


    
##################### 1---> USING ELASTIC NET #####################



from sklearn.linear_model import ElasticNet

base_elasticnet_model= ElasticNet()

##param_grid= {"alpha":[0.1,1,5,10,50,100], "l1_ratio":[0.1,0.5,0.7,0.95,0.99,1]}
## Gives alpha=5  and l1_ratio= 1

param_grid= {"alpha":[4.5,5,5.5], "l1_ratio":[0.995, 0.999,1]}
# Gives alpha=4.5  and l1_ratio= 1

from sklearn.model_selection import GridSearchCV
grid_model= GridSearchCV(estimator= base_elasticnet_model, 
                         param_grid= param_grid,
                         scoring= "neg_mean_squared_error",
                         cv= 5, verbose=2)

grid_model.fit(X_train, y_train)


# Now we need best estimator
grid_model.best_estimator_ # Gives alpha=5  and l1_ratio= 1
grid_model.best_params_

gridresults= pd.DataFrame(grid_model.cv_results_) # Statistics for all combinations

## Calibration prediction
y_pred= grid_model.predict(X_eval)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
rmse= np.sqrt(mean_squared_error(y_eval, y_pred))
R_sq= r2_score(y_eval, y_pred)
rmse   ## 1048.40
R_sq   ##  0.59


## Validation prediction
y_val_pred= grid_model.predict(X_test)
len(y_val_pred)
len(X_test)
rmse= np.sqrt(mean_squared_error(y_test, y_val_pred))
R_sq= r2_score(y_test, y_val_pred)
rmse   ## 1103.18
R_sq   ##  0.54

##----> Our model is not overfitting but the RMSE is high and the r2_score is fair 



################# Helper function ##################


def run_model(model, X_train, y_train, X_test, y_test):
    
    # Fit the training model to the test data
    model.fit(X_train,y_train)
    
    # Get metrics
    predictions= model.predict(X_test)
    rmse= np.sqrt(mean_squared_error(y_test, predictions))
    r2= r2_score(y_test, predictions)
    print(f'rmse:{rmse}')
    print(f'r2:{r2}')
    
    
    

#################### SUPPORT VECTOR REGRESSION ##################

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV
svr= SVR()
param_grid={'C':[0.01,0.1,1,10], 'gamma':['auto','scale']}
grid= GridSearchCV(svr, param_grid)

run_model(grid, X_train, y_train, X_test, y_test)


############## RANDOM FOREST REGRESSION ###########


from sklearn.ensemble import RandomForestRegressor
rfr= RandomForestRegressor(n_estimators=250)
# Calibration
run_model(rfr, X_train, y_train, X_eval, y_eval)
#  rmse:  1039
#  r2:  0.60

# Validation
run_model(rfr, X_train, y_train, X_test, y_test)
rmse:1086
r2:0.56

##------------> Model not overfitting but no improvement on RMSE and r2_score

################# BOOSTING #################

from sklearn.ensemble import GradientBoostingRegressor

model= GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred= model.predict(X_other)

rmse= np.sqrt(mean_squared_error(y_other, y_pred))
R_sq= r2_score(y_other, y_pred)
rmse  ## 1042
R_sq  ## 0.595

## So far the best model in terms of RMSE and r2_score

import joblib
joblib.dump(model, 'final_model.pkl')
list(df.columns)
joblib.dump(list(df.columns), 'col_names.pkl')







########################### DIFFERENT APPROACH ########################








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("combined_df.csv")

df.isnull().sum()
df= df.drop("Item_Identifier", axis= 1)
df.columns


## Splitting our dataframe into two parts Numeric features and string features
df_nums = df.select_dtypes(exclude='object')
df_objs = df.select_dtypes(include='object')
df_objs.columns


# Replace Low Fat with 1 and regular with 4
df_objs = df_objs.replace(['Low Fat','Regular'],[1,3])

df_objs = df_objs.replace(['Small','Medium','High'],[1,3,9])
df_nums2= df_objs.drop(['Item_Type' ,'Outlet_Identifier','Outlet_Type','Outlet_Location_Type'], axis=1)

df_objs_ohe= df_objs.drop(['Item_Fat_Content','Outlet_Size'], axis=1)

## Now we have further splitted our dataframe to do label encoding and One hot encoding

df_objs_ohe = pd.get_dummies(df_objs_ohe,drop_first=True) 
# Drop first =True, to solve dummy variable trap (Multicolinearity)

# Now we have created dumy vars, we will concatenate the splitted df
final_df = pd.concat([df_nums,df_objs_ohe, df_nums2],axis=1)


## Seperating train and test set from concatenated df
df_train = concat_df.iloc[:8523,:] 
## We will divide this into training set, Internal validation set and external validation set

df_test = concat_df.iloc[8523:,:] ## We have to predict tem store sale for this set
df_test= df_test.drop(['Item_Outlet_Sales'],axis=1)

df_train.isnull().sum()
df_test.isnull().sum()

X= df_train.drop(['Item_Outlet_Sales'], axis=1)
y= df_train['Item_Outlet_Sales']

df_train.replace(to_replace='reg',value=3, inplace= True)

from sklearn.model_selection import train_test_split

X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.3, random_state=101)
X_eval, X_test, y_eval, y_test= train_test_split(X_other, y_other, test_size=0.5, random_state=101)
## We have divided our train csv into training, calibration and validation set 




##################### 1---> USING ELASTIC NET #####################



from sklearn.linear_model import ElasticNet

base_elasticnet_model= ElasticNet()

##param_grid= {"alpha":[0.1,1,5,10,50,100], "l1_ratio":[0.1,0.5,0.7,0.95,0.99,1]}
## Gives alpha=5  and l1_ratio= 1

param_grid= {"alpha":[4.5,5,5.5], "l1_ratio":[0.995, 0.999,1]}
# Gives alpha=4.5  and l1_ratio= 1

from sklearn.model_selection import GridSearchCV
grid_model= GridSearchCV(estimator= base_elasticnet_model, 
                         param_grid= param_grid,
                         scoring= "neg_mean_squared_error",
                         cv= 5, verbose=2)

grid_model.fit(X_train, y_train)


# Now we need best estimator
grid_model.best_estimator_ # Gives alpha=5  and l1_ratio= 1
grid_model.best_params_


## Calibration prediction
y_pred= grid_model.predict(X_eval)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
rmse= np.sqrt(mean_squared_error(y_eval, y_pred))
R_sq= r2_score(y_eval, y_pred)
rmse   ## 1048.57
R_sq   ##  0.59


## Validation prediction
y_val_pred= grid_model.predict(X_test)
len(y_val_pred)
len(X_test)
rmse= np.sqrt(mean_squared_error(y_test, y_val_pred))
R_sq= r2_score(y_test, y_val_pred)
rmse   ## 1104
R_sq   ##  0.55

##----> Our model is not overfitting but the RMSE is high and the r2_score is fair 



############## RANDOM FOREST REGRESSION ###########


from sklearn.ensemble import RandomForestRegressor
rfr= RandomForestRegressor(n_estimators=250)
# Calibration
run_model(rfr, X_train, y_train, X_eval, y_eval)
#  rmse:  1074
#  r2:  0.56

# Validation
run_model(rfr, X_train, y_train, X_test, y_test)
rmse:1100
r2:0.55



################# BOOSTING #################

from sklearn.ensemble import GradientBoostingRegressor

model= GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred= model.predict(X_other)

rmse= np.sqrt(mean_squared_error(y_other, y_pred))
R_sq= r2_score(y_other, y_pred)
rmse  ## 1030
R_sq  ## 0.605

## So far the best model in terms of RMSE and r2_score








