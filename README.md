# Store_sales_predictions
The classical machine learning tasks like Data Exploration, Data Cleaning, Feature Engineering, Model Building and Model Testing. Different machine learning algorithms that’s best fit for the prediction of sales.
We had 17% na in Item_weight, 28% in Outlet_Size, we so we filled the na values by the mean of Item_Weights because the standard deviation of item weights was very less.
Filled the nan values of the Outlet_Size by the mode of Outlet_Size.
Hence all null values were been filled up and we did not drop any row.
Changed the Outlet_Establishment_Year into the the number of year established by substracting each year by 2022.

We have tried two approach:
1- Using conventional one hot encoding on whole categorical variables.
2- Dropping the Item_Identifier column because it had very high count of unique categorical values (1559). So doing one hot encoding would have increased time complexity to a large extent.
We changed the categorical values from strings into integer type for that we could use Label encoding or one hot encoding..
So, we did label encoding for the columns on which it makes sense on other columns, we did one hot encoding by splitting our dataframe.

1- First model was built using elastic net: 
>> first approach-     rmse: 1103.18
                    R_sq:  0.54

>> second approach-   rmse: 1104
                   R_sq: 0.55

2- Second model using SVR:
did not perform well

3- Third model- Random forest regression:
(A parameter grid was set and the best n_estimator= 250)
>> first approach-  rmse:1100
                 R_sq:0.55
                 
>> second approach- rmse:1086
                  R_sq:0.56
                  
4- Fourth model was boosting:
>> first approach- rmse: 1042
                R_sq: 0.595
                
>> second approach- rmse: 1030
                 R_sq: 0.605

Hence the second approach using gradient voosting gave the best result.
