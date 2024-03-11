# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:26:49 2019

@author: agentimis1
"""
#%%import necessary libraries/functions
import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics, linear_model
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

#%%==== Read/clean data
#main_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick a directory for the Project')
main_dir='D:/Google Drive/Collaborations/01_Current/Digital_Ag/Digital_Ag_Class/01_General/Notes/Machine_Learning/Random_Forest/Python'
Ex1 = pd.read_csv(main_dir+'/Data/TG_CleanData_12_05_2018.csv')
Ex1.dropna(subset=['YLD'],inplace=True)
#%%==== Move the YLD column first 
cols = list(Ex1)
cols.insert(0, cols.pop(cols.index('YLD')))
Ex1=Ex1.loc[:,cols]
print('This cell moved the target variable first')
#%%==== Finding columns that have missing values and fills them in with the average of that column
Nancols=Ex1.columns[Ex1.isna().any()]
Ex1[Nancols]=Ex1[Nancols].fillna(Ex1.mean().iloc[0])
#%%==== Create dummy variables for categorical variables
Ex1feat = pd.get_dummies(Ex1)
#%%==== Split The Input and output values 
X=Ex1feat.iloc[:,1:len(Ex1feat.columns)].values
y=Ex1feat.iloc[:,0].values.flatten()
#%%==== Create empty lists to store errors for 3 models(initialization)
rf_errors = []
regr_errors = []
recursive_errors = []

#%%==== Create 150 repetitions
for i in range(150):
	   # Simple train test split with 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    #Standardize input variables
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)  

				#random forest 
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)  
    regressor.fit(X_train, y_train)  
    y_pred_rf = regressor.predict(X_test)  
    #calculate MSE for rf
    a=metrics.mean_squared_error(y_test, y_pred_rf)
    rf_errors.append(a)
    
				#linear regression 
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred_lr=regr.predict(X_test)
    #calculate MSE for mlr
    a=metrics.mean_squared_error(y_test, y_pred_lr)
    regr_errors.append(a)
       
    #recursive regression 
    selector = RFE(regr, n_features_to_select=5, step=1)
    selector = selector.fit(X, y)
    y_pred_fs = selector.predict(X_test)
    #calculate MSE for recursive regression
    a=metrics.mean_squared_error(y_test, y_pred_fs)
    recursive_errors.append(a)

#%%==== Combine three error arrays, clean up and select 100 repetitions at random, save as csv
Result_errors1=pd.DataFrame({'Random_Forest':rf_errors,"Simple_Regression":regr_errors})
#Result_errors1=pd.DataFrame({'Random_Forest':rf_errors,"Simple_Regression":regr_errors,"Recursive_Regression":recursive_errors})
Result_errors2=Result_errors1[Result_errors1["Simple_Regression"]<600]
Result_errors3=Result_errors2.sample(n=100)
Result_errors3.to_csv(main_dir+'/Results/Errors.csv')
#%%==== Ploting Side by Side the RMSE
fig, ax = plt.subplots()
# build a box plot

ax.boxplot(Result_errors3)
# title and axis labels
ax.set_title('Side by Side Boxplot of RMSE for different Models')
ax.set_xlabel('Predictive Models')
ax.set_ylabel('Root Mean Square Errors')
xticklabels=['Random_Forest','Simple_Regression']
#xticklabels=['Random_Forest','Simple_Regression',"Recursive_Regression"]
ax.set_xticklabels(xticklabels)
# add horizontal grid lines
ax.yaxis.grid(True)
# show the plot
plt.savefig(main_dir+'/Results/Side_by_Side.png')
#%%========== Describe Errors and save a csv file ========	
Res=Result_errors3.describe()
Res.to_csv(main_dir+'/Results/Error_description.csv')