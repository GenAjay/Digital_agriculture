# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:15:01 2018
@author: agentimis1
"""
#%% Loading appropriate libraries ===============================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics, linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import Xgboost as XGBRegressor

#import tkinter as tk
#from tkinter import filedialog
#root = tk.Tk()
#root.withdraw()
#%% Loading data ===========================================================
#main_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick a directory for the project')
#main_dir=r'G:\My Drive\01_Collaborations\01_Current\01_LSU\Digital_Ag\Digital_Ag_Class\01_General\Notes\04_MachineLearning\Random_Forest\Python'
#main_dir='G:/My Drive/Collaborations/01_Current/Digital_Ag/Digital_Ag_Class/01_General/Notes/04_MachineLearning/Random_Forest/Python'
Ex1 = pd.read_csv('//Users//admin//Desktop//Dig_Ag_Assignment//Data//TG_ALLDATA_05_17_2021.csv')
#%% Removing entries with missing values ==============================
#Ex1.dropna(subset=['YLD'],inplace=True)
Ex1=Ex1.dropna()
#%%Remove outlier yields (top 5% and bottom 5%)
top_percentile = Ex1['YIELD'].quantile(0.95)
bottom_percentile = Ex1['YIELD'].quantile(0.05)
Ex1_cleaned = Ex1[(Ex1['YIELD'] >= bottom_percentile) & (Ex1['YIELD'] <= top_percentile)]

#%%Description of the dataset and the columns, re-arranging so that Yield is first
cols = list(Ex1_cleaned)
cols.insert(0, cols.pop(cols.index('YIELD')))
Ex1_cleaned = Ex1_cleaned.loc[:,cols]
df1 = Ex1_cleaned.describe()

#%%Converting character to numeric dummy variables
Ex1feat = pd.get_dummies(Ex1_cleaned)
print('The shape of the Dataset with Dummy Variables is :',Ex1feat.shape)

cols = list(Ex1)
cols.insert(0, cols.pop(cols.index('YIELD')))
Ex1=Ex1.loc[:,cols]
df1=Ex1.describe()
#%% Training and test separations ===============================================
X=Ex1feat.iloc[:,1:len(Ex1feat.columns)].values
y=Ex1feat.iloc[:,0].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
#%% Normalization of the training and test =====================================
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 
#%% Setting up the random forest and Fiting the dataset ===================
regressor = RandomForestRegressor(n_estimators=1200, random_state=0)  
regressor.fit(X_train, y_train)  
#%% Prediction and computation of metrics of errors for random errors =============================
y_pred_rf = regressor.predict(X_test)  
print('Mean Absolute Error for Random Forests:', metrics.mean_absolute_error(y_test, y_pred_rf))  
print('Mean Squared Error for Random Forests:', metrics.mean_squared_error(y_test, y_pred_rf))  
print('Root Mean Squared Error for Random Forests:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))  
#%% Create linear regression object =============================================
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
#%% Computing metrics for Linear regression ====================================
y_pred_lr=regr.predict(X_test)
print('Mean Absolute Error for Linear Regression:', metrics.mean_absolute_error(y_test, y_pred_lr))  
print('Mean Squared Error for Linear Regression:', metrics.mean_squared_error(y_test, y_pred_lr))  
print('Root Mean Squared Error for Linear Regression:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)))  
#%%Setting up XGBoost and fitting the dataset
xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
xgb_regressor.fit(X_train, y_train)

#%%Prediction and computation of metrics of errors for XGBoost
y_pred_xgb = xgb_regressor.predict(X_test)
print('Mean Absolute Error for XGBoost:', metrics.mean_absolute_error(y_test, y_pred_xgb))
print('Mean Squared Error for XGBoost:', metrics.mean_squared_error(y_test, y_pred_xgb))
print('Root Mean Squared Error for XGBoost:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb)))

#%% Performing Recursive feature elimination
selector = RFE(regr, n_features_to_select=5, step=1)
selector = selector.fit(X, y)
X_selected = selector.transform(X)
#Now we again perform the process of training and evaluation after performing recursive feature evaluation 
#%%Training and test separations with selected features
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=0)

#%%Setting up the random forest with selected features and fitting the dataset
regressor_sel = RandomForestRegressor(n_estimators=1200, random_state=0)  
regressor_sel.fit(X_train_sel, y_train_sel)

#%%Prediction and computation of metrics of errors for Random Forest with selected features
y_pred_rf_sel = regressor_sel.predict(X_test_sel)
print('Mean Absolute Error for Random Forests with Selected Features:', metrics.mean_absolute_error(y_test_sel, y_pred_rf_sel))  
print('Mean Squared Error for Random Forests with Selected Features:', metrics.mean_squared_error(y_test_sel, y_pred_rf_sel))  
print('Root Mean Squared Error for Random Forests with Selected Features:', np.sqrt(metrics.mean_squared_error(y_test_sel, y_pred_rf_sel)))

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
Result_errors3.to_csv('//Users//admin//Desktop//Dig_Ag_Assignment//Data//Errors.csv')
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
plt.savefig('//Users//admin//Desktop//Dig_Ag_Assignment//Side_by_Side.png')
#%%========== Describe Errors and save a csv file ========	
Res=Result_errors3.describe()
Res.to_csv('//Users//admin//Desktop//Dig_Ag_Assignment//Data//Error_description.csv')
