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
import seaborn as sns
import matplotlib.pyplot as plt

#import tkinter as tk
#from tkinter import filedialog
#root = tk.Tk()
#root.withdraw()
#%% Loading data ===========================================================
#main_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick a directory for the project')
#main_dir='C:\Users\admin\Downloads\Random_Forest\Python'
#main_dir='G:/My Drive/Collaborations/01_Current/Digital_Ag/Digital_Ag_Class/01_General/Notes/04_MachineLearning/Random_Forest/Python'
Ex1 = pd.read_csv('//Users//admin//Desktop//warner_data.csv')
####only use this method in Mac
#%% Removing entries with missing values ==============================
#Ex1.dropna(subset=['YLD'],inplace=True)
Ex1=Ex1.dropna()
#%% Description of the dataset and the columns, re-arranging so that Yield is first ========================
cols = list(Ex1)
cols.insert(0, cols.pop(cols.index('Treatments')))
Ex1=Ex1.loc[:,cols]
df1=Ex1.describe()
import pandas as pd

# Assuming Ex1 is your original DataFrame with variables 'Treatment' and 'Location' to be converted into dummy variables

# Re-arranging Columns
cols = list(Ex1)
cols.insert(0, cols.pop(cols.index('Treatments')))
Ex1 = Ex1.loc[:, cols]

# Convert 'Treatment' variable into dummy variables
treatment_dummies = pd.get_dummies(Ex1['Treatments'], prefix='Treatments')

# Convert 'Location' variable into dummy variables
location_dummies = pd.get_dummies(Ex1['Location'], prefix='Location')

# Concatenate the original DataFrame and the dummy variables DataFrames
Ex1_with_dummies = pd.concat([Ex1, treatment_dummies, location_dummies], axis=1)

# Drop the original categorical variables if needed
Ex1_with_dummies.drop(['Treatments', 'Location'], axis=1, inplace=True)

# Display the modified DataFrame with dummy variables
print(Ex1_with_dummies)

#%% Converting character to numeric dummy variables ===============================================
Ex1feat = (Ex1_with_dummies)
print('The shape of the Dataset with Dummy Variables is :',Ex1feat.shape)
#%% Training and test separations ===============================================
X=Ex1feat.iloc[:,1:len(Ex1feat.columns)].values
y=Ex1feat.iloc[:,0].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
#%% Normalization of the training and test =====================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 
#%% Setting up the random forest and Fiting the dataset ===================
regressor = RandomForestRegressor(n_estimators=1000, random_state=0) ###1000 n_tress 
regressor.fit(X_train, y_train)  
#%% Prediction and computation of metrics of errors =============================
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

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Error values for Random Forest and Linear Regression
random_forest_errors = [101.97253333333332, 13912.5902332, 117.95164362229126]
linear_regression_errors = [122.68707211762133, 23969.706229778392, 154.82153025266993]

# Error metric labels
error_labels = ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error']

# Create a DataFrame for error data
error_data = pd.DataFrame({
    'Error Metric': error_labels * 2,  # Repeat error metric labels for both models
    'Model': ['Random Forest'] * len(error_labels) + ['Linear Regression'] * len(error_labels),  # Model labels
    'Error Value': random_forest_errors + linear_regression_errors  # Concatenated error values
})

# Define colors for Random Forest and Linear Regression
colors = {'Random Forest': 'blue', 'Linear Regression': 'orange'}

# Plotting density curves with different colors
plt.figure(figsize=(10, 6))
sns.kdeplot(data=error_data[error_data['Model'] == 'Random Forest'], x='Error Value', fill=True,
            color=colors['Random Forest'], label='Random Forest')
sns.kdeplot(data=error_data[error_data['Model'] == 'Linear Regression'], x='Error Value', fill=True,
            color=colors['Linear Regression'], label='Linear Regression')
plt.xlabel('Error Value')
plt.ylabel('Density')
plt.title('Density Curve of Error Metrics')
plt.legend()
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Error values for Random Forest and Linear Regression
random_forest_errors = [101.97253333333332, 13912.5902332, 117.95164362229126]
linear_regression_errors = [122.68707211762133, 23969.706229778392, 154.82153025266993]

# Error metric labels
error_labels = ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error']

# Create a DataFrame for error data
error_data = pd.DataFrame({
    'Error Metric': error_labels * 2,  # Repeat error metric labels for both models
    'Model': ['Random Forest'] * len(error_labels) + ['Linear Regression'] * len(error_labels),  # Model labels
    'Error Value': random_forest_errors + linear_regression_errors  # Concatenated error values
})

# Define colors for Random Forest and Linear Regression
colors = {'Random Forest': 'blue', 'Linear Regression': 'orange'}

# Plotting two different boxplots with different colors
plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='Error Value', data=error_data, palette=colors)
plt.xlabel('Model')
plt.ylabel('Error Value')
plt.title('Error Metrics Comparison')
plt.show()

#