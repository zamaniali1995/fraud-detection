#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:10:34 2018

@author: ali
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#%%
import numpy as np 
import pandas as pd
import tensorflow as tf
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline
#from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing f5) will list the files in the input directory
import os
print(os.listdir("./dataset"))
#%%
#Start Function to load dataset
def loadDataSet(directory,trainNum):
    credit_card=pd.read_csv(directory)
    X = credit_card.drop(columns='Class', axis=1)
    y = credit_card.Class.values
    X_train, y_train = X[:trainNum], y[:trainNum]
    X_test, y_test = X[trainNum:], y[trainNum:]
    return credit_card,X_train, y_train,X_test,y_test,X,y
#End Function to load dataset
# Any results you write to the current directory are saved as output.
#Load data and visualize the data
trainNum=136598
credit_card,X_train, y_train,X_test,y_test,X,y=loadDataSet(
        './dataset/creditcard.csv',trainNum)
f, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x='Class', data=credit_card)
_ = plt.title('# Fraud vs NonFraud')
_ = plt.xlabel('Class (1==Fraud)')       
base_line_accuracy = 1-np.sum(credit_card.Class)/credit_card.shape[0]
print('% of non fraud=',base_line_accuracy)
#%%
corr=X.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#%%
np.random.seed(42)
