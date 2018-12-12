#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 00:35:48 2018

@author: ali
"""
#%%Parameters
train_set_num=.8
seed=5
#%%Imort packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#%% Define the normalized function
def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    col_mean = np.mean(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)
#%%Check datasret
import os
print(os.listdir("./dataset"))
#%%Read the data
print('Loading the dataset.....')
credit_card = pd.read_csv('./dataset/creditcard.csv')
print('Dataset shape: ',credit_card.shape)
print('Dataset was loaded!!!')
#%%Plot fraud vs nonfraud and heatmap
f, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x='Class', data=credit_card)
_ = plt.title('# Fraud vs NonFraud')
_ = plt.xlabel('Class (1==Fraud)')   

corr=credit_card.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#%%
# set replace=False, Avoid double sampling
X = credit_card.drop(columns='Class', axis=1).values.reshape(-1,30)
y = credit_card.Class.values.reshape(-1,1)
train_index = np.random.choice(len(X), round(len(X) * train_set_num),
                               replace=False)
test_index = np.array(list(set(range(len(X))) - set(train_index)))
train_X = X[train_index]
train_y = y[train_index]
test_X = X[test_index]
test_y = y[test_index]
#%% Normalized processing
train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)
#%%
