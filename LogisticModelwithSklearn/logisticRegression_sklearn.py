#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:10:34 2018

@author: ali
"""
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt
import os
print(os.listdir("../dataSet"))
credit_card = pd.read_csv('../dataSet/creditcard.csv')
#%%
X = credit_card.drop(columns='Class', axis=1)
y = credit_card.Class.values
#%%
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y)
#%%
scaler = StandardScaler()
lr = LogisticRegression()
model1 = Pipeline([('standardize', scaler),
                    ('log_reg', lr)])
model1.fit(X_train, y_train)
y_test_hat = model1.predict(X_test)
y_test_hat_probs = model1.predict_proba(X_test)[:,1]
test_accuracy = accuracy_score(y_test, y_test_hat)*100
test_auc_roc = roc_auc_score(y_test, y_test_hat_probs)*100
print('Confusion matrix:\n', confusion_matrix(y_test, y_test_hat))
print('Training accuracy: %.4f %%' % test_accuracy)
print('Training AUC: %.4f %%' % test_auc_roc)
print(classification_report(y_test, y_test_hat, digits=6))
fpr, tpr, thresholds = roc_curve(y_test, y_test_hat_probs, drop_intermediate=True)
f, ax = plt.subplots(figsize=(9, 6))
_ = plt.plot(fpr, tpr, [0,1], [0, 1])
_ = plt.title('AUC ROC')
_ = plt.xlabel('False positive rate')
_ = plt.ylabel('True positive rate')
plt.style.use('seaborn')

plt.savefig('auc_roc.png', dpi=600)
y_hat_90 = (y_test_hat_probs > 0.90 )*1
print('Confusion matrix for 90%:\n', confusion_matrix(y_test, y_hat_90))
print('Report for 90%',classification_report(y_test, y_hat_90, digits=6))
y_hat_10 = (y_test_hat_probs > 0.05)*1
print('Confusion matrix for 5%:\n', confusion_matrix(y_test, y_hat_10))
print('Report for 5%',classification_report(y_test, y_hat_10, digits=4))