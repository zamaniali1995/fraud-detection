#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:06:00 2018

@author: ali(zamanilai1995@gmail.com)
"""
#%%Imort packages
import numpy as np # linear algebra
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#%%Check datasret
import os
print(os.listdir("../dataSet"))
#%%Read the data
print('Loading the dataset.....')
credit_card = pd.read_csv('../dataSet/creditcard.csv')
print('Dataset shape: ',credit_card.shape)
print('Dataset was loaded!!!')
#%%Plot fraud vs nonfraud
f, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x='Class', data=credit_card)
_ = plt.title('# Fraud vs NonFraud')
_ = plt.xlabel('Class (1==Fraud)')   
plt.savefig("fraudvsnonfraud"+".pdf")
plt.show()
#%%Heatmap
corr=credit_card.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig("heatmap"+".pdf")
plt.show()
#%%Fraud and non-fraud data describe
non_fraud = credit_card[credit_card.Class == 0]
fraud = credit_card[credit_card.Class == 1]
non_fraud.Amount.describe()
fraud.Amount.describe()
#%%plot of high value transactions
bins = np.linspace(200, 2500, 100)
plt.hist(non_fraud.Amount, bins, alpha=1, normed=True, label='Non_fraud')
plt.hist(fraud.Amount, bins, alpha=0.6, normed=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Amount by percentage of transactions (transactions \$200+)")
plt.xlabel("Transaction amount (USD)")
plt.ylabel("Percentage of transactions (%)");
plt.savefig("Amountbypercentageoftransactions"+".pdf")
plt.show()
#%%Visual Exploration of the Data by Hour
bins = np.linspace(0, 48, 48) #48 hours
plt.hist((non_fraud.Time/(60*60)), bins, alpha=1, normed=True, label='Non_fraud')
plt.hist((fraud.Time/(60*60)), bins, alpha=0.6, normed=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Percentage of transactions by hour")
plt.xlabel("Transaction time as measured from first transaction in the dataset (hours)")
plt.ylabel("Percentage of transactions (%)");
plt.savefig("VisualExplorationoftheDatabyHour"+".pdf")
plt.show()
#%%Visual Exploration of Transaction Amount vs. Hour
plt.scatter((non_fraud.Time/(60*60)), non_fraud.Amount, alpha=0.6, label='non-fraud')
plt.scatter((fraud.Time/(60*60)), fraud.Amount, alpha=0.9, label='Fraud')
plt.title("Amount of transaction by hour")
plt.xlabel("Transaction time as measured from first transaction in the dataset (hours)")
plt.ylabel('Amount (USD)')
plt.legend(loc='upper right')
plt.savefig("VisualExplorationofTransactionAmountvsHour"+".pdf")
plt.show()