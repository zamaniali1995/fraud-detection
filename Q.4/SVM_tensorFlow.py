#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 05:47:51 2018

@author: ali
"""
#%%Parameters
train_set_num=.9
seed=5
# Define the learning rate， batch_size etc.
learning_rate = 0.003
batch_size = 150
epoch_num = 300
#%%Imort packages
import numpy as np # linear algebra
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
#%% Define the normalized function
def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    col_mean = np.mean(data, axis=0)
    return np.divide(data - col_mean, col_max - col_min)
#%%Check datasret
import os
print(os.listdir("../dataSet"))
#%%Read the data
print('Loading the dataset.....')
credit_card = pd.read_csv('../dataSet/creditcard.csv')
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
non_fraud = credit_card[credit_card.Class == 0]
non_fraud.Amount.describe()
#%%
fraud = credit_card[credit_card.Class == 1]
fraud.Amount.describe()
#%%
#plot of high value transactions
bins = np.linspace(200, 2500, 100)
plt.hist(non_fraud.Amount, bins, alpha=1, normed=True, label='Non_fraud')
plt.hist(fraud.Amount, bins, alpha=0.6, normed=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Amount by percentage of transactions (transactions \$200+)")
plt.xlabel("Transaction amount (USD)")
plt.ylabel("Percentage of transactions (%)");
plt.show()
#%%
bins = np.linspace(0, 48, 48) #48 hours
plt.hist((non_fraud.Time/(60*60)), bins, alpha=1, normed=True, label='Non_fraud')
plt.hist((fraud.Time/(60*60)), bins, alpha=0.6, normed=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Percentage of transactions by hour")
plt.xlabel("Transaction time as measured from first transaction in the dataset (hours)")
plt.ylabel("Percentage of transactions (%)");
#plt.hist((df.Time/(60*60)),bins)
plt.show()
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
#train_y = tf.cast(train_y, dtype = tf.float64)
#%% Normalized processing
train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)
#%%Build the model framework
# Begin building the model framework
# Declare the variables that need to be learned and initialization
# There are 30 features here, A's dimension is (30, 1)
w = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[30, 1]), name="w")
b = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[1, 1]), name="b")
#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)
# Define placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 30], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y")
# Define logistic Regression
#logit = tf.matmul(x, w) + b
#y_predicted = 1.0 / (1.0 + tf.exp(-logit))
# Declare loss function
model_output = tf.subtract(tf.matmul(x, w), b)
predicr_test = tf.sign(model_output)
y_predicted = tf.cast(tf.equal(tf.sign(model_output), 1), dtype=tf.float32)

l2_norm = tf.reduce_sum(tf.square(w))
alpha = tf.constant([0.5])
classification_term = tf.reduce_mean(tf.maximum(0.,
                                                tf.subtract(1.,tf.multiply(model_output,
                                                                           y))))

loss = tf.add(classification_term, tf.multiply(alpha,l2_norm))

# Define optimizer: GradientDescent         
optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)
# Define the accuracy
# The default threshold is 0.5, rounded off directly
#prediction = tf.sign(model_output)
# Bool into float32 type
#correct = tf.cast(tf.equal(y_predicted, y), dtype=tf.float32)
# Average
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_predicted, y), dtype=tf.float32))
# End of the definition of the model framework
#label=[tf.count_nonzero(y) ,tf.subtract(tf.size(y),tf.count_nonzero(y))]
#confusion_matrix_tf = tf.confusion_matrix(labels=[10 ,100],
#                                          predictions=[2 ,108])
#FN=tf.metrics.false_negatives(labels=y, predictions=tf.round(y_predicted))
#confiution=np.zeros(shape=[2,2])
#%%
print("Parameters were initialized, Session is runing ...")
train_error_list = []
train_acc_list = []
test_acc_list = []
test_error_list=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_num): 
        train_loss = 0
        for idx in range(len(train_X)//batch_size):
            input_list = {x: train_X[idx*batch_size:(idx+1)*batch_size],
                          y: train_y[idx*batch_size:(idx+1)*batch_size]}
            _, train_loss1 = sess.run([optimizer, loss], feed_dict=input_list)
            train_loss += train_loss1
        train_error_list.append(train_loss/len(train_X))
        train_acc_list.append(sess.run(
                accuracy, feed_dict={x: train_X, y: 
                    train_y})*100)
        test_acc_list.append(sess.run(accuracy
                                      , feed_dict={x: test_X,
                                                   y:  test_y})*100)
        test_error_list.append(sess.run(loss,
                                       feed_dict={x: test_X,
                                                  y:test_y})/len(test_y))
        if (epoch + 1) % 50 == 0:
            print('epoch: {:4d}'  .format(epoch + 1))
            print('test accuracy:', test_acc_list[epoch])
            print('train accuracy:', train_acc_list[epoch])
            print('test error:', test_error_list[epoch])
            print('train accuracy:', train_error_list[epoch])
                 
    train_pridected = sess.run(y_predicted, feed_dict = {x: train_X})
    w_value, b_value = sess.run([w, b])
    test_pridected = sess.run(y_predicted, feed_dict = {x:test_X})
##%%1
#train_y_hat=np.cast(np.sign(tf.subtract(tf.matmul(train_X, w_value), b_value)), 1))
#test_y_hat=np.cast(np.sign(tf.subtract(tf.matmul(test_X, w_value), b_value)), 1)
#predict_test = tf
test_accuracy = accuracy_score(test_y,test_pridected)*100
test_auc_roc = roc_auc_score(test_y, test_pridected)*100
#%%
print('Confusion matrix for train data:\n', confusion_matrix(test_y, 
                                                             test_pridected))
print('Confusion matrix for test data:\n', confusion_matrix(train_y, 
                                                             train_pridected))
print('Training accuracy: ' ,test_accuracy)
print('Training AUC: ' , test_auc_roc)
print(classification_report(test_y, test_pridected, digits=6))
fpr, tpr, thresholds = roc_curve(test_y, test_pridected, drop_intermediate=True)
                                 
#select_tereshold=np.zeros_like(thresholds)
#recall=tpr/(tpr+fpr)
#precision=
#select_tereshold=2*(tpr*fpr)/(tpr+fpr)
#select_tereshold.append
f, ax = plt.subplots(figsize=(9, 6))
_ = plt.plot(fpr, tpr, [0,1], [0, 1])
_ = plt.title('AUC ROC')
_ = plt.xlabel('False positive rate')
_ = plt.ylabel('True positive rate')
plt.style.use('seaborn')
plt.savefig('auc_roc.png', dpi=600)                                               
#%%
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("test accuracy = " + str(test_acc_list[epoch]))
for a in ax.reshape(-1,1):
    a[0].set_xlabel("epochs")
ax[0][0].plot(train_error_list[:100], color='red', label='train loss')
ax[0][0].plot(test_error_list[:100], color='blue', label='test loss')
ax[0][0].legend()
ax[1][0].plot(train_error_list, color='red', label='train loss')
ax[1][0].plot(test_error_list, color='blue', label='test loss')
ax[1][0].legend()
ax[0][1].plot(train_acc_list[:100], color='red', label='train accuracy')
ax[0][1].plot(test_acc_list[:100], color='blue', label='test accuracy')
ax[0][1].legend()
ax[1][1].plot(train_acc_list, color='red', label='train accuracy') 
ax[1][1].plot(test_acc_list, color='blue', label='test accuracy')
ax[1][1].legend()
plt.savefig("Section1"+".pdf")
#End main program