# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 03:54:33 2018

@author: Hossein
"""
import numpy as np
import matplotlib.pyplot as plt
#%% define function

runfile('functions.py')

#%% import data

runfile('importdata5.py') #this function uses the preprocessed data and performs tfidf on it
from sklearn.decomposition import PCA
pca = PCA(n_components = 2000)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
    
#%% ensemble of averaged perceptron

n_estimators=100
a_t=None
y_t=None
acc_t=[]
for i in range(n_estimators):
    s=np.arange(np.size(X_train,0))
    np.random.shuffle(s)
    w_all, w_best, numofmistakes, accs = averaged_perceptron(X_train[s[:5000]],y_train[s[:5000]],X_test,y_test,0.1,10)
    if a_t is None:
        a_t=np.reshape(w_best,[1,len(w_best)],0)
    else:
        a_t=np.append(a_t,np.reshape(w_best,[1,len(w_best)]),0)
    y_p= predict_perceptron(X_test,w_best)
    if y_t is None:
        y_t=np.reshape(y_p,[len(y_p),1])
    else:
        y_t=np.append(y_t,np.reshape(y_p,[len(y_p),1]),1)
    y_pred=np.sign(np.sum(y_t,1)+0.001)
    acc=(np.sum(y_pred==y_test)/len(y_pred))
    acc_t=np.append(acc_t,acc)
    
plt.plot(np.arange(1,n_estimators+1).astype(int),acc_t)
plt.title('Ensemble of Avg Perceptrons')
plt.ylabel('accuracy')
plt.xlabel('number of estimators')
plt.show()

y_te=None
for a in a_t:
    y_p= predict_perceptron(X_eval,a)
    if y_te is None:
        y_te=np.reshape(y_p,[len(y_p),1])
    else:
        y_te=np.append(y_te,np.reshape(y_p,[len(y_p),1]),1)
y_prede=np.sign(np.sum(y_te,1)+0.001)
y_prede=(y_prede+1)/2

np.max(acc_t)
np.where(acc_t==np.max(acc_t))[0]
    