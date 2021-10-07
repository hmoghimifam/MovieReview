# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:48:38 2018

@author: Hossein
"""

def logistic_regression(X,y,r0,S):
    t=0
#    w=-0.01+np.random.rand(1,np.size(X,1))*0.02
    w=np.zeros([1,np.size(X,1)])
    accs=[]
    while t<3 or np.sum(np.diff(accs[-4:])<0.0001)<3:
        r=r0/(1+t)
        s=np.arange(np.size(X,0))
        np.random.shuffle(s)
        for dum1 in range(np.size(X,0)):
            w=(1-2*r/S)*w+r*y[s[dum1]]*X[s[dum1]]*(np.exp(-y[s[dum1]]*(np.matmul(w,np.transpose(X[s[dum1]]))[0]))/(1+np.exp(-y[s[dum1]]*(np.matmul(w,np.transpose(X[s[dum1]]))[0])))) 
        y_pred=((1/(1+np.exp(-np.matmul(X,np.transpose(w)))))>=0.5)*1
        y_pred=y_pred+(y_pred-1)
        y_pred=np.ravel(y_pred)
        acc=np.sum(y_pred==y)/np.size(y_pred)
        accs=np.append(accs,acc)
        t+=1
    return w

def predictLR(X,classifier):
    y_pred=((1/(1+np.exp(-np.matmul(X,np.transpose(classifier)))))>=0.5)*1
    y_pred=y_pred+(y_pred-1)
    y_pred=np.ravel(y_pred)
    return y_pred
    

import numpy as np
import matplotlib.pyplot as plt
#%% define function

runfile('functions.py')

#%% import data
    
runfile('importdata5.py')


#%% cross-validation for Logistic Regression
from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_eval = pca.transform(X_eval)

trainfolds={}
for i in range(5):
    s=np.arange(np.size(X_train,0))
    trainfolds[i]={}
    trainfolds[i]['vars']= X_train[s[i*5000:(i+1)*5000]]
    trainfolds[i]['label']= y_train[s[i*5000:(i+1)*5000]]
    
    
    
acc_report=[['learning rate','tradeoff','mean accuracy', 'standard deviation']]
learningrates=10.0**np.arange(-4,1)
tradeoffs=10.0**np.arange(-1,5)
for learningrate in learningrates:
    for tradeoff in tradeoffs:
        accs=[]
        for i in range(len(trainfolds)):
            train_set_X, test_set_X, train_set_y, test_set_y= createTrain(trainfolds,i)
            w= logistic_regression(train_set_X,train_set_y,learningrate,tradeoff)
            y_pred=predictLR(test_set_X,w)
            acc=np.sum(y_pred==test_set_y)/len(test_set_y)
            accs=np.append(accs,acc)
        std= np.std(accs)
        acc_mean= np.mean(accs)
        acc_report=np.append(acc_report,[[learningrate,tradeoff,acc_mean,std]],axis=0)
print('Logistic Regression:\n',pd.DataFrame(acc_report[1:,:],columns=acc_report[0,:]),'\n\n')


#%% Test set
w= logistic_regression(X_train,y_train,0.01,10)
y_pred=predictLR(X_test,w)
acc=np.sum(y_pred==y_test)/len(y_test)
print('accuracy for Logistic Regression on the test set',acc)

#%% Evaluation set

y_pred=predictLR(X_eval,w)
y_pred=(y_pred+1)/2
