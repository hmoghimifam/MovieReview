# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 04:03:07 2018
------------------------- All Functions --------------------------------------
@author: Hossein
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 10)
import matplotlib.pyplot as plt

def simple_perceptron(X,y,Xdev,ydev,r,max_t):
    t=0
    numofmistakes=0
    w=-0.01+np.random.rand(1,np.size(X,1))*0.02
    accs=[]
    w_all= None
    while t<max_t:
        s=np.arange(np.size(X,0))
        np.random.shuffle(s)
        for dum1 in range(np.size(X,0)):
            if y[s][dum1]*np.matmul(w,np.transpose(X[s[dum1,:]]))[0]<=0:
                numofmistakes+=1
                w=w+r*y[s[dum1]]*X[s[dum1,:]]        
        acc=np.sum(np.ravel(np.sign(np.matmul(Xdev,np.transpose(w))))==ydev)/np.size(Xdev,0)
        accs=np.append(accs,acc)
        if w_all is None:
            w_all=w
        else:
            w_all=np.append(w_all,w,axis=0)
        t+=1
    accs_list=list(accs)
    bestidx=accs_list.index(max(accs_list))
    w_best= w_all[bestidx]
    return w_all, w_best, numofmistakes, accs

def averaged_perceptron(X,y,Xdev,ydev,r,max_t):
    t=0
    numofmistakes=0
    w=-0.01+np.random.rand(1,np.size(X,1))*0.02
    a=np.zeros_like(w)
    accs=[]
    w_all= None
    while t<max_t:
        s=np.arange(np.size(X,0))
        np.random.shuffle(s)
        for dum1 in range(np.size(X,0)):
            if y[s[dum1]]*np.matmul(w,np.transpose(X[s[dum1]]))[0]<=0:
                numofmistakes+=1
                w=w+r*y[s[dum1]]*X[s[dum1]]        
            a+=w
        acc=np.sum(np.ravel(np.sign(np.matmul(Xdev,np.transpose(a))))==ydev)/np.size(Xdev,0)
        accs=np.append(accs,acc)
        if w_all is None:
            w_all=a
        else:
            w_all=np.append(w_all,a,axis=0)
        t+=1
    accs_list=list(accs)
    bestidx=accs_list.index(max(accs_list))
    w_best= w_all[bestidx]
    return w_all, w_best, numofmistakes, accs

def predict_perceptron(X,a):
    return np.ravel(np.sign(np.matmul(X,np.transpose(a))))

def findacc(X,y,w):
    y_pred=np.ravel(np.sign(np.matmul(X,np.transpose(w))))
    acc=np.sum(y_pred==y)/len(y)
    return acc


#SVM-sgd
    
def svm_sgd(X,y,r0,C):
    t=0
#    w=-0.01+np.random.rand(1,np.size(X,1))*0.02
    w=np.zeros([1,np.size(X,1)])
    accs=[]
    while t<3 or np.sum(np.diff(accs[-4:])<0.0001)<3:
        r=r0/(1+t)
        s=np.arange(np.size(X,0))
        np.random.shuffle(s)
        for dum1 in range(np.size(X,0)):
            if y[s[dum1]]*np.matmul(w,np.transpose(X[s[dum1]]))[0]<=1:
                w=(1-r)*w+r*C*y[s[dum1]]*X[s[dum1]]
            else:
                w=(1-r)*w  
        y_pred=np.ravel(np.sign(np.matmul(X,np.transpose(w))))
        acc=np.sum(y_pred==y)/np.size(y_pred)
        
        accs=np.append(accs,acc)
        t+=1
    return w

def predictSVM(X,classifier):
    y_pred=np.ravel(np.sign(np.matmul(X,np.transpose(classifier))))
    return y_pred

#CrossValidation
    
def createTrain(trainfolds,testno):
    folds=np.delete(np.array(range(len(trainfolds))),testno)
    train_set_X=None
    train_set_y=None
    for fold in folds:
        if train_set_X is None:
            train_set_X=trainfolds[fold]['vars']
        else:
            train_set_X= np.append(train_set_X,trainfolds[fold]['vars'],axis=0)
        if train_set_y is None:
            train_set_y=trainfolds[fold]['label']
        else:
            train_set_y= np.append(train_set_y,trainfolds[fold]['label'],axis=0)
    test_set_X= trainfolds[testno]['vars']
    test_set_y= trainfolds[testno]['label']
    return train_set_X, test_set_X, train_set_y, test_set_y