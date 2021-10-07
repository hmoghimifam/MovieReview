# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 03:57:58 2018

@author: Hossein
"""
import numpy as np
from nltk.stem.porter import PorterStemmer
import pandas as pd
pd.set_option('display.max_columns', 10)
import matplotlib.pyplot as plt

DATA_DIR='data-splits/'

def svmlib2nparray(data):
    maxvarnum=0
    for x in data:
        row = x.split()
        delim=row[-1].find(':')
        dummaxvarnum=int(row[-1][0:delim])
        if dummaxvarnum > maxvarnum:
            maxvarnum=dummaxvarnum
    output_X=np.zeros([len(data),maxvarnum+1])
    output_X[:,0]=1
    dum1=0
    output_y=[]
    for x in data:
        row = x.split()
        for col in row:
            delim=col.find(':')
            if delim!= -1:
                idx=int(col[0:delim])
                val=float(col[delim+1:])
                output_X[dum1][idx]=val
            else:
                output_y=np.append(output_y,int(col))
        dum1+=1
    return output_X,output_y

def makedata(name,DATA_DIR=DATA_DIR):
    with open(DATA_DIR + name, 'r') as f:
        rawdata = f.readlines()
    rawdata=np.asarray(rawdata)
    return rawdata

def makedataliblinear(name,DATA_DIR=DATA_DIR):
    with open(DATA_DIR + name, 'r') as f:
        content = f.readlines()
        output_X,output_y=svmlib2nparray(content)
    return output_X,output_y

def TF(X):
    X_TF=X/np.reshape(np.sum(X,1),[np.size(X,0),1])
    return X_TF

def IDF(X):
    idf=np.log(np.size(X,0)/np.sum((X_train!=0)*1,0))
    return np.reshape(idf,[1,np.size(X,1)])

#%% import data
X_train,y_train=makedataliblinear('train.n5')
idf=IDF(X_train)
X_train=idf*TF(X_train)

X_test,y_test=makedataliblinear('test.n5')
X_test=np.append(X_test,np.zeros([np.size(X_test,0),np.size(X_train,1)-np.size(X_test,1)]),1)
X_test=idf*TF(X_test)


X_eval,y_eval=makedataliblinear('eval.n5')
X_eval=np.append(X_eval,np.zeros([np.size(X_eval,0),np.size(X_train,1)-np.size(X_eval,1)]),1)
X_eval=idf*TF(X_eval)
