# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 01:04:32 2018

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


#%% preprocessing

vocab=makedata('vocab')
ps = PorterStemmer()
stemed_words=[]
for word in vocab:
    stemed_words.append(ps.stem(word.split()[0]))
stemed_words=np.array(stemed_words)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def stemX(X,stemed_words):
    for uword in np.unique(stemed_words):
        matchs=np.where(stemed_words==uword)[0]+1
        if len(matchs)>1:
            for match in matchs[1:]:
                X[:,matchs[0]]=X[:,matchs[0]]+X[:,match]
                X[:,match]=np.zeros_like(X[:,match])
    for word in stemed_words:
        if word in set(stopwords.words('english')) or word in set(['one', 'thi', 'hi', 'br', 'film', 'movi', 'wa']):
            matchs=np.where(stemed_words==word)[0]+1
            for match in matchs:
               X[:,match]=np.zeros_like(X[:,match]) 
    return X

# import data
X_train,y_train=makedataliblinear('data.train')
y_train=2*y_train-1
X_train=stemX(X_train,stemed_words)
ss=np.sum(X_train,0)
X_test,y_test=makedataliblinear('data.test')
y_test=2*y_test-1
X_test=np.append(X_test,np.zeros([np.size(X_test,0),np.size(X_train,1)-np.size(X_test,1)]),1)
X_test=stemX(X_test,stemed_words)
X_eval,y_eval=makedataliblinear('data.eval.anon')

#the new datas were saved in liblinear format and were used later on 

