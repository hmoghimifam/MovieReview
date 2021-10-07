# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:52:20 2018

@author: Hossein
"""

# SVM over Perceptron

import numpy as np
import matplotlib.pyplot as plt
#%% define function

runfile('functions.py')

#%% import data
    
runfile('importdata5.py')

#%% 1st Layer- Perceptron



from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_eval = pca.transform(X_eval)


n_estimators=200
a_t=None
acc_t=[]
for i in range(n_estimators):
    s=np.arange(np.size(X_train,0))
    np.random.shuffle(s)
    w_all, w_best, numofmistakes, accs = averaged_perceptron(X_train[s[:5000]],y_train[s[:5000]],X_test,y_test,0.1,10)
    if a_t is None:
        a_t=np.reshape(w_best,[1,len(w_best)])
    else:
        a_t=np.append(a_t,np.reshape(w_best,[1,len(w_best)]),0)

z_train=np.ones([np.size(X_train,0),1])
for a in a_t:
    y_p= predict_perceptron(X_train,a)
    z_train=np.append(z_train,np.reshape(y_p,[len(y_p),1]),1)
        
z_test=np.ones([np.size(X_test,0),1])
for a in a_t:
    y_p= predict_perceptron(X_test,a)
    z_test=np.append(z_test,np.reshape(y_p,[len(y_p),1]),1)

z_eval=np.ones([np.size(X_eval,0),1])
for a in a_t:
    y_p= predict_perceptron(X_eval,a)
    z_eval=np.append(z_eval,np.reshape(y_p,[len(y_p),1]),1)

# 2nd Layer- SVM
    
#%% CV
trainfolds={}
for i in range(5):
    s=np.arange(np.size(z_train,0))
    trainfolds[i]={}
    trainfolds[i]['vars']= z_train[s[i*5000:(i+1)*5000]]
    trainfolds[i]['label']= y_train[s[i*5000:(i+1)*5000]]

acc_report=[['learning rate','tradeoff','mean accuracy', 'standard deviation']]
learningrates=10.0**np.arange(-4,1)
tradeoffs=10.0**np.arange(-4,2)
for learningrate in learningrates:
    for tradeoff in tradeoffs:
        accs=[]
        prft=np.zeros([5,3])
        for i in range(len(trainfolds)):
            train_set_X, test_set_X, train_set_y, test_set_y= createTrain(trainfolds,i)
            w= svm_sgd(train_set_X,train_set_y,learningrate,tradeoff)
            y_pred=predictSVM(test_set_X,w)
            acc=(np.sum(y_pred==test_set_y)/len(y_pred))
            accs=np.append(accs,acc)
        std= np.std(accs)
        acc_mean= np.mean(accs)
        acc_report=np.append(acc_report,[[learningrate,tradeoff,acc_mean,std]],axis=0)
print('SVM:\n',pd.DataFrame(acc_report[1:,:],columns=acc_report[0,:]),'\n\n')


#%% predict dev set

w=svm_sgd(z_train,y_train,0.001,1)
y_pred= predictSVM(z_test,w)
acc=(np.sum(y_pred==y_test)/len(y_pred))

#%% predict eval set
y_pred= predictSVM(z_eval,w)
y_pred=(y_pred+1)/2



