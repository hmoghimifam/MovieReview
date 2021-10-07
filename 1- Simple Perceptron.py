# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:20:09 2018

@author: Hossein
"""

import numpy as np
import matplotlib.pyplot as plt
#%% define function

runfile('functions.py')

#%% import data

runfile('importdata5.py') #this function uses the preprocessed data and performs tfidf on it
from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

trainfolds={}
for i in range(5):
    s=np.arange(np.size(X_train,0))
    trainfolds[i]={}
    trainfolds[i]['vars']= X_train[s[i*5000:(i+1)*5000]]
    trainfolds[i]['label']= y_train[s[i*5000:(i+1)*5000]]


#%% CV
    
acc_report=[['learning rate','total # of updates','mean accuracy', 'standard deviation']]
learningrates=[1,0.1,0.01]
epochs=10
for learningrate in learningrates:
    accs=[]
    for i in range(len(trainfolds)):
        train_set_X, test_set_X, train_set_y, test_set_y= createTrain(trainfolds,i)
        w_all, w_best, totalnumberofmistakes, accsfe= simple_perceptron(train_set_X,train_set_y,test_set_X,test_set_y,learningrate,epochs)
        acc=np.mean(accsfe)
        accs=np.append(accs,acc)
    std= np.std(accs)
    acc_mean= np.mean(accs)
    acc_report=np.append(acc_report,[[learningrate,totalnumberofmistakes,acc_mean,std]],axis=0)
print('for Averaged Perceptron:\n',pd.DataFrame(acc_report[1:,:],columns=acc_report[0,:]),'\n\n')

#%% Calculate the accurary of the test set
        
acc_report=[['epoch','total # of updates','dev set accuracy']]
learningrate=0.01
epochs=20
w_all, w_best, numofmistakes, accs=simple_perceptron(X_train,y_train,X_test,y_test,learningrate,epochs)
epochnum=np.arange(1,epochs+1)
print('-------Averaged Perceptron-------')
print('The total number of updates on the training set:',numofmistakes,'\n')
print('The development set accuracy for the best classifier:',findacc(X_test,y_test,w_best),'\n')
#print('The test set accuracy for the best classifier:',findacc(X_test,y_test,w_best),'\n')
rep=np.reshape(np.array(['epoch #','dev set accuracy']),[1,2])
rep=np.append(rep,np.append(np.reshape(epochnum,[len(epochnum),1]),np.reshape(accs,[len(accs),1]),axis=1),axis=0)
plt.plot(epochnum,accs)
plt.title('learning curve (Averaged Perceptron)')
plt.ylabel('accuracy')
plt.xlabel('epoch id')
plt.show()


#%% Predict the Evaluation set

y_pred=np.ravel(np.sign(np.matmul(X_eval,np.transpose(w_best))))
y_pred=(y_pred+1)/2