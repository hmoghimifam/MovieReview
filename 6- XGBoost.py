# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:28:18 2018

@author: Hossein
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

# Fitting XGBoost to the Training set
!pip install xgboost
!pip install msgpack
from xgboost import XGBClassifier
classifier = XGBClassifier(depth=7,n_estimators=90,n_jobs=4)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
acc=np.sum(y_pred==y_test)/len(y_test)

# Predicting the Evaluation set
y_prede = classifier.predict(X_eval)
y_prede= (y_prede+1)/2
