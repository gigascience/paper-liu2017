#!/usr/bin/python3
#-*- coding: UTF-8 -*-
import os,sys,re, operator
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from sklearn.metrics import roc_auc_score
import gbdtLr
from time import sleep
import xgboost as xgb
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import operator
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
#def print and plot
def lr_coef(path, result):
        o = open(path, 'w')        
        for (value, key) in result:
            if 0==value:
               pass
            else:
               writeln = "%s, %.3f\n" % (key, value)
               o.write(writeln)
        o.close()

def main():
    """
    Testing UCI Diabetes Data
    """
    #read data

    data = pd.read_csv("diabetes.csv", header=None, \
        names=["Number_of_times_pregnant", "Plasma_glucose_2h", "Diastolic_blood_pressure", "Triceps_skin_fold_thickness", "2Hour_serum_insulin","Body_mass_index", "Diabetes_pedigree_function", \
        "Age", "TARGET"], na_values="?")
    y = data["TARGET"].astype(np.int)
    drop_cols = ["TARGET"]
    X = data.drop(drop_cols, axis=1)
    X.fillna(-1, inplace=True)

    #label encoder
    for f in X.columns:
        if X[f].dtype=='object':
            print(f)
            lbl = preprocessing.LabelEncoder()
            X[f] = lbl.fit_transform(list(X[f].values))

    print("data shape: %d, %d" % (X.shape[0], X.shape[1]))
    origin_clf1 = xgb.XGBClassifier(n_estimators=110, max_depth=11, learning_rate=0.01, seed=42)
    #logistic
    l1 = LogisticRegression(C=1., penalty="l1", random_state=42)
    clf1 = gbdtLr.gbdtLrClassifier(gbdt_nrounds=12, gbdt_seed=42, gbdt_max_depth=4, gbdt_learning_rate=0.06, gbdt_subsample=1, gbdt_colsample_bytree=1, lr_penalty='l1', lr_c=1)     
    """
    data = X
    features = X.columns
    data['TARGET'] = pd.Series(y)
    """
    #train model
    
    #skf = StratifiedKFold(data['TARGET'].values, n_folds=4, shuffle=True, random_state=42)
    lst =[0.4, 0.5]
    for test_size in lst:
        i=1
        train_size = 1. - test_size
        X_train, X_test, y_train, y_test = train_test_split(X, y.values, random_state=42, test_size=train_size)
        
        #normalization for lr
        X_train_lr = pd.DataFrame(X_train, copy=True)
        X_test_lr = pd.DataFrame(X_test, copy=True)
        X_train_lr = (X_train_lr - X_train.mean()) / (X_train.max() - X_train.min())
        X_test_lr = (X_test_lr - X_train.mean()) / (X_train.max() - X_train.min())

        origin_clf1.fit(X_train, y_train)
        l1.fit(X_train_lr, y_train)
        clf1.fit(X_train, y_train)
        
        y_pred1 = origin_clf1.predict_proba(X_test)[:,1]
        y_pred2 = l1.predict_proba(X_test_lr)[:,1]
        y_pred3 = clf1.predict_proba(X_test)[:,1]

        rf_fpr, rf_tpr, _ = roc_curve(y_test, y_pred1)
        rf_score = roc_auc_score(y_test, y_pred1)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred2)
        lr_score = roc_auc_score(y_test, y_pred2)
        rf_lr_fpr, rf_lr_tpr, _ = roc_curve(y_test, y_pred3)
        rf_lr_score = roc_auc_score(y_test, y_pred3)
        print(roc_auc_score(y_test, y_pred2))
        #plot the pic
        lw=1
        plt.plot(rf_fpr, rf_tpr, color='darkorange',
             lw=lw, label='GBDT: %.3f' % rf_score)
        plt.plot(lr_fpr, lr_tpr, color='navy',
             lw=lw, label='LR: %.3f' % lr_score)
        plt.plot(rf_lr_fpr, rf_lr_tpr, color='deeppink',
             lw=lw, label='gbdt-space lasso: %.3f' % rf_lr_score)
        #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("hold out data: %d%%" % int(100*test_size))
        plt.legend(loc="best")
        plt.show()
        result = sorted(zip(map(lambda x:round(x,3), l1.coef_[0]), list(X_train.columns)), reverse=True)
        lr_coef("lr_coef.txt", result)

        #plt.savefig("hold out data: %d%%.png" % int(100*test_size))

if __name__=="__main__":
    main()
