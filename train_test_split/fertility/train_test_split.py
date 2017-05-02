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
def main():
    """
    Testing UCI Fertility Data
    """
    #read data
    list_of_names = [
           "Season",       # 1) winter, 2) spring, 3) Summer, 4) fall. (-1, -0.33, 0.33, 1) 

            "Age",         # 18-36 (0, 1) 

            "Childish diseases",          # 1) yes, 2) no. (0, 1) 

            "Accident",                   # 1) yes, 2) no. (0, 1) 

            "Surgical intervention",      # 1) yes, 2) no. (0, 1) 

            "High fevers",                # in the last year 1) less than three months ago, 2) more than three months ago, 3) no. (-1, 0, 1) 

            "Frequency of alcohol consumption",  # 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never (0, 1) 

            "Smoking habit",           # 1) never, 2) occasional 3) daily. (-1, 0, 1) 

            "Number of hours spent sitting",    # per day ene-16    (0, 1) 

            "target"                             # Diagnosis   normal (N), altered (O) 
    ]
    data = pd.read_csv("fertility.csv", header=None, names=list_of_names, na_values="?")
    y = data["target"].apply(lambda x: 1 if "N"==x else 0)
    drop_cols = ["target"]
    X = data.drop(drop_cols, axis=1)
    X.fillna(-1, inplace=True)
    """
    #label encoder
    for f in X.columns:
        if X[f].dtype=='object':
            print(f)
            lbl = preprocessing.LabelEncoder()
            X[f] = lbl.fit_transform(list(X[f].values))
    """     
    #get dummy for catgory features
    to_dummy = ["Season", 
            "Childish diseases",          # 1) yes, 2) no. (0, 1) 

            "Accident",                   # 1) yes, 2) no. (0, 1) 

            "Surgical intervention",      # 1) yes, 2) no. (0, 1) 

            "High fevers",                # in the last year 1) less than three months ago, 2) more than three months ago, 3) no. (-1, 0, 1) 

            "Frequency of alcohol consumption",  # 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never (0, 1) 

            "Smoking habit"]
    
    num_cols = list(set(list(X.columns)) - set(to_dummy))
    X = pd.get_dummies(X, columns=to_dummy)
    catgory_cols = list(set(list(X.columns)) - set(num_cols))
    print("catgory cols are: %s" % catgory_cols)
     
    print("data shape: %d, %d" % (X.shape[0], X.shape[1]))
    #origin_clf1 = xgb.XGBClassifier(n_estimators=200, max_depth=6, seed=42)
    origin_clf1 = xgb.XGBClassifier(n_estimators=110, max_depth=4, seed=42, learning_rate=0.08, subsample=0.9, colsample_bytree=0.6)

    l2 = LogisticRegression(C=.9, penalty="l1", random_state=42)
    clf1 = gbdtLr.gbdtLrClassifier(gbdt_nrounds=12, gbdt_seed=42, gbdt_max_depth=4, gbdt_learning_rate=0.1, gbdt_subsample=1, gbdt_colsample_bytree=1, lr_penalty='l1', lr_c=1)     
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
        
        origin_clf1.fit(X_train, y_train)
        l2.fit(X_train, y_train)
        clf1.fit(X_train, y_train)
        
        y_pred1 = origin_clf1.predict_proba(X_test)[:,1]
        y_pred2 = l2.predict_proba(X_test)[:,1]
        y_pred3 = clf1.predict_proba(X_test)[:,1]

        rf_fpr, rf_tpr, _ = roc_curve(y_test, y_pred1)
        rf_score = roc_auc_score(y_test, y_pred1)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred2)
        lr_score = roc_auc_score(y_test, y_pred2)
        rf_lr_fpr, rf_lr_tpr, _ = roc_curve(y_test, y_pred3)
        rf_lr_score = roc_auc_score(y_test, y_pred3)
        #plot the pic
        lw=1
        plt.plot(rf_fpr, rf_tpr, color='darkorange',
             lw=lw, label='GBDT: %.3f' % rf_score)
        plt.plot(lr_fpr, lr_tpr, color='navy',
             lw=lw, label='LR: %.3f' % lr_score)
        plt.plot(rf_lr_fpr, rf_lr_tpr, color='deeppink',
             lw=lw, label='gbdt-space lasso: %.3f' % rf_lr_score)
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("hold out data: %d%%" % int(100*test_size))
        plt.legend(loc="best")
        plt.show()
        plt.savefig("hold out data: %d%%.png" % int(100*test_size))

if __name__=="__main__":
    main()
