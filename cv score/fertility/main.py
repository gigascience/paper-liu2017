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
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import operator
def skf_auc(clf, data, y="TARGET", do_print=0):
    """
    Cross-Validation with Stratified K-Fold
    """
    skf_data = pd.DataFrame(data, copy=True)
    skf_data.reset_index(drop=True, inplace=True)
    features = list(skf_data.columns)
    features.remove(y)
    skf = StratifiedKFold(skf_data[y].values, n_folds=10, shuffle=True, random_state=42)
    y_true_list = []
    y_pred_list = []
    auc_list = []
    for i, (train_idx, test_idx) in enumerate(skf):
        x_train = skf_data.loc[train_idx, features]
        y_train = skf_data.loc[train_idx, y].values
        x_test = skf_data.loc[test_idx, features]
        y_test = skf_data.loc[test_idx, y].values
        clf.fit(x_train, y_train)
        y_pred = clf.predict_proba(x_test)[:,1]
        auc1 = roc_auc_score(y_test, y_pred)
        if 1==do_print:
            print(auc1)
        auc_list.append(auc1)
    return np.array(auc_list).mean()

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
    origin_clf1 = xgb.XGBClassifier(n_estimators=110, max_depth=4, seed=42, learning_rate=0.08, subsample=0.9, colsample_bytree=0.6)
    clf1 = gbdtLr.gbdtLrClassifier(gbdt_nrounds=12, gbdt_seed=42, gbdt_max_depth=4, gbdt_learning_rate=0.05, gbdt_subsample=1, gbdt_colsample_bytree=1, lr_penalty='l1', lr_c=1)     
    
    data = X
    data['TARGET'] = pd.Series(y)

    auc1 = skf_auc(origin_clf1, data)
    auc3 = skf_auc(clf1, data)
    print("xgboost origin auc is %.4f" % auc1)
    print("xgb transformed auc is %.4f" % auc3)
if __name__=="__main__":
    main()