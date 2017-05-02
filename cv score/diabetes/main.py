#!/usr/bin/python3
#-*- coding: UTF-8 -*-
import os,sys,re, operator
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from sklearn.metrics import roc_auc_score
import gbdtLr
#import rfLr_naive
from time import sleep
import xgboost as xgb
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
def skf_auc(clf, data, y="TARGET"):
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
        print("Fold %d auc is: %.3f" % (i, auc1))
        auc_list.append(auc1)
    return np.array(auc_list).mean()

def main():
    """
    Testing UCI Diabetes Data
    """
    #read data

    data = pd.read_csv("diabetes.txt", header=None, \
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
    origin_clf1 = xgb.XGBClassifier(n_estimators=70, max_depth=2, seed=42, learning_rate=0.08, subsample=0.9, colsample_bytree=0.6)
    #logistic
    clf1 = gbdtLr.gbdtLrClassifier(gbdt_nrounds=8, gbdt_seed=42, gbdt_max_depth=2, gbdt_learning_rate=0.1, gbdt_subsample=0.7, gbdt_colsample_bytree=0.7, lr_penalty='l1', lr_c=1)     
    data = X
    data['TARGET'] = pd.Series(y)
     
    auc1 = skf_auc(origin_clf1, data)
    auc3 = skf_auc(clf1, data)
    print("xgboost origin auc is %.4f" % auc1)
    print("xgb transformed auc is %.4f" % auc3)
    #print("coef: %s" % l2.coef_[0])
if __name__ == "__main__":
    main()
