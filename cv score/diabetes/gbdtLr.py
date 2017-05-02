#!/usr/bin/python3
#-*- coding: UTF-8 -*-
#author: Grimgor Ironhide
#this is a gbdt+lr classifier, regressor may come later
#so just keep calm
#and
#WAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGH!!!!!!!
import re
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
IMPT=3
class gbdtLrClassifier(BaseEstimator, TransformerMixin):
    """
    gbdt+lr,sklearn-clf style
    """
    def __init__(self, gbdt_nrounds=50, gbdt_learning_rate=0.01, gbdt_seed=1580, gbdt_max_depth=6, 
           gbdt_min_child_weight=11, gbdt_subsample=0.7, gbdt_colsample_bytree=0.7, 
           lr_penalty = 'l2', lr_c=1.0, lr_random_state=42, 
           gbdt_on=True, dump_path="gbdt_node.txt"):
        """
        #xgboost parameters
        self._gbdt_nrounds = gbdt_nrounds
        self._gbdt_learning_rate = gbdt_learning_rate
        self._gbdt_max_depth = gbdt_max_depth
        self._gbdt_min_child_weight = gbdt_min_child_weight
        self._gbdt_subsample = gbdt_subsample
        self._gbdt_colsample_bytree = gbdt_colsample_bytree
        self._gbdt_seed = gbdt_seed
        #lr parameters
        self._lr_penalty = lr_penalty
        self._lr_c = lr_c
        self._lr_random_state = lr_random_state
        """
        self.__xgb_model = xgb.XGBClassifier(n_estimators=gbdt_nrounds, max_depth=gbdt_max_depth, seed=gbdt_seed, learning_rate = gbdt_learning_rate, subsample=gbdt_subsample, colsample_bytree=gbdt_colsample_bytree)
                          #learning_rate = gbdt_learning_rate,\
                        #min_child_weight=gbdt_min_child_weight, subsample=gbdt_subsample, \
                         #colsample_bytree=gbdt_colsample_bytree
        self.__lr_model = LogisticRegression(C=lr_c, penalty=lr_penalty, random_state=lr_random_state)


        #path related,and tag to switch xgboost on/off
        self.__gbdt_on = gbdt_on
        self.__dump_path = dump_path

        #linear coef
        self.__coef = []

        #values dict for data transform
        self.__feature_dict = {}

        self.__SYS_MIN = -999999
        self.__SYS_MAX = 999999

    def transform(self, data):
        #feature encoder
        da = pd.DataFrame(data, copy=True)
        origin_cols = list(data.columns)
        for feature,value_set in self.__feature_dict.items():
            #create two inf of the value_list
            value_list = sorted(list(value_set))
            #min1 = value_list[0]
            #max1 = value_list[-1]
            min0 = self.__SYS_MIN
            max0 = self.__SYS_MAX
            value_list.insert(0,min0)
            value_list.insert(len(value_list),max0)
            for i, value in enumerate(value_list):
                 #no need for the last
                 if len(value_list)==i+1:
                     break
                 #rule: right area of the value
                 low_bound = value
                 high_bound = value_list[i+1]
                 low_bound_name = str(low_bound) if low_bound!=self.__SYS_MIN else "MIN"
                 high_bound_name = str(high_bound) if high_bound!=self.__SYS_MAX  else "MAX"
                 col = "%s<=%s<%s" % (low_bound_name, feature, high_bound_name)    #name the col
                 da[col] = da[feature].apply(lambda x: 1 if x>=low_bound and x<high_bound else 0)

        #remove original feature
        da = da.drop(origin_cols, axis=1) 
        return da

    def fit_transform(self, data, y=None):
        if self.__gbdt_on:
            self.__xgb_model.fit(data, y)
            self.__xgb_model.booster().dump_model(self.__dump_path)

            f = open(self.__dump_path, 'r')
            feature_dict = {}
            
            for line in f.readlines():
                if '<' in line:           #feature line
                       line = line.split(':')[1].strip()
                       feature_re = re.match('\[(.*)?\]', line)
                       info = feature_re.group(0)              #should be only one group
                       info = re.sub('\[|\]','',info)
                       feature = info.split('<')[0].strip()
                       value = float(info.split('<')[1].strip())
                       value_set = feature_dict[feature] if feature in feature_dict else set()
                       value_set.add(value)
                       feature_dict[feature] = value_set

            self.__feature_dict = feature_dict

        result_data = self.transform(data)
        return result_data

    def fit(self, data, y=None):
        global IMPT
        da = data     #pd.DataFrame(data, copy=True)
        for i in list(da.columns):
            self.__SYS_MIN = self.__SYS_MIN if self.__SYS_MIN<da[i].min() else da[i].min()-1
            self.__SYS_MAX = self.__SYS_MAX if self.__SYS_MAX>da[i].max() else da[i].max()+1
           
        da = self.fit_transform(da, y)
        self.__lr_model.fit(da, y)
        self.__coef = self.__lr_model.coef_[0]
        result = sorted(zip(map(lambda x:round(x,IMPT), self.__coef), list(da.columns)), reverse=True)
        self.save_coef("gbdt_coef_analysis.txt", result)    
        #plot_coef("coef_analysis.png", result)   
        return self             

    def predict(self, test):
        test1 = self.transform(test)
        return self.__lr_model.predict(test1)

    def predict_proba(self, test): 
        test1 = self.transform(test)
        return self.__lr_model.predict_proba(test1)

    #def print and plot
    def save_coef(self, path, result):
        o = open(path, 'w')        
        o.write("len of feature: %s\n" % len(result))
        for (value, key) in result:
            if 0==value:
               pass
            else:
               writeln = "%s: %.3f\n" % (key, value)
               o.write(writeln)
        o.close()


class gbdtLrRegressor(BaseEstimator, TransformerMixin):
    """
    gbdt+lr,sklearn-clf style
    """
    def __init__(self, gbdt_nrounds=50, gbdt_learning_rate=0.01, gbdt_seed=1580, gbdt_max_depth=6, 
           gbdt_min_child_weight=11, gbdt_subsample=0.7, gbdt_colsample_bytree=0.7, 
           lr_penalty = 'l1', lr_c=1.0, lr_random_state=42, 
           gbdt_on=True, dump_path="gbdt_node.txt"):
        """
        #xgboost parameters
        self._gbdt_nrounds = gbdt_nrounds
        self._gbdt_learning_rate = gbdt_learning_rate
        self._gbdt_max_depth = gbdt_max_depth
        self._gbdt_min_child_weight = gbdt_min_child_weight
        self._gbdt_subsample = gbdt_subsample
        self._gbdt_colsample_bytree = gbdt_colsample_bytree
        self._gbdt_seed = gbdt_seed
        #lr parameters
        self._lr_penalty = lr_penalty
        self._lr_c = lr_c
        self._lr_random_state = lr_random_state
        """
        self.__xgb_model = xgb.XGBRegressor(n_estimators=gbdt_nrounds, max_depth=gbdt_max_depth, seed=gbdt_seed)
                          #learning_rate = gbdt_learning_rate,\
                        #min_child_weight=gbdt_min_child_weight, subsample=gbdt_subsample, \
                         #colsample_bytree=gbdt_colsample_bytree
        self.__lr_model = Lasso(alpha=lr_c, random_state=lr_random_state)


        #path related,and tag to switch xgboost on/off
        self.__gbdt_on = gbdt_on
        self.__dump_path = dump_path

        #linear coef
        self.__coef = []

        #values dict for data transform
        self.__feature_dict = {}

        self.__SYS_MIN = -999999
        self.__SYS_MAX = 999999

    def transform(self, data):
        #feature encoder
        da = pd.DataFrame(data, copy=True)
        origin_cols = list(data.columns)
        for feature,value_set in self.__feature_dict.items():
            #create two inf of the value_list
            value_list = sorted(list(value_set))
            #min1 = value_list[0]
            #max1 = value_list[-1]
            min0 = self.__SYS_MIN
            max0 = self.__SYS_MAX
            value_list.insert(0,min0)
            value_list.insert(len(value_list),max0)
            for i, value in enumerate(value_list):
                 #no need for the last
                 if len(value_list)==i+1:
                     break
                 #rule: right area of the value
                 low_bound = value
                 high_bound = value_list[i+1]
                 low_bound_name = str(low_bound) if low_bound!=self.__SYS_MIN else "MIN"
                 high_bound_name = str(high_bound) if high_bound!=self.__SYS_MAX  else "MAX"
                 col = "%s<=%s<%s" % (low_bound_name, feature, high_bound_name)    #name the col
                 da[col] = da[feature].apply(lambda x: 1 if x>=low_bound and x<high_bound else 0)

        #remove original feature
        da = da.drop(origin_cols, axis=1) 
        return da

    def fit_transform(self, data, y=None):
        if self.__gbdt_on:
            self.__xgb_model.fit(data, y)
            self.__xgb_model.booster().dump_model(self.__dump_path)

            f = open(self.__dump_path, 'r')
            feature_dict = {}
            
            for line in f.readlines():
                if '<' in line:           #feature line
                       line = line.split(':')[1].strip()
                       feature_re = re.match('\[(.*)?\]', line)
                       info = feature_re.group(0)              #should be only one group
                       info = re.sub('\[|\]','',info)
                       feature = info.split('<')[0].strip()
                       value = float(info.split('<')[1].strip())
                       value_set = feature_dict[feature] if feature in feature_dict else set()
                       value_set.add(value)
                       feature_dict[feature] = value_set

            self.__feature_dict = feature_dict

        result_data = self.transform(data)
        return result_data

    def fit(self, data, y=None):
        global IMPT
        da = data     #pd.DataFrame(data, copy=True)
        for i in list(da.columns):
            self.__SYS_MIN = self.__SYS_MIN if self.__SYS_MIN<da[i].min() else da[i].min()-1
            self.__SYS_MAX = self.__SYS_MAX if self.__SYS_MAX>da[i].max() else da[i].max()+1
           
        da = self.fit_transform(da, y)
        self.__lr_model.fit(da, y)
        self.__coef = self.__lr_model.coef_
        result = sorted(zip(map(lambda x:round(x,IMPT), self.__coef), list(da.columns)), reverse=True)
        self.save_coef("gbdt_coef_analysis.txt", result)    
        #plot_coef("coef_analysis.png", result)   
        return self             

    def predict(self, test):
        test1 = self.transform(test)
        return self.__lr_model.predict(test1)

    #def print and plot
    def save_coef(self, path, result):
        o = open(path, 'w')        
        o.write("len of feature: %s\n" % len(result))
        for (value, key) in result:
            if 0==value:
               pass
            else:
               writeln = "%s: %.3f\n" % (key, value)
               o.write(writeln)
        o.close()