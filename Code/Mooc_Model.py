#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: omkar
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

outputformat = "%Y/%m/%d"
inputformat = "%Y-%m-%d"
X = pd.read_csv("/home/omkar/Cleaned_partial.csv")

def normalize(df):
        max_value = df.max()
        min_value = df.min()
        result= (df - min_value) / (max_value - min_value)
        return result

X['ndays_act'] = normalize(X['ndays_act'])
X['nevents'] = normalize(X['nevents'])
X['nplay_video'] = normalize(X['nplay_video'])
X['nchapters'] = normalize(X['nchapters'])

#Bayesian Ridge regression
def fillna_knn_reg( df, base, target, fraction=1, threshold = 10, n_neighbors=5):
    assert isinstance( base , list ) or isinstance( base , np.ndarray ) and isinstance( target, str ) 
    whole = [ target ] + base
    print(threshold,"\n",fraction,"\n",n_neighbors)
    miss = df[target].isnull()
    notmiss = ~miss 
    
    X_target = df.loc[ notmiss, whole ] 
    Y =  X_target[ target ]
    X = X_target[ base  ] 
    X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size = 0.2,random_state = 5)

    print( 'fitting' )
    
    n_neighbors = n_neighbors
    clf =BayesianRidge()
    clf.fit( X, Y ) 
    print( 'predicting' )
    
    print("Fit a model X_test and claculate Mean Squared Error with Y_test:")
    print(np.mean((Y_test-clf.predict(X_test))** 2))

    Z = clf.predict(df.loc[miss, base]) 
    print( 'writing result to df' )    
    df.loc[ miss, target ]  = Z

#function to deal with variables that are actually string/categories
def zoningcode2int( df, target ):
    storenull = df[ target ].isnull()
    enc = LabelEncoder( )
    df[ target ] = df[ target ].astype( str )

    print('fit and transform')
    df[ target ]= enc.fit_transform( df[ target ].values )
    print( 'num of categories: ', enc.classes_.shape  )
    df.loc[ storenull, target ] = np.nan
    print('recover the nan value')
    return enc

zoningcode2int( df = X,target = 'grade' )
zoningcode2int( df = X,target = 'LoE_DI' )
zoningcode2int( df = X,target = 'final_cc_cname_DI' )
zoningcode2int( df = X,target = 'course_id' )
fillna_knn_reg( df = X,
                 base = [  'course_id','certified','explored','grade','LoE_DI','viewed','final_cc_cname_DI'] ,
                  target = 'ndays_act', fraction = 0.2, n_neighbors = 1)

fillna_knn_reg( df = X,
                base = [  'certified','explored','grade','LoE_DI','viewed','final_cc_cname_DI'] ,
                target = 'nchapters',fraction = 0.2,n_neighbors = 1)

fillna_knn_reg( df = X,
                base = [   'certified','explored','grade','LoE_DI','viewed','final_cc_cname_DI','ndays_act'] ,
                target = 'nevents',fraction = 0.2,n_neighbors = 1)

fillna_knn_reg( df = X,
                base = [  'certified','nevents'] ,
                target = 'nplay_video',fraction = 0.2,n_neighbors = 1)

X = X.dropna()
from sklearn.utils import resample

df_majority = X[X.certified==0]
df_minority = X[X.certified==1]
print(df_majority)
print(df_minority)
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                replace=True,     # sample with replacement
                                 n_samples=444509,    # to match majority class
                                random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
print(df_upsampled.certified.value_counts())




print(X.isnull().sum())
def fillna_mlp_classifier( df, base, target):
    assert isinstance( base , list ) or isinstance( base , np.ndarray ) and isinstance( target, str ) 
    whole = [ target ] + base
    miss = df[target].isnull()
    notmiss = ~miss 
    
    X_target = df.loc[ notmiss, whole ] 
    enc = OneHotEncoder()
    enc.fit( X_target[ target ].unique().reshape( (-1,1) ) )
    
    Y = enc.transform( X_target[ target ].values.reshape((-1,1)) ).toarray()
    X = X_target[ base  ]

    Y =  X_target[ target ]
    X = X_target[ base  ] 


    #Testing all models
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn import model_selection
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.tree import _tree
    classifiers =[LogisticRegression(), DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(solver="svd", store_covariance=True),
    GradientBoostingClassifier()]
    names = ["LR","DT","RF","AB","GNB","LDA","GB"]
    
    from sklearn import metrics, preprocessing
    from sklearn.model_selection import train_test_split
    mean_auc = 0.0
    n = 1 
    for i in range(n):
    # for each iteration, randomly hold out 20% of the data as CV set
        X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=.20, random_state=i*48)
       # clf =  MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,), random_state=1)
    # iterate over classifiers
        accs = {}
        aucs = {}
        f1s = {}
        precisions = {}
        recalls = {}
    
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            y_pred_clf = clf.predict(X_test)
            auc = roc_auc_score(y_test, y_pred_clf)
            acc = accuracy_score(y_test, y_pred_clf)
            f1 = f1_score(y_test, y_pred_clf)
            precision = precision_score(y_test, y_pred_clf)
            recall = recall_score(y_test, y_pred_clf)
            cf = confusion_matrix(y_test, y_pred_clf)
            clf_rp = classification_report(y_test, y_pred_clf)
            preds = clf.predict_proba(X_test)[:, 1]

    # compute AUC metric for this CV fold
            fpr, tpr, thresholds = metrics.roc_curve(y_test, preds)
            roc_auc = metrics.auc(fpr, tpr)
            print("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))
            mean_auc += roc_auc

            print("Mean AUC: %f" % (mean_auc/n))

            print(name)
            print(cf)
            print(clf_rp)
            accs[name] = acc
            aucs[name] = auc
            f1s[name] = f1
            precisions[name] = precision
            recalls[name] = recall
            """if(name == "DT"):
    
                def tree_to_code(tree, feature_names):
                    tree_ = tree.tree_
                    feature_name = [
                            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                            for i in tree_.feature
                            ]
                    print ("def tree({}):".format(", ".join(feature_names)))
    
                    def recurse(node, depth):
                        indent = "  " * depth
                        if tree_.feature[node] != _tree.TREE_UNDEFINED:
                            name = feature_name[node]
                            threshold = tree_.threshold[node]
                            print("{}if {} <= {}:".format(indent, name, threshold))
                            recurse(tree_.children_left[node], depth + 1)
                            print("{}else:  # if {} > {}".format(indent, name, threshold))
                            recurse(tree_.children_right[node], depth + 1)
                        else:
                            print ("{}return {}".format(indent, tree_.value[node]))
    
                    recurse(0, 1)
                #tree_to_code(clf,X_train.columns.to_list())
            
            print(name, ' done')
               """ 
    print('\nAccuracies:')
    for name, acc in accs.items():
        print("%20s | Accuracy: %0.10f" % (name, acc))
     
    print('\nAUC Scores:')
    for name, auc in aucs.items():
        print("%20s | AUC: %0.10f" % (name, auc))
    
    print('\nF1 Scores:')
    for name, f1 in f1s.items():
        print("%20s | F1: %0.10f" % (name, f1))
    
    print('\nPrecision Scores:')
    for name, precision in precisions.items():
        print("%20s | Precision: %0.10f" % (name, precision))
    
    print('\nRecall Scores:')
    for name, recall in recalls.items():
        print("%20s | Recall: %0.10f" % (name, recall))
        
        
    
fillna_mlp_classifier( df = df_upsampled,
                  base = ['explored','LoE_DI','viewed','final_cc_cname_DI','grade','gender','nplay_video','ndays_act','nevents','nchapters','course_id'] ,
                  target = 'certified')


