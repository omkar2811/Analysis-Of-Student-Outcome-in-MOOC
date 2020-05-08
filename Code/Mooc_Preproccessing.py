#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: omkar
"""

#import the required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import neighbors
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Read the csv file.
df = pd.read_csv("/home/omkar/Downloads/HMXPC13_DI_v2_5-14-14.csv")

print("---------------------------Data Preprocessing and Memory Management--------------------------")

# Print the first 5 contents of csv file.
print(df.head())

print("--------------------------")
#print info of the dataset and the memory usage.
print(df.info(memory_usage='deep'))

print("--------------------------")
#print memory usage datatype wise.
for dtype in ['float','int','object']:
    selected_dtype = df.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

print("--------------------------")
#memory usage function.
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

#memory usage of ints after conversion and before conversion.
df_int = df.select_dtypes(include=['int'])

#convert int into unsigned int
converted_int = df_int.apply(pd.to_numeric,downcast='unsigned')

#inintal memory usage
print(mem_usage(df_int))

#usage after conversion
print(mem_usage(converted_int))
print("--------------------------")

compare_ints = pd.concat([df_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['before','after']
compare_ints.apply(pd.Series.value_counts)
print(compare_ints)
print("--------------------------")

#memory usage after conversion and before conversion.
df_float = df.select_dtypes(include=['float'])

#convert float64 into float32
converted_float = df_float.apply(pd.to_numeric,downcast='float')

#initial memory usage
print(mem_usage(df_float))

#usage after conversion
print(mem_usage(converted_float))
print("--------------------------")

compare_floats = pd.concat([df_float.dtypes,converted_float.dtypes],axis=1)
compare_floats.columns = ['before','after']
compare_floats.apply(pd.Series.value_counts)
print(compare_floats)

print("--------------------------")

#comparing the output files after reduction in memory
optimized_df = df.copy()

optimized_df[converted_int.columns] = converted_int
optimized_df[converted_float.columns] = converted_float

#inital memory usage
print(mem_usage(df))

#memory usage after conversion
print(mem_usage(optimized_df))

print("--------------------------")

#memory usage of data type object
df_obj = optimized_df.select_dtypes(include=['object']).copy()
df_obj.describe()
print(df_obj.describe())

#memory usage by variable gender
gen = df_obj.gender
print(gen.head())

gen_cat = gen.astype('category')
print(gen_cat.head())

print("--------------------------")
#initial memory usage
print(mem_usage(gen))

#memory usage after conversion
print(mem_usage(gen_cat))

print("--------------------------")

print(gen_cat.head().cat.codes)

#conversion for all objects
converted_obj = pd.DataFrame()

for col in df_obj.columns:
    num_unique_values = len(df_obj[col].unique())
    num_total_values = len(df_obj[col])
    if num_unique_values / num_total_values < 0.5:
        converted_obj.loc[:,col] = df_obj[col].astype('category')
    else:
        converted_obj.loc[:,col] = df_obj[col]

#initial memory usage        
print(mem_usage(df_obj))

#memory usage after conversion
print(mem_usage(converted_obj))

print("--------------------------")

compare_obj = pd.concat([df_obj.dtypes,converted_obj.dtypes],axis=1)
compare_obj.columns = ['before','after']
compare_obj.apply(pd.Series.value_counts)

print(compare_obj)

print("--------------------------")

optimized_df[converted_obj.columns] = converted_obj

print(mem_usage(optimized_df))
print(optimized_df.grade.cat.codes)
print("--------------------------")



print("--------------------------Data Cleaning and Imputations of missing values----------------------------")
#mask = np.isnan(optimized_df.registered)
#optimized_df[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), optimized_df[~mask])
print(optimized_df.isnull().sum())

#Drop unnecassary values
optimized_df = optimized_df.drop(['roles','YoB'],axis = 1)
X = optimized_df.copy()

#Imputation of Categorical variable grade
X['grade'] = X['grade'].cat.add_categories([0])
X['grade'] = X['grade'].fillna(0)

#Imputation of Categorical Variable Level Of Education
X['LoE_DI'] = X['LoE_DI'].cat.add_categories("Unknown")
X['LoE_DI'] = X['LoE_DI'].fillna("Unknown")

#Identifying numerical columns to produce heatmaps
catcols = ['course_id ','userid_DI','final_cc_cname_DI','LoE_DI','gender','grade','start_time_DI','last_event_DI']
numcols = [x for x in X.columns if x not in catcols]

#plotting hearmap of numeric data
plt.figure(figsize = (12,8))
sns.heatmap(data=X[numcols].corr(),annot=True, cmap="coolwarm",fmt='.2f',linewidths=.05)
fig = plt.gcf()
plt.title("Corelation between the all features" )
fig.savefig("/home/omkar/Correleance_1.png")
plt.show()
plt.gcf().clear()

#Plot missing values and arrange in ascending order
missing_df = X.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,12))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
fig.savefig("/home/omkar/MissingCount.png")
plt.show()

#Imputing missing alues of flags in incomplete flags.
X['incomplete_flag'] = X['incomplete_flag'].fillna(0)


#Reconfirmig the missing values
missing_df = X.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,12))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
fig.savefig("/home/omkar/MissingCount.png")
plt.show()

#Knn classifier
def fillna_knn( df, base, target, fraction=1, threshold = 100, n_neighbors=5):
    assert isinstance( base , list ) or isinstance( base , np.ndarray ) and isinstance( target, str ) 
    whole = [ target ] + base
    print(threshold,"\n",fraction,"\n",n_neighbors)
    miss = df[target].isnull()
    notmiss = ~miss 
    nummiss = miss.sum()
    
    from sklearn.neural_network import MLPClassifier
    
    enc = OneHotEncoder()
    X_target = df.loc[ notmiss, whole ].sample( frac = fraction )
    enc.fit( X_target[ target ].unique().reshape( (-1,1) ) )
    
    Y = enc.transform( X_target[ target ].values.reshape((-1,1)) ).toarray()
    X = X_target[ base  ]
    
    print( 'fitting' )
    n_neighbors = n_neighbors
    clf = KNeighborsRegressor()
    clf.fit( X, Y )
    
    print( 'the shape of active features: ' ,enc.active_features_.shape )
    
    print( 'predicting' )
    Z = clf.predict(df.loc[miss, base])
    
    numunperdicted = Z[:,0].sum()
    if numunperdicted / nummiss *100 < threshold :
        print( 'writing result to df' )    
        df.loc[ miss, target ]  = np.dot( Z , enc.active_features_ )
        print( 'num of unperdictable data: ', numunperdicted )
        return enc
    else:
        print( 'out of threshold: {}% > {}%'.format( numunperdicted / nummiss *100 , threshold ) )
from sklearn.model_selection import train_test_split
#KNN regressor
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
    clf = KNeighborsRegressor( n_neighbors, weights = 'uniform' )
    clf.fit( X, Y ) 
    print( 'predicting' )
    
    print("Fit a model X_test and claculate Mean Squared Error with Y_test:")
    print(np.mean((Y_test-clf.predict(X_test)) ** 2))

    Z = clf.predict(df.loc[miss, base]) 
    print( 'writing result to df' )    
    df.loc[ miss, target ]  = Z

def fillna_linear_reg( df, base, target):
    assert isinstance( base , list ) or isinstance( base , np.ndarray ) and isinstance( target, str ) 
    whole = [ target ] + base
    miss = df[target].isnull()
    notmiss = ~miss 
    
    X_target = df.loc[ notmiss, whole ] 
    Y =  X_target[ target ]
    X = X_target[ base  ] 
    X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size = 0.2,random_state = 5)

    print( 'fitting' )
    
    clf = SGDRegressor(tol = 1e-3 )
    clf.fit( X, Y ) 
    print( 'predicting' )
    
    print("Fit a model X_test and claculate Mean Squared Error with Y_test:")
    print(np.mean((Y_test-clf.predict(X_test)) ** 2))

    Z = clf.predict(df.loc[miss, base]) 
    print( 'writing result to df' )    
    df.loc[ miss, target ]  = Z

def fillna_mlp_classifier( df, base, target):
    assert isinstance( base , list ) or isinstance( base , np.ndarray ) and isinstance( target, str ) 
    whole = [ target ] + base
    miss = df[target].isnull()
    notmiss = ~miss 
    nummiss = miss.sum()
    threshold = 10
    X_target = df.loc[ notmiss, whole ] 
    enc = OneHotEncoder()
    enc.fit( X_target[ target ].unique().reshape( (-1,1) ) )
    
    Y = enc.transform( X_target[ target ].values.reshape((-1,1)) ).toarray()
    X = X_target[ base  ]

    Y =  X_target[ target ]
    X = X_target[ base  ] 
    X_train,X_test,y_train,y_test =train_test_split(X,Y,test_size = 0.2,random_state = 5)

    #Testing all models
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
    from sklearn.neural_network import MLPClassifier
    
    names = ["MLP"]
             
    
    classifiers = [
        MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,), random_state=1)
        
    ]
    
    clf =  MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,), random_state=1)
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
        accs[name] = acc
        aucs[name] = auc
        f1s[name] = f1
        precisions[name] = precision
        recalls[name] = recall
        print(name, ' done')
            
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
        
    print( 'the shape of active features: ' ,enc.active_features_.shape )
    
    print( 'predicting' )
    Z = clf.predict(df.loc[miss, base])
    
    numunperdicted = Z[:,0].sum()
    if numunperdicted / nummiss *100 < threshold :
        print( 'writing result to df' )    
        df.loc[ miss, target ]  = np.dot( Z , enc.active_features_ )
        print( 'num of unperdictable data: ', numunperdicted )
    else:
        print( 'out of threshold: {}% > {}%'.format( numunperdicted / nummiss *100 , threshold ) )

        
        
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



X.hist(bins=4, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 1.2, 1.2)) 
zoningcode2int( df = X,target = 'gender' )
zoningcode2int( df = X,target = 'LoE_DI' )
zoningcode2int( df = X,target = 'grade' )
fillna_knn( df = X,
                  base = [  'certified' ,'explored','LoE_DI','viewed','grade'] ,
                  target = 'gender',fraction =0.2,threshold =100,n_neighbors = 5)


file = "/home/omkar/Cleaned_partial.csv"
X.to_csv(file)


