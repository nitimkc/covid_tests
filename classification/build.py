################################################################################
# Author : (c) Niti Mishra
# Date   : 2022.05.26
# Part 6 model training pipeline with gridsearch
################################################################################
import imp
import warnings
warnings.filterwarnings('ignore') 

from pathlib import Path
import pandas as pd
import numpy as np 

import time
import pickle

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from sklearn.inspection import permutation_importance

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix 
from sklearn.inspection import _permutation_importance

binary_models = []
binary_models.append( LogisticRegression(max_iter=1000, random_state=123) )
binary_models.append( tree.DecisionTreeClassifier(random_state = 0) )
binary_models.append( RandomForestClassifier(random_state = 0) )
# binary_models.append( SVC(random_state = 0, probability=True) )
binary_models.append( GradientBoostingClassifier(random_state = 0) )
binary_models.append( MLPClassifier(random_state=0, hidden_layer_sizes=(6,3,1), activation='relu', solver='adam') )

parameters = [
    {'clf__C': ( np.logspace(-5, 1, 5) ),
        'clf__penalty': ['none', 'l2', 'l1', 'elasticnet'], # regularization parameter
        # 'clf__solver': ['newton-cg', 'lbfgs', 'sag'], # solver
        },#logistic
    {'clf__criterion': ['entropy', 'gini'],
        'clf__min_samples_split': np.arange(0.01, 0.1,.02), #109 Minimum Size Split
        # 'clf__max_leaf_nodes': list(range(2, 100,4)),
        # 'clf__max_depth':range(3,20,7), #20
        },#decisiontree
    {'clf__n_estimators':range(50,100,10), #67 Number of Trees in the Forest:
        'clf__min_samples_split': np.arange(0.01, 0.1,.02), #109 Minimum Size Split
        'clf__max_features': ["sqrt", "log2"], #9
        # 'clf__min_samples_leaf': [1, 2, 4],
        },#randomforest
    # {'clf__C': [0.001, 0.01, 0.1, 1, 10],
    #     'clf__gamma': [0.001, 0.01, 0.1, 1]
    #     },#svm
    {'clf__n_estimators':range(50,100,10), #86
        'clf__max_depth':range(10,30,5), #20
        # 'clf__min_samples_split':range(12,25,4),
        # 'clf__max_features':range(7,10,2),
        # 'clf__learning_rate':[0.01,.1], #0.1,
        # 'clf__subsample':[0.6,0.7,0.8], #0.6
        },#gradientboosting
    {'clf__hidden_layer_sizes':[(2,2,2),(3,3,3),(4,4,4), (5,5,5)],
        # 'clf__batch_size':[10, 50, 75, 100, 150], 
        'clf__activation':["relu", "Tanh"],
        'clf__learning_rate':["adaptive"],
        'clf__learning_rate_init':[0.01],
        }#NN
]

# X = val[0]
# y = val[1]
# split_idx =val[2]
# k=5
# filter_idx = filtered_test_idx
# outpath = RESULTS
# std_cols=num_features
# models=binary_models[0]

################################################################################  
# function to calculate all metric and return them in a dictionary
################################################################################  

def allmetrics(best_model,X_test,y_test,name,start,ntrain,nvalid):

    best_estimator = best_model.best_estimator_['clf']
    best_param = best_model.best_params_

    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, _ = roc_curve(y_test,  y_pred_prob)

    importances = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42).importances_mean

    scores = {
        'time': time.time() - start,
        'name': name,
        'model': str(best_estimator),
        'size': [ntrain, nvalid, len(X_test)],
        'best_param': best_param,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'sensitivity': tp/(tp+fn),
        'specificity': tn/(tn+fp),
        'fpr': fpr,
        'tpr': tpr,
        'importances':importances,
        'AUC': roc_auc_score(y_test, y_pred_prob),
        'f1_test': f1_score(y_test, y_pred),
    }
    print(f"     scores saved to dictionary")
    print(f"     AUC: {scores['AUC']}")
    return scores

################################################################################  
# function to split data into train, test and validation
# perform gridsearch for best parameter
# calculate scores of best model on test set(s)
################################################################################  

def score_models(models, X, y, split_idx, std_cols, k=5, filter_idx=None, outpath=None):

    # split data for training valid and test
    print(f"     splitting data into train, validation and test sets")
    if split_idx is None:
        train_ratio = 0.50
        validation_ratio = 0.25
        test_ratio = 0.25
        X_train, X_test, y_train, y_test = tts(X, y, test_size=1 - train_ratio, random_state=42)
        X_valid, X_test, y_valid, y_test = tts(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42)
    else:
        train_idx = split_idx == "Training"
        valid_idx = split_idx == "Validation"
        test_idx = split_idx == "Test"
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        X_test, y_test = X[test_idx], y[test_idx] 
    n_Xtrain = len(X_train)
    n_Xvalid = len(X_valid)
    n_Xtest = len(X_test)
    print(f"     n_train: {n_Xtrain}, n_test: {n_Xtest}, n_valid: {n_Xvalid}")

    # encode the targets
    labels = LabelEncoder()
    y_train = labels.fit_transform(y_train)
    y_valid = labels.fit_transform(y_valid)
    y_test = labels.fit_transform(y_test)
    print('     encoded labels for train valid and test sets')

    # scale the numeric features of the training data
    scaler = StandardScaler()
    X_train[std_cols] = scaler.fit_transform(X_train[std_cols])
    X_valid[std_cols] = scaler.transform(X_valid[std_cols])
    X_test[std_cols] = scaler.transform(X_test[std_cols])
    print('     scaled features for train valid and test sets')

    # save scaler if path provided
    if outpath:
        with open(Path.joinpath(outpath, "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)

    # create pipeline for each model to be trained
    names = [str(i).split('(')[0] for i in models]
    for model, name, params in zip(models, names, parameters):
        print(f"     Training:{model}\n")#\n     {params}")
        
        pipe = Pipeline([
            ('clf', model),
            ])

        # create gridsearch with specified valid set
        if split_idx is not None: 
            # Create a list where train data indices are -1 and validation data indices are 0
            test_fold = [-1 for i in range(n_Xtrain)] + [0 for i in range(n_Xvalid)]
            ps = PredefinedSplit(test_fold=test_fold)
            grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=ps, n_jobs=-1, verbose=0)
        else:
            grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=k, n_jobs=-1, verbose=0)
        
        # train
        start = time.time()
        best_model = grid_search.fit( pd.concat((X_train,X_valid)), np.concatenate((y_train,y_valid)) )
        print(f'     grid search for best model in {name} completed')

        best_param = best_model.best_params_
        best_score = best_model.best_score_
        best_estimator = best_model.best_estimator_['clf']
        print('     best parameters:', best_param)
        print('     best score:', best_score)
        print('     best estimator:',best_estimator)   

        # save best model based on gridsearch if path provided
        if outpath:
            with open(Path.joinpath(outpath, name + ".pkl"), 'wb') as f:
                pickle.dump(best_estimator, f)

        # if filter index is provided obtain scores of best model for each filter
        if filter_idx is not None:
            print(f'     No. of filter_idx provided to separate test set: {len(filter_idx)}')
            print(f'     No. of index for each filter: {len(filter_idx[0])}, {len(filter_idx[1])}')
            print(f'     No. of instances in original test data: {len(X_test)}, {len(y_test)}')
            
            y_test_full = pd.Series(y_test, index=X_test.index)            # add index to the target values of test data 
            X_test_divided = [X_test.loc[i] for i in filter_idx]           # create test feature matrix for each filter
            y_test_divided = [y_test_full.loc[i] for i in filter_idx]      # create test target vector for each filter
            y_test_divided = [i.values for i in y_test_divided]            # convert target vectors back to numpy array

            # get prediction and probability in each set of test data
            scores_divided = []
            for filtered_Xtest, filtered_ytest  in zip(X_test_divided, y_test_divided):
                print(f'     No. of instances in filtered test set: {len(filtered_Xtest)}, {len(filtered_ytest)}')
                filtereddata_scores = allmetrics(best_model, filtered_Xtest, filtered_ytest, 'trainedall__'+name, start, n_Xtrain, n_Xvalid)
                scores_divided.append(filtereddata_scores)
            yield scores_divided

        else:
            # get prediction and probability on test data
            scores = allmetrics(best_model, X_test, y_test, name, start, n_Xtrain, n_Xvalid)
            yield scores
 