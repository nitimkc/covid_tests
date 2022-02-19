from logging import raiseExceptions
from pathlib import Path
import pandas as pd
import numpy as np 
import os
import csv
import time
import pickle
from collections import Counter
# import unicodedata

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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_auc_score, confusion_matrix 

binary_models = []
binary_models.append( LogisticRegression(max_iter=1000, random_state=123) )
binary_models.append( RandomForestClassifier(random_state = 0) )
# binary_models.append( SVC(random_state = 0, probability=True) )
binary_models.append( tree.DecisionTreeClassifier(random_state = 0) )
binary_models.append( GradientBoostingClassifier(random_state = 0) )
binary_models.append( MLPClassifier(random_state=0, hidden_layer_sizes=(6,3,1), activation='relu', solver='adam') )

parameters = [
    {
        'clf__C': ( np.logspace(-5, 1, 5) ),
        'clf__penalty': ['none', 'l2', 'l1', 'elasticnet'], # regularization parameter
        # 'clf__solver': ['newton-cg', 'lbfgs', 'sag'], # solver
        },#logistic
    {'clf__n_estimators':range(50,100,10), #67 Number of Trees in the Forest:
        'clf__min_samples_split': np.arange(0.01, 0.1,.02), #109 Minimum Size Split
        'clf__max_features': ["sqrt", "log2"], #9
        # 'clf__min_samples_leaf': [1, 2, 4],
        },#randomforest
    # {'clf__C': [0.001, 0.01, 0.1, 1, 10],
    #     'clf__gamma': [0.001, 0.01, 0.1, 1]
    #     },#svm
    {'clf__criterion': ['entropy', 'gini'],
        'clf__min_samples_split': np.arange(0.01, 0.1,.02), #109 Minimum Size Split
        # 'clf__max_leaf_nodes': list(range(2, 100,4)),
        # 'clf__max_depth':range(3,20,7), #20
        },#decisiontree
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

# split_idx = validation_idx
# outpath = RESULTS
# model = binary_models[0]
# name = str(model).split('(')[0]
# params = parameters[0]
# k = 5
# test_set = test_nofilter
def score_models(models, X, y, split_idx, std_cols, k=5, test_set=None, outpath=None):

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

    # if test set without any filter is provided use that as test_set
    if test_set is not None:
        X_test, y_test = test_set[0], test_set[1]
        print(f"Test set without any filter used")

    X_test.to_csv(Path.joinpath(outpath,'X_test.csv'), index=False)
    print("X_test written out to {}".format(outpath))
    
    n_Xtrn = len(X_train)
    n_Xval = len(X_valid)
    n_Xtst = len(X_test)

    print('n_train:', n_Xtrn, '\nn_test:' , n_Xtst, '\nn_valid:', n_Xval)

    # Label encode the targets
    labels = LabelEncoder()
    y_train = labels.fit_transform(y_train)
    y_valid = labels.fit_transform(y_valid)
    y_test = labels.fit_transform(y_test)
    print('completed training and testing data set-up')

    # scale the numeric features of the training data
    scaler = StandardScaler()
    X_train[std_cols] = scaler.fit_transform(X_train[std_cols])
    X_valid[std_cols] = scaler.transform(X_valid[std_cols])
    X_test[std_cols] = scaler.transform(X_test[std_cols])

    names = [str(i).split('(')[0] for i in models]
    for model, name, params in zip(models, names, parameters):
        print(model, '\n', name, '\n', params)
        
        pipe = Pipeline([
            ('clf', model),
            ])

        # Create gridsearch with specified valid set
        if split_idx is not None: 
            # Create a list where train data indices are -1 and validation data indices are 0
            test_fold = [-1 for i in range(n_Xtrn)] + [0 for i in range(n_Xval)]
            ps = PredefinedSplit(test_fold=test_fold)
            grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=ps, verbose=2, n_jobs=-1)#, scoring='roc_auc')
        else:
            grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=k, verbose=2, n_jobs=-1)#, scoring='roc_auc')
        
        start = time.time()
        best_model = grid_search.fit( pd.concat((X_train,X_valid)), np.concatenate((y_train,y_valid)) )
        best_param = best_model.best_params_
        best_score = best_model.best_score_
        best_estimator = best_model.best_estimator_['clf']
        print('best parameters:', best_param)
        print('best score:', best_score)
        print('best estimator:',best_estimator)

        #  retrain it on the entire dataset
        features = list(X_train.columns)
        coef = []
        if name=='LogisticRegression':
            coef.append( dict(zip(['intercept']+features, np.concatenate((best_estimator.intercept_ ,best_estimator.coef_[0])))) )
        elif name=='MLPClassifier':
            coef.append( None )
        elif name=='SVC':
            coef.append( None )
        else:
            coef.append( dict(zip(features, best_estimator.feature_importances_)) )

        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # save results
        scores = {
            'time': time.time() - start,
            'name': name,
            'model': str(best_estimator),
            'size': [len(X_train), len(X_valid), len(X_test)],
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'sensitivity': tp/(tp+fn),
            'specificity': tn/(tn+fp),
            'AUC': roc_auc_score(y_test, y_pred_prob[:, 1]),
            'f1_test': f1_score(y_test, y_pred),
            'coef': coef,
            'best_param': best_param,
        }
        
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in scores.items()) + "}")

        # if scores['name']=='LogisticRegression':
        #     import shap
        #     explainer = shap.TreeExplainer(model)
        #     scores['shap_values'] = explainer.shap_values(X_train)
        if outpath:
            with open(Path.joinpath(outpath, name + ".pkl"), 'wb') as f:
                pickle.dump(grid_search.best_estimator_, f)
            print("Model written out to {}".format(outpath))
            with open(Path.joinpath(outpath, name + "_prob.pkl"), 'wb') as f:
                pickle.dump(y_pred_prob, f)
            print("Model probabilities written out to {}".format(outpath))

        yield scores

