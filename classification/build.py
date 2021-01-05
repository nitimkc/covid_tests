from logging import raiseExceptions
from pathlib import Path
import numpy as np 
import os
import time
import pickle
from collections import Counter
# import unicodedata

from loader import CorpusLoader

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
# binary_models.append( LogisticRegression(random_state = 0, penalty='none') )
# binary_models.append( RandomForestClassifier(random_state = 0) )
# binary_models.append( SVC(random_state = 0, probability=True) )
binary_models.append( tree.DecisionTreeClassifier(random_state = 0) )
binary_models.append( GradientBoostingClassifier(random_state = 0) )
binary_models.append( MLPClassifier(random_state=0, hidden_layer_sizes=(6,3,1), activation='relu', solver='adam') )

parameters = [
    # {'clf__C': ( np.logspace(-5, 1, 5) ),
    #     # 'clf__penalty': ['l1', 'l2', 'none'] # regularization paramter
    #     },
    # {'clf__n_estimators':range(67,88,4), #67 Number of Trees in the Forest:
    #     'clf__max_depth': range(10,25,4), # 10 Minimum Splits per Tree:
    #     'clf__min_samples_split': range(6,13,2), #109 Minimum Size Split
    #     'clf__max_features': range(5,11,2), #9
    #     # 'clf__min_samples_leaf': [1, 2, 4],
    #     },
    # {'clf__C': [0.001, 0.01, 0.1, 1, 10],
    #     'clf__gamma': [0.001, 0.01, 0.1, 1]
    #     },
    {'clf__max_depth':[2,4,6,8,10,12],
        'clf__max_leaf_nodes': list(range(2, 50,4)),
        'clf__min_samples_split': [2, 3, 4],
        },
    {'clf__n_estimators':range(70,88,4), #86
        'clf__max_depth':range(12,25,4), #20
        # 'clf__min_samples_split':range(12,25,4),
        # 'clf__max_features':range(7,10,2),
        # 'clf__learning_rate':[0.01,.1], #0.1,
        # 'clf__subsample':[0.6,0.7,0.8], #0.6
        },
    {'clf__batch_size':[10, 50, 75, 100, 150], 
     'clf__max_iter':[10, 25, 50],
        }
]

 
def score_models(models, loader, split_idx=False, k=5, features=None, outpath=None):
    
    train, valid, test = loader.sets()
    X_train, y_train = train[0], train[1]
    X_valid, y_valid = valid[0], valid[1]
    X_test, y_test = test[0], test[1]

    n_Xtrn = len(X_train)
    n_Xval = len(X_valid)
    n_Xtst = len(X_test)

    print('n_train:', n_Xtrn, '\nn_valid:', n_Xval, '\nn_test:' , n_Xtst)

    # Label encode the targets
    labels = LabelEncoder()
    y_train = labels.fit_transform(y_train)
    y_valid = labels.fit_transform(y_valid)
    y_test = labels.fit_transform(y_test)

    names = [str(i).split('(')[0] for i in models]
    for model, name, params in zip(models, names, parameters):
        print(model, '\n', name, '\n', params)
        
        pipe = Pipeline([
            ('clf', model),
            ])

        # Create gridsearch with specified valid set
        if split_idx: 
            # Create a list where train data indices are -1 and validation data indices are 0
            test_fold = [-1 for i in range(n_Xtrn)] + [0 for i in range(n_Xval)]
            ps = PredefinedSplit(test_fold=test_fold)
            grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=ps)#, scoring='roc_auc')
        else:
            grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=k)#, scoring='roc_auc')
        
        start = time.time()
        best_model = grid_search.fit( X_train+X_valid, np.concatenate( (y_train,y_valid)) )
        best_param = best_model.best_params_
        best_score = best_model.best_score_
        best_estimator = best_model.best_estimator_['clf']
        print('best parameters:', best_param)
        print('best score:', best_score)
        print('best estimator:',best_estimator)

        #  retrain it on the entire dataset
        coef = []
        if features:
            if name=='LogisticRegression':
                coef.append( dict(zip(['intercept']+features, np.concatenate((best_estimator.intercept_ ,best_estimator.coef_[0])))) )
                # coef = np.concatenate((best_estimator.intercept_ , best_estimator.coef_[0]))
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
            print("Prediction probabilities written out to {}".format(outpath))

        yield scores