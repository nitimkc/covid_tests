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
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_auc_score

binary_models = []
# binary_models.append( LogisticRegression(random_state = 42) )
# binary_models.append( RandomForestClassifier(random_state = 42) )
binary_models.append( SVC(random_state = 42, probability=True) )
# binary_models.append( SGDClassifier(random_state = 42) )
# binary_models.append( MultinomialNB(random_state = 42) )
# binary_models.append( GaussianNB(random_state = 42) )


parameters = [
    # {'clf__C': ( np.logspace(-5, 1, 5) ),
    #     'clf__penalty': ['l1', 'l2', 'none'] # regularization paramter
    #     },
    # {'clf__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    #     'clf__min_samples_split': [2, 5, 10],
    #     'clf__min_samples_leaf': [1, 2, 4]
    #     },
    {'clf__C': [0.001, 0.01, 0.1, 1, 10],
        'clf__gamma': [0.001, 0.01, 0.1, 1]
        }
]

 
def score_models(models, loader, split_idx=False, k=5, outpath=None):
    
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
            grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=ps, scoring='roc_auc')
        else:
            grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=k, scoring='roc_auc')
        
        start = time.time()
        best_model = grid_search.fit(X_train+X_valid, np.concatenate( (y_train,y_valid)))
        best_param = best_model.best_params_
        best_score = best_model.best_score_
        best_estimator = best_model.best_estimator_['clf']
        print(best_param)
        print(best_score)
        print(best_estimator)
        #  retrain it on the entire dataset
        if name=='LogisticRegression':
            coef = np.concatenate((best_estimator.intercept_ , best_estimator.coef_[0]))
        else:
            coef = list([None])
        
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)

        # y_valid_pred = best_model.predict(X_valid)

        # save results
        scores = {
            'time': time.time() - start,
            'name': name,
            'model': str(best_estimator),
            'size': [len(X_train), len(X_valid), len(X_test)],
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            # 'auc': roc_auc_score(y_test, y_pred, average='weighted'),
            'auc': roc_auc_score(y_test, y_pred_prob[:, 1], average='weighted'),
            'f1_test': f1_score(y_test, y_pred, average='weighted'),
            # 'f1_valid': f1_score(y_valid, y_valid_pred, average='weighted'),
            'coef': list(coef),
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