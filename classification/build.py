from logging import raiseExceptions
import numpy as np 
import os
import time
from collections import Counter
# import unicodedata

from loader import CorpusLoader

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc

def create_pipeline(estimator, reduction=False):

    steps = [ ]
    steps.append(('classifier', estimator))

    return Pipeline(steps)

binary_models = []
binary_models.append(create_pipeline(LogisticRegression(), False))
binary_models.append(create_pipeline(RandomForestClassifier(n_estimators = 1000, random_state = 42), False))

def score_models(models, loader):

    for model in models:
        name = model.named_steps['classifier'].__class__.__name__
        print(model, name)  

        X_train, X_valid, X_test, y_train, y_valid, y_test = loader.sets()
        n_Xtrn = len(X_train)
        n_Xval = len(X_valid)
        n_Xtst = len(X_test)
        print('n_train:', n_Xtrn, '\nn_valid:', n_Xval, '\nn_test:' , n_Xtst)

        # Create a list where train data indices are -1 and validation data indices are 0
        test_fold = [-1 for i in range(n_Xtrn)] + [0 for i in range(n_Xval)]
        ps = PredefinedSplit(test_fold=test_fold)
        split = [(range(n_Xtrn), range(n_Xtrn, n_Xtrn + n_Xval))]
        
        gridsearch_pipe = Pipeline([
            ('classifier', model)
        ])
        
        C = np.logspace(0,4,10)
        penalty = ['l1', 'l2']

        if name == 'logistic':               
            param_grid = {'classifier__C':C, 'classifier__penalty':penalty}
            # param_grid = dict(C=C, penalty=penalty)
            # param_grid = {'kernel':('linear', 'rbf'), 'C':np.logspace(0,4,10), 'penalty':penalty}
        elif name == 'randomforest':         
            param_grid = {'classifier__C':C, 'classifier__penalty':penalty}
            # param_grid = dict(C=C, penalty=penalty)
            # param_grid = {'kernel':('linear', 'rbf'), 'C':np.logspace(0,4,10), 'penalty':penalty}
        else:           
            param_grid = {'classifier__C':C, 'classifier__penalty':penalty}
            # param_grid = dict(C=C, penalty=penalty)
            # param_grid = {'kernel':('linear', 'rbf'), 'C':np.logspace(0,4,10), 'penalty':penalty}

        grid_search = GridSearchCV(estimator=gridsearch_pipe, param_grid=param_grid, cv=split)
        grid_search.fit(X_train+X_valid, y_train+y_valid)
        best_param = grid_search.best_params_
        print(grid_search.best_score_)
        print(best_param)

        best_model = model()

        scores = {
            'model': str(model),
            'name': name,
            'size': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_valid': [],
            'f1_train': [],
            'auc': [],
            'time': [],
        }

        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        scores['time'].append(time.time() - start)
        scores['size'].append([len(X_train), len(X_test)])
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        scores['f1_valid'].append(f1_score(y_test, y_pred, average='weighted'))
        scores['f1_train'].append(f1_score(y_train, y_train_pred, average='weighted'))
        scores['auc'].append(f1_score(y_train, y_train_pred, average='weighted'))
            
        print('model: ', scores['name'])
        print('accuracy: ', scores['accuracy'])
        print('precision: ', scores['precision'])
        print('recall: ', scores['recall'])
        print('f1_valid: ', scores['f1_valid'])
        print('f1_train: ', scores['f1_train'])
        print('auc: ', scores['auc'])
        print('time: ', scores['time'])

        yield scores

# if __name__ == '__main__':
#     for scores in score_models(binary_models, loader):
#         with open('results.json', 'a') as f:
#             f.write(json.dumps(scores) + "\n")