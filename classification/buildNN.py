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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_auc_score


# Function to create model, required for KerasClassifier
def create_NNmodel():
    # create model
    model = Sequential()
    model.add(Dense(6, input_dim=11, activation='relu')) # first layer
    model.add(Dense(3, activation='relu'))                       # second layer
    model.add(Dense(1, activation='sigmoid'))                    # output layer
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# create model
np.random.seed(42)
NNmodel = KerasClassifier(build_fn=create_NNmodel, verbose=1)

# grid search parameters
parameters = [
    {'clf__batch_size':[10, 50, 75, 100, 150], 
     'clf__epochs':[10, 25, 50],
#     'clf__learn_rate':[0.001, 0.01, 0.1, 0.2, 0.3],
#     'clf__momentum':[0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
#     'clf__init_mode':['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
    },
]

def score_NN(model, loader, split_idx=False, k=5, outpath=None):
    
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

    pipe = Pipeline([
        ('clf', model),
        ])

    # Create gridsearch with specified valid set
    if split_idx: 
        # Create a list where train data indices are -1 and validation data indices are 0
        test_fold = [-1 for i in range(n_Xtrn)] + [0 for i in range(n_Xval)]
        ps = PredefinedSplit(test_fold=test_fold)
        grid_search = GridSearchCV(estimator=pipe, param_grid=parameters, cv=ps, scoring='roc_auc', n_jobs=-1, verbose=1, return_train_score=True)
    else:
        grid_search = GridSearchCV(estimator=pipe, param_grid=parameters, cv=k, scoring='roc_auc', n_jobs=-1)
    
    start = time.time()
    grid_results = grid_search.fit( np.concatenate((X_train,X_valid)), np.concatenate((y_train,y_valid)) )
    print( time.time() - start)
    best_param = grid_results.best_params_
    best_score = grid_results.best_score_
    print('Best : {}, using {}'.format(best_param, best_score))
    
    means = grid_results.cv_results_['mean_test_score']
    stds = grid_results.cv_results_['std_test_score']
    params = grid_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('{},{} with: {}'.format(mean, stdev, param))

    # use best paramters and retrain model and get prediction probabilities
    model = KerasClassifier(build_fn=create_NNmodel, verbose=10,  batch_size=best_param['clf__batch_size'], epochs=best_param['clf__epochs'])
    model.fit(np.concatenate((X_train,X_valid)), np.concatenate((y_train,y_valid)))
    y_pred = model.predict(np.array(X_test))
    y_pred_prob = model.predict_proba(np.array(X_test))

    # save results
    name = '2 layer NN'
    coef = list([None])
    scores = {
        'time': time.time() - start,
        'name': name,
        'model': best_param,
        'size': [len(X_train), len(X_valid), len(X_test)],
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'auc': roc_auc_score(y_test, y_pred_prob[:, 1], average='weighted'),
        'f1_test': f1_score(y_test, y_pred, average='weighted'),
        'coef': list(coef),
    }
    print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in scores.items()) + "}")

    if outpath:
        with open(Path.joinpath(outpath, name + ".pkl"), 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

        with open(Path.joinpath(outpath, name + "_prob.pkl"), 'wb') as f:
            pickle.dump(y_pred_prob, f)

        print("Prediction probabilities written out to {}".format(outpath))

    yield scores