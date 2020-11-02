import numpy as np
from logging import raiseExceptions
from numpy.core.defchararray import split
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as tts

def get_key(dictionary, string):
    if len(dictionary.keys()) > 3:
        raiseExceptions('There should be only 3 splits of data')
    return [key for key in dictionary.keys() if string in key.lower()][0]

class CorpusLoader(object):

    def __init__(self, X, y, idx=None):
        self.X = X
        self.y = y
        self.n = len(y)
        self.index = idx
        self.groups = np.unique(idx)
    
    # Validation data (X_val, y_val) is currently inside X_train, which will be split using PredefinedSplit inside GridSearchCV
    def sets(self):
        n = len(self.y)
        m = int(n*0.25)
        
        X_test, X_rest, y_test, y_rest = self.X[ :m], self.X[m+1: ], self.y[ :m], self.y[m+1: ]
        # X_test, X_rest, y_test, y_rest = X[ :m], X[m+1: ], y[ :m], y[m+1: ]
        X_train, X_valid, y_train, y_valid = tts(X_rest, y_rest, train_size = 0.666, random_state = 2020)
        print(len(X_train), len(X_valid), len(X_test))

        return X_train, X_valid, X_test, y_train, y_valid, y_test

# dt_split = list(data.fields('Validation'))
# loader = CorpusLoader(X, y, idx=dt_split)
# grps = loader.groups
# train, valid, test = loader.sets()


