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
        self.index = idx
        self.groups = np.unique(idx)
    
    def get_idx(self, index=None):    
        if self.index is None:
            return range(0, len(self.y))
        else:
            if len(self.index) != len(self.y):
                raiseExceptions('length of index must be equal to length of X')
            idxval = np.array(self.index)
            pos_sorted = np.argsort(idxval)
            sorted_idxval = idxval[pos_sorted]

            vals, start_idx, count = np.unique(sorted_idxval, return_counts=True, return_index=True)
            splitted_idx = np.split(pos_sorted, start_idx[1:])
            splitted_idx = [list(i) for i in splitted_idx]
            return splitted_idx

    def documents(self):
        if self.index is None:
            return self.X
        else:           
            split_X = [ [self.X[i] for i in sub_idx ] for sub_idx in self.get_idx() ]
            return split_X

    def labels(self):
        if self.index is None:
            return self.y
        else:           
            split_y = [ [self.y[i] for i in sub_idx ] for sub_idx in self.get_idx() ]
            return split_y

    def sets(self):
        if self.index is None:
            train_ratio = 0.50
            validation_ratio = 0.25
            test_ratio = 0.25
            X_train, X_test, y_train, y_test = tts(self.documents(), self.labels(), test_size=1 - train_ratio, random_state=42)
            X_valid, X_test, y_valid, y_test = tts(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42)
            
            train = [X_train, y_train]
            valid = [X_valid, y_valid]
            test = [X_test, y_test]
        else:
            X_dict = {}
            y_dict = {}
            for grp, X_grp, y_grp in zip(self.groups, self.documents(), self.labels()):
                X_dict[grp] = X_grp
                y_dict[grp] = y_grp
            train = [ X_dict[ get_key(X_dict, '0')], y_dict[ get_key(y_dict, '0')] ]
            valid = [ X_dict[ get_key(X_dict, '1')], y_dict[ get_key(y_dict, '1')] ]
            test  = [ X_dict[ get_key(X_dict, '2')], y_dict[ get_key(y_dict, '2')] ]

        return train, valid, test

# dt_split = list(data.fields('Validation'))
# loader = CorpusLoader(X, y, idx=dt_split)
# grps = loader.groups
# train, valid, test = loader.sets()


