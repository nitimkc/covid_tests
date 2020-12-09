import os
import csv
import errno
import time
import glob
from six import string_types
import pandas as pd
import numpy as np

class CsvReader(object):

    def __init__(self, path):
        self.path = path

    def rows(self):
        try:
            for file in glob.glob(self.path+'\*.csv'):
                with open(file, "r", encoding='utf8') as f:
                    self.reader = csv.DictReader(f)
                    for row in self.reader:
                        yield row
        except IOError as err:
            print("I/O error({0}): {1}".format(errno, os.strerror))
        return

    def n_rows(self):
        print(sum( (1 for doc in self.rows()) ))

    def fields(self, fields):
        """
        extract particular fields from the csv doc. Can be string or an 
        iterable of fields. If just one fields in passed in, then the values 
        are returned, otherwise dictionaries of the requested fields returned
        """
        if isinstance(fields, string_types):
            fields = [fields,]

        if len(fields) == 1:
            for row in list(self.rows()):
                    yield row[fields[0]]
        else:
            for row in self.rows():
                yield{
                    key: row.get(key, None)
                    for key in fields
                }

    def features(self, fields):
        X = pd.DataFrame(self.fields(fields))
        for i in X:
            X[i] = pd.to_numeric(X[i], downcast="float")        # convert to numeric
            vals = [0,1]
            if X[i].isin(vals).all()==False:                   # if contains values other than 0 and 1
                X[i+'_1'] = np.where(X[i].isnull(), 1.0, 0.0)       # create new dummy column, 1=missing in original
                mu = X[i].mean()
                X[i] = [mu if i not in vals else i for i in X[i]]   # fill value other than 0 and 1 with mean
        return X.to_dict('records')