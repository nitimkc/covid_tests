import os
import csv
import errno
import time
import glob
from six import string_types
import pandas as pd

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
        extract particular fields from the json doc. Can be string or an 
        iterable of fields. If just one fields in passed in, then the values 
        are returned, otherwise dictionaries of the requested fields returned
        """
        if isinstance(fields, string_types):
            fields = [fields,]

        if len(fields) == 1:
            for row in list(self.rows()):
                if fields[0] in row:
                    yield row[fields[0]]
        else:
            for row in self.rows():
                yield{
                    key: row.get(key, None)
                    for key in fields
                }

    def dummies(self, field):
        """
        take the unique values of a field and convert them to dummy columns
        """               
        X_df =  pd.DataFrame(self.fields(field))
        X_dummy = pd.get_dummies(X_df, drop_first=True)   

        return X_dummy.to_dict('records')
