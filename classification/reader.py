import os
import csv
import errno
import time
import glob
from six import string_types
import pandas as pd
import numpy as np

class CsvReader(object):

    def __init__(self, path, prediction=False):
        self.path = path
        self.prediction = prediction

    def rows(self):
        try:
            for file in glob.glob(self.path+'\*.csv'):
                with open(file, "r", encoding='utf-8') as f:
                    self.reader = csv.DictReader(f)
                    for row in self.reader:
                        if self.prediction:
                            yield row
                        else:
                            if (dict(row)['testresult']=='0') or  (dict(row)['testresult']=='1'): # if training ensure testresult has as 0,1
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
