from pathlib import Path
import os
import csv
import errno
import json
import logging

from reader import CsvReader
from loader import CorpusLoader
from build import binary_models
from build import score_models

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

ROOT = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\covid_symptoms')
RECORDS = Path.joinpath(ROOT, 'data')#, 'RawData_IMOH_Sep_20.csv')
RESULTS = Path.joinpath(ROOT, 'results')

filename = 'RawData_IMOH_Sep_20.csv'

if __name__ == '__main__':
    data = CsvReader( RECORDS.joinpath(filename) )
    docs = list(data.rows())

    cols = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'sixtiesplus', 'Gender', 'contact', 'abroad']#, 'Validation']
    X = [list(i.values()) for i in data.fields(cols) ]
    X_int = [[int(i) for i in elist] for elist in X ]
    y = list(data.fields('testresult'))
    groups = list(data.fields('Validation'))

    loader = CorpusLoader(X_int, y, idx=groups) 

    for scores in score_models(binary_models, loader):
        print(scores)
        result_filename = 'results.json'
        with open(Path.joinpath(RESULTS,result_filename), 'a') as f:
            f.write(json.dumps(scores) + '\n')


