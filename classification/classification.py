###################################################################
# To run this script in command line provide following arguments:
#   1. path to project folder (MUST have the two directories below)
#   2. name of directory that has the data files (csv file only)
#   3. name of directory where results should be stored

# The script loads data from data directory, selects following columns -
#  'Validation', 'testresult', 
#  'cough', 'fever', 'sore_throat', 'shortness_of_breath', 
#  'head_ache', 'sixtiesplus', 'Gender', 'contact', 'abroad'
# and processes them through loader.py to obtain train, test and 
# validation splits as specified in 'Validation' column. Finally,
# it runs multiple models using build.py and saves the scores of 
# each of these models in results directory in file "results.json"

# file path:
# C:\Users\niti.mishra\Documents\2_TDMDAL\covid_tests\covid_tests
###################################################################

from pathlib import Path
import os
import csv
import errno
import json
import logging
import argparse 

from reader import CsvReader
from loader import CorpusLoader
from build import binary_models
from build import score_models

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    Creates path profiles from provided path, data directory and results directory
    """)
    parser.add_argument("root_dir", help="Main directory of the project")
    parser.add_argument("data_dir", help="Directory where the data is stored")
    parser.add_argument("results_dir", help="Directory to store results")

    args = parser.parse_args()

    ROOT = Path(args.root_dir)
    DATA = Path.joinpath(ROOT, args.data_dir)
    RESULTS = Path.joinpath(ROOT, args.results_dir)

    print("Main : " , ROOT)
    print("Data : " , DATA)
    print("Results : " , RESULTS)

    data = CsvReader( str(DATA) ) 
    # data = CsvReader( str(r'C:\Users\niti.mishra\Documents\2_TDMDAL\covid_tests\covid_tests\data') )
    docs = list(data.rows())
    print('no. of patient records :', len(docs))
    
    cols = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'sixtiesplus', 'Gender', 'contact', 'abroad']#, 'Validation']
    X = [list(i.values()) for i in data.fields(cols) ]
    X_int = [[int(i) for i in elist] for elist in X ]
    y = list(data.fields('testresult')) # will label encode in build.py
    split_set = list(data.fields('Validation'))

    loader = CorpusLoader(X_int, y, idx=split_set) 

    # all_scores = []
    for scores in score_models(binary_models, loader):
        # all_scores.append(scores)
        result_filename = 'results.json'
        with open(Path.joinpath(RESULTS,result_filename), 'a') as f:
            f.write(json.dumps(scores) + '\n')