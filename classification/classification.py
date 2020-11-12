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

# Example run:
# python classification.py C:\Users\XYZ\covid_tests data results
# ###################################################################

from pathlib import Path
import os
import csv
import errno
import json
import pickle
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
    parser.add_argument("root_dir", help="Main directory of the project. Must contain two subdirectories namely, data and results")
    # parser.add_argument("data_dir", help="Directory where the data is stored")
    # parser.add_argument("results_dir", help="Directory to store results")
    parser.add_argument("app_dir", help="Heroku app directory path to store best model results for online upload")

    args = parser.parse_args()

    ROOT = Path(args.root_dir)
    DATA = Path.joinpath(ROOT, 'data')
    RESULTS = Path.joinpath(ROOT, 'results')
    APP_RESULTS = Path(args.app_dir, 'model')

    print("Main : " , ROOT)
    print("Data : " , DATA)
    print("Model Results : " , RESULTS)
    print("Best Model Results : " , APP_RESULTS)

    data = CsvReader( str(DATA) ) 
    # data = CsvReader( str(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\data') )
    docs = list(data.rows())
    print('no. of patient records :', len(docs))
    
    cols = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'contact', 'abroad']
    catg_cols = ['sixtiesplus', 'Gender']

    X_dummy = data.dummies(catg_cols)
    final_cols = cols + list(X_dummy[0].keys())
    X_dummy = [ list(i.values()) for i in data.dummies(catg_cols) ]

    X = [ list(i.values()) for i in data.fields(cols) ]
    
    X =[ i+j for i,j in zip(X, X_dummy) ]
    X_int =  [[int(i) for i in elist] for elist in X ]
    print('no of fields with None value:', sum([i.count(None) for i in X_int]))
    y = list(data.fields('testresult')) # will label encode in build.py
    
    # to save model and its scores
    all_models = [] 
    all_scores = []
    all_auc = []
    # RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\results')

    # train the data and save test results as well as the model itself
    split_set = list(data.fields('Validation'))
    loader = CorpusLoader(X_int, y, idx=split_set) # using predefined split
    # loader = CorpusLoader(X_int, y, idx=None)    # using cv

    for scores in score_models(binary_models, loader, split_idx=True, outpath=RESULTS):
    # for scores in score_models(binary_models, loader, k=10, outpath=RESULTS):
        all_models.append(scores['name'])
        all_scores.append(scores)
        all_auc.append(scores['auc'])
        result_filename = 'results.json'
        with open(Path.joinpath(RESULTS, result_filename), 'a') as f:
            f.write(json.dumps(scores) + '\n')

    # save the best model based on stored score
    auc = [i['auc'] for i in all_scores]
    best_model = all_models[0]#all_models[auc.index(max(auc))]
    
    # load the best model and best model probabilities
    with open(Path.joinpath(RESULTS, (best_model+'.pkl')), 'rb') as f: 
        model = pickle.load(f)
    with open(Path.joinpath(RESULTS, best_model+'_prob.pkl'), 'rb') as f: 
        prob = pickle.load(f)
    
    # save above as best_model.pkl and best_model_prob.pkl in the app folder
    APP_RESULTS = r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_predictor\model'
    with open(Path.joinpath(APP_RESULTS, "best_model.pkl"), 'wb') as f:
                pickle.dump(model, f)
    with open(Path.joinpath(APP_RESULTS, "best_model_prob.pkl"), 'wb') as f:
                pickle.dump(prob, f)
