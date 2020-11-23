#########################################################################
# To run this script in command line provide following arguments:
#   1. path to project folder. MUST have the two folders:
#      - "data" folder containing data files (csv file only)
#      - "results" folder to store each model info
#   2. path to heroku app to store best model info. 
#      ust have "model" folder inside

# This script 
#   1. loads data from data directory and selects following columns:
#      'Validation', 'testresult', 
#      'cough', 'fever', 'sore_throat', 'shortness_of_breath', 
#      'head_ache', 'sixtiesplus', 'Gender', 
#   2. adds dummy variables for:
#      'sixtiesplus', 'Gender', 'contact', 'abroad'
#   3. processes them through loader.py to obtain train, test and 
#      validation splits as specified in 'Validation' column 
#   4. Finally, runs multiple models using build.py and saves the 
#      scores of each of thse in results folder in filename
#      "results.json"
#   5. Picks the best model based on "auc" score and saves in heroku 
#      app folder

# Example run:
# python classification.py C:\Users\XYZ\covid_tests C:\Users\XYZ\covid_app
# #########################################################################

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

from buildNN import NNmodel
from buildNN import score_NN 

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
    
    cols = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache']
    catg_cols = ['sixtiesplus', 'Gender', 'contact', 'abroad']

    X_dummy = data.dummies(catg_cols)
    final_cols = cols + list(X_dummy[0].keys())
    X_dummy = [ list(i.values()) for i in data.dummies(catg_cols) ]

    X = [ list(i.values()) for i in data.fields(cols) ] 
    X =[ i+j for i,j in zip(X, X_dummy) ]
    X_int =  [[int(i) for i in elist] for elist in X ]
    print('no of fields with None value:', sum([i.count(None) for i in X_int]))

    y = list(data.fields('testresult')) # will label encode in build.py
    # print('no of target with None value:', sum([i.count(None) for i in y]))

    # train the data and save test results as well as the model itself
    split_set = list(data.fields('Validation'))
    loader = CorpusLoader(X_int, y, idx=split_set) # using predefined split
    # loader = CorpusLoader(X_int, y, idx=None)    # using cv

    # to save model and its scores
    # RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\results')
    for scores in score_models(binary_models, loader, split_idx=True, outpath=RESULTS):
    # for scores in score_models(binary_models, loader, k=10, outpath=RESULTS):
        print(scores)
        result_filename = 'results.json'
        with open(Path.joinpath(RESULTS, result_filename), 'a') as f:
            f.write(json.dumps(scores) + '\n')
    
    # for scores in score_NN(NNmodel, loader, split_idx=True, outpath=RESULTS):
    # # for scores in score_models(binary_models, loader, k=10, outpath=RESULTS):
    #     print(scores)
    #     result_filename = 'results.json'
    #     with open(Path.joinpath(RESULTS, result_filename), 'a') as f:
    #         f.write(json.dumps(scores) + '\n')
    
#########################################################################
# Part 5
#########################################################################
    # load all results
    all_scores = []
    with open(Path.joinpath(RESULTS, 'results.json'), 'r') as f: 
        for line in f:
            print(line)
            all_scores.append(json.loads(line))

    # find best model and its scores
    aucs = [i['auc'] for i in all_scores]
    names = [i['name'] for i in all_scores]
    best_model_idx = aucs.index(max(aucs))
    best_model = names[best_model_idx]
    print('best_model is: ', best_model)
    
    # load the best model and its probabilities on test set
    with open(Path.joinpath(RESULTS, (best_model+'.pkl')), 'rb') as f: 
        model = pickle.load(f)
    with open(Path.joinpath(RESULTS, best_model+'_prob.pkl'), 'rb') as f: 
        prob = pickle.load(f)
    best_model_scores = all_scores[best_model_idx]
    
    # save req info of best model in the heroku app folder
    # APP_RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_predictor\model')
    with open(Path.joinpath(APP_RESULTS, "best_model.pkl"), 'wb') as f:
            pickle.dump(model, f)
    with open(Path.joinpath(APP_RESULTS, "best_model_prob.pkl"), 'wb') as f:
            pickle.dump(prob, f)
    with open(Path.joinpath(APP_RESULTS, "best_model_score.pkl"), 'wb') as f:
            pickle.dump(best_model_scores, f)
