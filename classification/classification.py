from pathlib import Path
import os
import ast
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
from buildNN import score_NN 

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    Creates path profiles from provided path, data directory and results directory
    """)
    parser.add_argument("root_dir", help="Main directory of the project. Must contain 3 subdirectories namely, data, info and results")
    parser.add_argument("app_dir", help="Heroku app directory path to store best model results for online upload")

    args = parser.parse_args()

    ROOT = Path(args.root_dir)
    DATA = Path.joinpath(ROOT, 'data')
    INFO = Path.joinpath(ROOT, 'info')
    RESULTS = Path.joinpath(ROOT, 'results')
    APP_RESULTS = Path(args.app_dir, 'model')

    print("Main : " , ROOT)
    print("Data : " , DATA)
    print("Additional info : " , INFO)
    print("Model Results : " , RESULTS)
    print("Best Model Results : " , APP_RESULTS)


    # read data
    data = CsvReader( str(DATA) ) 
    # data = CsvReader( str(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\data') )
    docs = list(data.rows())
    print('no. of patient records :', len(docs))
    
    # read data_info file
    # INFO = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\info')
    info_files = [f for f in os.listdir(INFO) if f.endswith(".txt")]
    data_info = {}
    if 'data_info.txt' not in info_files:
        print("file: 'data_info.txt' not in info folder")
    else:
        f = open(Path.joinpath(INFO, 'data_info.txt'),'r')
        contents = f.read()
        data_info = ast.literal_eval(contents)
        f.close()
    cols = data_info['cols']
    catg_cols = data_info['catg_cols']

    # use data _info file to find feature columns
    X = [ list(i.values()) for i in data.fields(cols) ] 
    final_cols = cols

    # determine if any columns needs to be treated as categorical
    if catg_cols is not None:
        X_dummy = data.dummies(catg_cols)
        final_cols = cols + list(X_dummy[0].keys())
        X_dummy = [ list(i.values()) for i in data.dummies(catg_cols) ]
        X =[ i+j for i,j in zip(X, X_dummy) ]

    # convert any string values to float
    X_int =  [[float(i) for i in elist] for elist in X ]
    print('no of fields with None value:', sum([i.count(None) for i in X_int]))

    # will label encode in build.py
    y = list(data.fields('testresult')) 

    # determine how to split test, train and validation set
    all_cols = list(list(data.rows())[0].keys())
    if 'Validation' in all_cols:
        split_set = list(data.fields('Validation'))
        loader = CorpusLoader(X_int, y, idx=split_set) # using predefined split
        split_idx=True
    else: 
        loader = CorpusLoader(X_int, y, idx=None)    # using cv
        split_idx=False

    # train models and save models and its scores    
    # RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\results')   
    for scores in score_models(binary_models, loader, split_idx=split_idx, k=5, features=final_cols, outpath=RESULTS):
        print(scores)
        result_filename = 'results.json'
        with open(Path.joinpath(RESULTS, result_filename), 'a') as f:
            f.write(json.dumps(scores) + '\n')
    
    # train NN and save it and its scores
    for scores in score_NN(loader, split_idx=split_idx, k=5, outpath=RESULTS):
        print(scores)
        result_filename = 'results.json'
        with open(Path.joinpath(RESULTS, result_filename), 'a') as f:
            f.write(json.dumps(scores) + '\n')
    
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
