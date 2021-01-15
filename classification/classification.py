# #################################################################################
# To run this script in command line provide following arguments:
#   1. path to project folder. MUST have the three folders:
#      - 1. "data" folder containing data files (see DATA REQUIREMENTS below)
#      - 2. "info" folder containing information on which columns to use
#      - 3. "results" folder to store each model info
#   2. path to heroku app to store best model info
#      - Must have "model" folder inside

# This script 
#   1. loads data in csv file from "data" directory using reader.py
#   2. loads 'data_info.txt' from "info" directory to determine 
#      features, target and validation (if exisits) columns to select 
#      from data 
#   3. processes selected data and saves columns means
#   4. obtains train, test and validation sets through loader.py
#      If data contains "validation" column, splits data as specified 
#   5. runs multiple models using build.py and saves the 
#      scores of each of these in results folder in filename
#      "results.json"
#   6. Picks the best model based on "auc" score. Saves best model, its
#      score, test data and column means in designated app folder. 
#      Also saves the result in csv in order of model rank in results folder

# Example run:
# python classification.py C:\Users\path_to_project C:\Users\XYZ\path_to_herokuapp

# DATA REQUIREMENTS: 
#   1. Must be csv file
#   2. For all columns with missing values:
#       a. a new dummy column is created where 1 indicates missing in original column
#       b. missing value in original column is replaced with mean value of that column
#   3. For validation column each row must indicate: "training", "validation" or "test"

# #################################################################################


from pathlib import Path
import os
import ast
import csv
import errno
import json
import pickle
import logging
import argparse
from collections import Counter
import pandas as pd
import numpy as np

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

    # ROOT = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests')
    # DATA = Path.joinpath(ROOT, 'data')
    # INFO = Path.joinpath(ROOT, 'info')
    # RESULTS = Path.joinpath(ROOT, 'results')   
    # APP_RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_predictor\model')

    # 1. read data
    data = CsvReader(str(DATA)) 
    docs = list(data.rows())
    print('no. of patient records :', len(docs))
    
    # 2. read file specifying column names to select from data
    info_files = [f for f in os.listdir(INFO) if f.endswith("data_info.txt")]
    data_info = {}
    if 'data_info.txt' not in info_files:
        print("file: 'data_info.txt' not in info folder")
    else:
        f = open(Path.joinpath(INFO, 'data_info.txt'),'r')
        contents = f.read()
        data_info = ast.literal_eval(contents)
        f.close()

    # 2. select data
    X = pd.DataFrame(list(data.fields(data_info['features'])))
    col_means = {}
    for i in X:
        X[i] = pd.to_numeric(X[i], downcast="float")        # convert to numeric
        col_means[i] = None
        vals = [0,1]
        if X[i].isnull().any()==True:                       # if a value is missing
            mu = X[i].mean()
            col_means[i] = mu
            X[i+'_1'] = np.where(X[i].isnull(), 1.0, 0.0)               # create new dummy column, 1=missing in original
            X[i] = X[i].fillna(mu)                                      # fill missing with mean
    with open(Path.joinpath(RESULTS, 'column_means.pkl'), 'wb') as f: 
         pickle.dump(col_means, f)                                      # save column means for use in prediction.py
    
    records = X.to_dict('records')
    final_cols = list(records[0].keys())
    X = [ list(i.values()) for i in records ] 
    y = list(data.fields(data_info['target'])) # will label encode in build.py

    # 3. determine how to split test, train and validation set
    if data_info.get('validation'):
        split_idx = True
        split_set = [str(i) for i in list(data.fields('Validation'))]
        loader = CorpusLoader(X, y, idx=split_set) # using predefined split
    else: 
        split_idx = False
        loader = CorpusLoader(X, y, idx=None)      # using cv

    # 4. train models and save models and its scores    
    for scores in score_models(binary_models, loader, split_idx=split_idx, k=5, features=final_cols, outpath=RESULTS):
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
            all_scores.append(json.loads(line))

    # sort model by auc and add key rank to the dictionary
    all_scores = sorted(all_scores, key=lambda x:x['AUC'], reverse=True)
    for scores,rank in zip(all_scores, range(1,len(all_scores)+1)):
        print(rank)
        scores['rank'] = rank
        scores['positiverate'] = sum([int(i) for i in y if i=='1'])/len(y)
        print(scores)

    # save all scores as csv
    df = pd.DataFrame.from_dict(all_scores)
    df = df[['rank', 'name',  'AUC', 'sensitivity', 'specificity', 'accuracy', 'precision', 
             'recall', 'f1_test', 'model', 'size', 'coef', 'best_param', 'time' ]]
    df.to_csv(Path.joinpath(RESULTS,'results.csv'), index=False)
    
    # load the best model and test set
    best_model = all_scores[0]
    print('best_model is: ', best_model['name'])
    with open(Path.joinpath(RESULTS, (best_model['name']+'.pkl')), 'rb') as f: 
        model = pickle.load(f)
    with open(Path.joinpath(RESULTS, (best_model['name']+'_prob.pkl')), 'rb') as f:
        prob = pickle.load(f) 
    
    # save req info of best model in the heroku app folder
    with open(Path.joinpath(APP_RESULTS, "best_model.pkl"), 'wb') as f:
        pickle.dump(model, f)
    with open(Path.joinpath(APP_RESULTS, "best_model_score.pkl"), 'wb') as f:
        pickle.dump(best_model, f)
    with open(Path.joinpath(APP_RESULTS, "best_model_prob.pkl"), 'wb') as f:
        pickle.dump(prob, f)
    with open(Path.joinpath(APP_RESULTS, 'column_means.pkl'), 'wb') as f: 
         pickle.dump(col_means, f)       
    with open(Path.joinpath(APP_RESULTS, 'data_info.pkl'), 'wb') as f: 
         pickle.dump(data_info, f)                                      
        
