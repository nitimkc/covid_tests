# #################################################################################
# To run this script in command line provide following arguments:
#   1. path to project folder. MUST have the three folders:
#      - 1. "data" folder containing data files (see DATA REQUIREMENTS below)
#      - 2. "info" folder containing information on which columns to use
#      - 3. "results" folder to store each model info
#   2. path to heroku app to store best model info
#      - Must have "model" folder inside

# This script 
#   1. loads 'data_info.txt' from "info" directory to determine 
#      features (categorical and numerical), target and validation  
#      (if exists) columns to select from data 
#   2. loads data in csv file from "data" directory using reader.py
#   3. processes categorical columns
#   4. filters data using filter info given in 'data_info.txt'
#   5. removes rows if values are missing in numerical features 
#   6. obtains train, test and validation sets through loader.py
#      If data contains "validation" column, splits data as specified 
#   7. runs multiple models using build.py and saves the 
#      scores of each of these in results folder in filename
#      "results.json"
#   8. Picks the best model based on "auc" score. Saves best model, its
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
    
    # 1. read file specifying column names to select from data
    info_files = [f for f in os.listdir(INFO) if f.endswith("data_info.txt")]
    data_info = {}
    if 'data_info.txt' not in info_files:
        print("file: 'data_info.txt' not in info folder")
    else:
        f = open(Path.joinpath(INFO, 'data_info.txt'),'r')
        contents = f.read()
        data_info = ast.literal_eval(contents)
        f.close() 
    
    # 2. read data from features, target, validation columns
    # data = pd.read_csv(str(DATA), encoding="utf-8", header=1)
    data = CsvReader(str(DATA), target=data_info['target']) 
    X_cat = pd.DataFrame(list(data.fields(data_info['cat_features'])))
    X_num = pd.DataFrame(list(data.fields(data_info['num_features'])))
    y = pd.Series(list(data.fields(data_info['target'])))
    validation = pd.Series(list(data.fields(data_info['validation'])))

    # 3. transform categorical features
    col_means = {}
    for i in X_cat:
        X_cat[i] = pd.to_numeric(X_cat[i], downcast="float")        # convert to numeric
        col_means[i] = None
        if X_cat[i].isnull().any()==True:                       # if a value is missing
            mu = X_cat[i].mean()
            col_means[i] = mu
            X_cat[i+'_1'] = np.where(X_cat[i].isnull(), 1.0, 0.0)               # create new dummy column, 1=missing in original
            X_cat[i] = X_cat[i].fillna(mu)                                      # fill missing with mean
    with open(Path.joinpath(RESULTS, 'column_means.pkl'), 'wb') as f: 
         pickle.dump(col_means, f)                                      # save column means for use in prediction.py
    
    # 4. filter data as defined in data_info.txt file
    filter_col = pd.Series(list(data.fields(data_info['filter_feature'])))
    filter_logic = data_info['filter_logic'] 
    filter_val = data_info['filter_value']

    if filter_logic == '==':
        filter_name = 'eqto'
        filter_idx = filter_col[filter_col==filter_val].index
    if filter_logic == '>=':
        filter_name = 'grthan'
        filter_idx = filter_col[filter_col>=filter_val].index
    if filter_logic == '<=':
        filter_name = 'lsthan'
        filter_idx = filter_col[filter_col<=filter_val].index

    X = pd.concat([X_num, X_cat], axis=1)
    X = X.iloc[filter_idx]
    y = y[filter_idx]
    validation = validation[filter_idx]

    # 5. remove rows with missing values in numeric features
    X[data_info['num_features']] = X[data_info['num_features']].apply(pd.to_numeric, errors='coerce')
    null_identifier = X.isnull().any(axis=1)
    null_idx = null_identifier.index[null_identifier]

    X = X.drop(null_idx)
    y = y.drop(null_idx)
    validation = validation.drop(null_idx)

    # 6. prepare data for modelling and determine test, train and validation splits 
    records = X.to_dict('records')
    final_cols = list(records[0].keys())
    X = [ list(i.values()) for i in records ] 
    y = list(y) # will label encode in build.py

    if data_info.get('validation'):
        split_idx = True
        split_set = [str(i) for i in validation]
        loader = CorpusLoader(X, y, idx=split_set) # using predefined split
    else: 
        split_idx = False
        loader = CorpusLoader(X, y, idx=None)      # using cv

    # 7. train models and save models and its scores    
    for scores in score_models(binary_models, loader, split_idx=split_idx, k=5, features=final_cols, outpath=RESULTS):
        print(scores)
        result_filename = 'results.json'
        with open(Path.joinpath(RESULTS, result_filename), 'a') as f:
            f.write(json.dumps(scores) + '\n')
    

    
#########################################################################
# Part 8
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
    with open(Path.joinpath(APP_RESULTS, "best_model_"+filter_name+filter_val+".pkl"), 'wb') as f:
        pickle.dump(model, f)
    with open(Path.joinpath(APP_RESULTS, "best_model_score_"+filter_name+filter_val+".pkl"), 'wb') as f:
        pickle.dump(best_model, f)
    with open(Path.joinpath(APP_RESULTS, "best_model_prob_"+filter_name+filter_val+".pkl"), 'wb') as f:
        pickle.dump(prob, f)
    with open(Path.joinpath(APP_RESULTS, "column_means_"+filter_name+filter_val+".pkl"), 'wb') as f: 
         pickle.dump(col_means, f)       
    with open(Path.joinpath(APP_RESULTS, "data_info_"+filter_name+filter_val+".pkl"), 'wb') as f: 
         pickle.dump(data_info, f)                                      
        
