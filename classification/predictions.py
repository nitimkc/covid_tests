# #########################################################################
# To run this script in command line provide following arguments:
#   1. - path to folder where exists:
#        a. data for which predictions are to be made
#        b. data_info.txt file specifying feature and target columns
#   2. - path to folder where the best model info is stored

# The script 
# 1. loads data for which prediction is to be made
# 2. loads 'data_info_txt' that specifies the columns to use
#     as feature and target
# 3. loads 'column_means.pkl' which contains column mean for 
#    the columns best model was trained on
# 4. processes the columns specified in 2
# 5. loads information on best model: model, probabilities and scores
# 6. makes predictions based on above model and saves the prediction 
#    probabilities and percentiles along with the original data (as csv)
#    in "predictions" folder under same directory as the data

# Example run:
# python prediction.py C:\Users\path_to_herokuapp C:\Users\path_to_project 
# #########################################################################

from pathlib import Path
import sys
import os
import ast
import csv
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from reader import CsvReader

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    Creates path profiles from provided path, data directory and results directory
    """)
    parser.add_argument("predictiondata_dir", help="Directory where data (in csv) and information of which columns to use (in .txt) is stored")
    parser.add_argument("model_dir", help="Heroku app directory path where the best model info is stored")
    args = parser.parse_args()

    DATA = Path(args.predictiondata_dir)
    MODEL = Path(args.model_dir)

    # DATA = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\predictiondata')
    # MODEL = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_predictor\model')
    
    # check if directories exists
    if ( (DATA.exists()) and (MODEL.exists()) ):
        print("Data : " , DATA)
        print("Best model : " , MODEL)
    else:
        sys.exit('supplied directories does not exist')

    # 1. read data
    data = CsvReader( str(DATA), prediction=True ) 
    docs = list(data.rows())
    print('no. of patient records :', len(docs))
    
    # 2. read file specifying column names to select from data
    info_files = [f for f in os.listdir(DATA) if f.endswith("data_info.txt")]
    data_info = {}
    if 'data_info.txt' not in info_files:
        print("file: 'data_info.txt' not in info folder")
    else:
        f = open(Path.joinpath(DATA, 'data_info.txt'),'r')
        contents = f.read()
        data_info = ast.literal_eval(contents)
        f.close()

    # 3. read file specifying column means    
    with open(Path.joinpath(MODEL, 'column_means.pkl'), 'rb') as f: 
        col_means = pickle.load(f)

    # 4. select data
    features = pd.DataFrame(list(data.fields(data_info['features'])))
    X = features.copy() 
    for i in X:
        X[i] = pd.to_numeric(X[i], downcast="float")        # convert to numeric
        mu = col_means[i]
        vals = [0,1]
        if X[i].isnull().any()==True:                      # if a value is missing
            X[i+'_1'] = np.where(X[i].isnull(), 1.0, 0.0)   # create new dummy column, 1=missing in original
            X[i] = X[i].fillna(mu)                          # fill missing with mean
    records = X.to_dict('records')
    final_cols = list(records[0].keys())
    X = [ list(i.values()) for i in records ] 

    # 5. load best model info
    with open(Path.joinpath(MODEL, "best_model.pkl"), 'rb') as f: 
        best_model = pickle.load(f)
    print("best_model:", best_model['clf'])
    with open(Path.joinpath(MODEL, "X_test.pkl"), 'rb') as f:
        test_data = pickle.load(f)  
    with open(Path.joinpath(MODEL, "best_model_score.pkl"), 'rb') as f:
        score = pickle.load(f) 
    
    # 6. get predictions and percentile
    X_test = pd.DataFrame(test_data, columns=[k for k,v in records[0].items()])
    N = len(X)
    y, y_percentile = np.zeros(N), np.zeros(N)
    for n,i,j, in zip(range(N), records, X):
        print(n)
        print(i,j)
        y[n] = best_model.predict_proba(np.array(j).reshape(1,-1))[:,1]
        print(y)
        
        apt7 = i['Ave_Pos_Past7d']
        cond1 = X_test['Ave_Pos_Past7d']>=(apt7-.3)
        cond2 = X_test['Ave_Pos_Past7d']>=(apt7+.3)
        filtered_X_test = X_test[(cond1)&(cond2)]

        y_prob_test = best_model.predict_proba(filtered_X_test)[:,1]
        y_percentile[n] = np.round( stats.percentileofscore(y_prob_test, y[n]),1)
        print(y_percentile)

    # combine original data and results
    result = pd.DataFrame([i['Patient ID'] for i in docs], columns=['Patient ID'])
    result = pd.concat([result, pd.DataFrame(features)], axis=1)
    result['prediction_probability'] = y
    result['prediction_percentile'] = y_percentile
    
    # save above as csv "predictions" folder
    # create folder if it does not exist
    filepath = Path.joinpath(DATA, 'predictions', 'predictions.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if Path.joinpath(DATA, 'predictions')==filepath.parent:
        result.to_csv(filepath, index=False)
        print('model predictions saved as', filepath.name)
        print('model predictions saved in', Path.joinpath(DATA, 'predictions'))
