###################################################################
# To run this script in command line provide following arguments:
#   1. - path to folder where the data to make predictions exists
#   2. - path to folder where the best model info is stored

# The script 
# 1. loads data for which prediction is to be made
# 2. loads 'data_info_txt' that specified the columns to use
#     as regular and/or categorical and target
# 3. loads information on best model: model, probabilities and scores
# 4. makes predictions based on above model and saves the prediction 
#    probabilities and percentiles along with the original data (as csv)
#    in "predictions" folder under same directory as the data

# Example run:
# python prediction.py C:\Users\XYZ\best_model C:\Users\XYZ\covid_tests 
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

    # check if directories exists
    if ( (DATA.exists()) and (MODEL.exists()) ):
        print("Data : " , DATA)
        print("Best model : " , MODEL)
    else:
        sys.exit('listofitems not long enough')
   
    # 1. read data
    # DATA = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\predictiondata')
    data = CsvReader( str(DATA) ) 
    docs = list(data.rows())
    print('no. of patient records :', len(docs))
    
    # 2. read file specifying column names to select from data
    info_files = [f for f in os.listdir(DATA) if f.endswith(".txt")]
    data_info = {}
    if 'data_info.txt' not in info_files:
        print("file: 'data_info.txt' not in info folder")
    else:
        f = open(Path.joinpath(DATA, 'data_info.txt'),'r')
        contents = f.read()
        data_info = ast.literal_eval(contents)
        f.close()
    cols = data_info['cols']
    catg_cols = data_info['catg_cols']
    target = data_info['target']

    # # 2. select data of columns specified above
    # X = [ list(i.values()) for i in data.fields(cols) ] 
    # final_cols = cols

    # # 2. determine if any columns needs to be treated as categorical
    # if catg_cols is not None:
    #     X_dummy = data.dummies(catg_cols)
    #     final_cols = cols + list(X_dummy[0].keys())
    #     X_dummy = [ list(i.values()) for i in data.dummies(catg_cols) ]
    #     X =[ i+j for i,j in zip(X, X_dummy) ]

    # # 2. convert any string values to float
    # X_int =  [[float(i) for i in elist] for elist in X ]
    # print('no of fields with None value:', sum([i.count(None) for i in X_int]))
    # X_int.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    
    # # 2. remove asymptomatic data
    # X_int = [i for i in X_int if sum(i[:5])>0]

    # # 3. load best model info
    # # MODEL = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_predictor\model')
    # with open(Path.joinpath(MODEL, "best_model.pkl"), 'rb') as f: 
    #     best_model = pickle.load(f)
    # print(best_model['clf'])
    # with open(Path.joinpath(MODEL, "best_model_prob.pkl"), 'rb') as f:
    #     prob = pickle.load(f) 
    # with open(Path.joinpath(MODEL, "best_model_score.pkl"), 'rb') as f:
    #     score = pickle.load(f) 
    
    # # 3. pass X to predict y
    # y = best_model.predict_proba( X_int )[:,1]
    # y_percentile = []
    # for i in y: 
    #     y_percentile.append( np.round( stats.percentileofscore(prob[:, 1], i), 1 ))

    # result = pd.DataFrame(X_int, columns=cols)
    # result['prediction_probability'] = y
    # result['prediction_percentile'] = y_percentile
    
    # filepath = Path.joinpath(DATA, 'predictions', 'predictions.csv')
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # result.to_csv(filepath, index=False)
    # print('model predictions saved to', filepath)
