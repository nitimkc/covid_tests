###################################################################
# To run this script in command line provide following arguments:
#   1. -
#   2. -
#   3. -

# The script loads -----

# Example run:
# python classification.py C:\Users\XYZ\covid_tests 
# ###################################################################

from pathlib import Path
import os
import csv
import re
import errno
import json
import pickle
import logging
import argparse 
import numpy as np
from scipy import stats

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
   
    # each new record must be converted into dictionary
    input_vars = ['No','No','No','Yes','No','Below 60','Female','No','No',]
    # input_vars = ['Yes','Yes','Yes','Yes','No','Below 60','Female','Yes','Yes',]
    reg_vars = ['cough', 'fever', 'sorethroat', 'shortnessofbreath', 'headache']
    dummy_vars = ['sixtiesplus', 'gender', 'contact', 'abroad',]
    record = dict(zip(reg_vars+dummy_vars, input_vars))
        
    map_reg_vals = {'Yes':1, 'No':0}
    map_dummy_vals = {'Yes':[1,0], 'No':[0,0],'Unknown':[0,1],
                        'Below 60':[0,0], 'Above 60':[1,0], 
                        'Male':[0,0], 'Female':[1,0],}

    reg_X = [map_reg_vals.get(j,j)  for i,j in record.items() if i in reg_vars]
    dummy_X = [map_dummy_vals.get(j,j)  for i,j in record.items() if i in dummy_vars]
    X = reg_X + dummy_X
    
    X_all = []
    if sum(X[:5])==0:
        print("At least one symptom must be present")
    else:
        for i in X:
            if type(i)==list:
                X_all.extend(i)
            else:
                X_all.append(i)

        # prepare X for sklearn model
        X_int = np.array(X_all)
        if len(X_int.shape) == 1:
            X_int = X_int.reshape(1,-1)
        y_pred_prob = best_model.predict_proba(X_int)
        print(y_pred_prob[:, 1])
        
        # load the best model
        RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\results')
        with open(Path.joinpath(RESULTS, "LogisticRegression.pkl"), 'rb') as f:
        # RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_predictor\model')
        # with open(Path.joinpath(RESULTS, "best_model.pkl"), 'rb') as f: 
            best_model = pickle.load(f)
        print(best_model)
        
        # pass X to predict y
        X_int = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(1,-1)        # pass X to predict y
        y = best_model.predict_proba( X_int )[:,1]*100
        print(X_int, y)

        X_int = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]).reshape(1,-1)
        # pass X to predict y
        y = best_model.predict_proba( X_int )[:,1]*100
        print(X_int, y)
    

