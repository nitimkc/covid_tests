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
    vars = ['cough', 'fever', 'sorethroat', 'shortnessofbreath', 'headache',
            'sixtiesplus', 'sixtiesplus_1',  'gender', 'gender_1',
            'contact', 'contact_1', 'abroad', 'abroad_1', ]
    X = [0,0,0,1,1,0,0,1,0,0,1,0,1]
    vals = [X]
    record = [dict(zip(vars, i)) for i in vals]
    
    # obtain the keys of the data dictionary
    # remove spaces, special characters from keys and lower cases 
    keys = [re.sub('[^_a-zA-Z0-9 \n\.]','',i).lower() for i in record[0]]
    non_match_vars = set(vars) - set(keys)
    if (len(non_match_vars) > 0) :
        print("The following variables are missing in the data:")
        print(non_match_vars)
        print("Provide the missing variable in data and re-run")

    # convert X to numpy array
    X = [ [int(i[x]) for x in vars] for i in record]
    X = np.array(X)
    if len(X.shape) == 1:
        X = X.reshape(1,-1)
    
    # load the best model
    RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\results')
    # RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_predictor\model')
    with open(Path.joinpath(RESULTS, "best_model.pkl"), 'rb') as f: 
        best_model = pickle.load(f)
    print(best_model)
    
    # make predictions on X
    y_pred = best_model.predict( X )
    y_prob = best_model.predict_proba( X )[:,1]
    print(best_model.classes_)
    print(y_pred)
    for i in y_prob:
        print(i)
    

    # save results as csv (both X and y)
    import copy
    result = copy.deepcopy(record)
    result = [{**item, 'prediction':i} for i,item in zip(y_pred, result)]
    result = [{**item, 'probability':i} for i,item in zip(y_prob, result)]
    
    with open(Path.joinpath(RESULTS, 'predictions.csv'), 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, result[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(result)
        # for key in result.keys():
        #     f.write("%s,%s\n"%(key,result[key]))
