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
   
    # each new record must be converted into dictionary
    vals = ['0', '1', '0', '0', '0', '1', '1', '0', '0']
    record = [{vars[i]: vals[i] for i in range(len(vars))} , {vars[i]: vals[i] for i in range(len(vars))} ]

    # load the best model
    RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\covid_tests\covid_tests\results')
    with open(Path.joinpath(RESULTS, "best_model.pkl"), 'rb') as f: 
        best_model = pickle.load(f)
    vars = ['cough', 'fever', 'sorethroat', 'shortnessofbreath', 'headache', 'sixtiesplus', 'gender', 'contact', 'abroad']
    
    # obtain the keys of the data dictionary
    # remove spaces, special characters from keys and lower cases 
    keys = [re.sub('[^a-zA-Z0-9 \n\.]','',i).lower() for i in record[0]]
    non_match_vars = set(vars) - set(keys)
    if (len(non_match_vars) > 0) :
        print("The following variables are missing in the data:")
        print(non_match_vars)
        print("Provide the missing variable in data and re-run")

    X = [ [i[x] for x in vars] for i in record]

    X = np.array(X)
    if len(X.shape) == 1:
        X = X.reshape(1,-1)
    y_pred = best_model.predict( X )
    y_prob = best_model.predict_proba( X )[:,1]
    
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
