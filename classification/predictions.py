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
    vars = ['cough', 'fever', 'sorethroat', 'shortnessofbreath', 
            'headache', 'contact', 'abroad', 
            'sixtiesplus', 'gender', 'sixtiesplus_1', 'gender_1']
    x1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]	
    x2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]	
    x3 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]	
    x4 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]	
    x5 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]	
    x6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]	
    x7 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]	
    x8 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]	
    x9 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]	
    x10 = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1]	
    x11 = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]	
    x12 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]	
    x13 = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]	
    x14 = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]	
    x15 = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]	
    x16 = [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]	
    x17 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]	
    x18 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]	

    vals = ['x'+str(i) for i in range(1,19)]
    record = [{vars[i]: x1[i] for i in range(len(vars))} , 
              {vars[i]: x2[i] for i in range(len(vars))} ,
              {vars[i]: x3[i] for i in range(len(vars))} ,
              {vars[i]: x4[i] for i in range(len(vars))} ,
              {vars[i]: x5[i] for i in range(len(vars))} ,
              {vars[i]: x6[i] for i in range(len(vars))} ,
              {vars[i]: x7[i] for i in range(len(vars))} ,
              {vars[i]: x8[i] for i in range(len(vars))} ,
              {vars[i]: x9[i] for i in range(len(vars))} ,
              {vars[i]: x10[i] for i in range(len(vars))} ,
              {vars[i]: x11[i] for i in range(len(vars))} ,
              {vars[i]: x12[i] for i in range(len(vars))} ,
              {vars[i]: x13[i] for i in range(len(vars))} ,
              {vars[i]: x14[i] for i in range(len(vars))} ,
              {vars[i]: x15[i] for i in range(len(vars))} ,
              {vars[i]: x16[i] for i in range(len(vars))} ,
              {vars[i]: x17[i] for i in range(len(vars))} ,
              {vars[i]: x18[i] for i in range(len(vars))}]
    
    # obtain the keys of the data dictionary
    # remove spaces, special characters from keys and lower cases 
    keys = [re.sub('[^a-zA-Z0-9 \n\.]','',i).lower() for i in record[0]]
    non_match_vars = set(vars) - set(keys)
    if (len(non_match_vars) > 0) :
        print("The following variables are missing in the data:")
        print(non_match_vars)
        print("Provide the missing variable in data and re-run")

    X = [ [int(i[x]) for x in vars] for i in record]

    X = np.array(X)
    if len(X.shape) == 1:
        X = X.reshape(1,-1)
    
    # load the best model
    RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\results')
    # RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_predictor\model')
    with open(Path.joinpath(RESULTS, "LogisticRegression.pkl"), 'rb') as f: 
        best_model = pickle.load(f)
    print(best_model)
    y_pred = best_model.predict( X )
    y_prob = best_model.predict_proba( X )[:,1]
    print(best_model.classes_)
    print(y_pred)
    for i in y_prob:
        print(i)
    
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
