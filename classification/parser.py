###################################################################
# To run this script in command line provide following arguments:
#   1. path to project folder (MUST have the two directories below)
#   2. name of directory that has the data files (csv file only)
#   3. name of directory where results should be stored

# The script loads data from data directory, selects following columns -
#  'Validation', 'testresult', 
#  'cough', 'fever', 'sore_throat', 'shortness_of_breath', 
#  'head_ache', 'sixtiesplus', 'Gender', 'contact', 'abroad'
# and processes them through loader.py to obtain train, test and 
# validation splits as specified in 'Validation' column. Finally,
# it runs multiple models using build.py and saves the scores of 
# each of these models in results directory in file "results.json"

# Example run:
# python classification.py C:\Users\XYZ\covid_tests data results
# ###################################################################

from pathlib import Path
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    Creates path profiles from provided path, data directory and results directory
    """)
    parser.add_argument("root_dir", help="Main directory of the project")
    # parser.add_argument("data_dir", help="Directory where the data is stored")
    # parser.add_argument("results_dir", help="Directory to store results")
    parser.add_argument("app_dir", help="Directory to store results")

    args = parser.parse_args()

    ROOT = Path(args.root_dir)
    DATA = Path.joinpath(ROOT, 'data')
    RESULTS = Path.joinpath(ROOT, 'results')
    APP_RESULTS = Path(args.app_dir, 'model')

    print("Main : " , ROOT)
    print("Data : " , DATA)
    print("Results : " , RESULTS)
    print("App Results : " , APP_RESULTS)