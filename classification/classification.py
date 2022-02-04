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
import time
import errno
import json
import pickle
import logging
import argparse
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

from build import binary_models
from build import score_models

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

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
    APP_RESULTS = Path(args.app_dir)

    print(f"Main : {ROOT}")
    print(f"Data : {DATA}")
    print(f"Additional info : {INFO}")

    # ROOT = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests')
    # DATA = Path.joinpath(ROOT, 'data')
    # INFO = Path.joinpath(ROOT, 'info')
    # RESULTS = Path.joinpath(ROOT, 'results')   
    # APP_RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_predictor')

    # 1. read file specifying column names to select from data
    info_files = [f for f in os.listdir(INFO) if f.endswith("data_info.txt")]
    data_info = {}
    if 'data_info.txt' not in info_files:
        print("file: 'data_info.txt' not in info folder\n")
    else:
        f = open(Path.joinpath(INFO, 'data_info.txt'),'r')
        contents = f.read()
        data_info = ast.literal_eval(contents)
        f.close() 

     # store items from info file   
    cat_features = data_info['cat_features']
    num_features = data_info['num_features']
    target = data_info['target']
    validation = data_info['validation']
    filter_col = data_info['filter_feature']
    cluster_features = data_info['cluster_features'] if type(data_info['cluster_features'])==list else list(data_info['cluster_features'])
    col_list = cat_features + num_features + [validation] + [target] + [filter_col] + cluster_features
    print(f'columns in the data info file:\n{col_list}\n')
    
    # store text for file naming
    filter_logic = data_info['filter_logic'] 
    filter_val = data_info['filter_value']
    if filter_logic=="==":
        filtername = 'equalto'
    if filter_logic==">=":
        filtername = 'moreandequalto'
    if filter_logic=="<=":
        filtername = 'lessthan'
    current_filter = filter_col + filter_logic + filter_val
    current_filtername = filter_col + filtername + filter_val

    # create directory if it does not exists already
    RESULTS = Path.joinpath(ROOT, 'results',current_filtername)
    APP_RESULTS = Path(APP_RESULTS, 'model',current_filtername)
    if not os.path.exists(RESULTS):
        os.makedirs(RESULTS)
        print(f"Directory {current_filtername} created to store Model Results in {RESULTS}")
    print(f"Model Results will be saved to {RESULTS}")
    if not os.path.exists(APP_RESULTS):
        os.makedirs(APP_RESULTS)
        print(f"Directory {current_filtername} created to store Model Results in {APP_RESULTS}")
    print(f"Model Results will be saved to {APP_RESULTS}")

    # 2. read data from features, target, validation columns
    for filename in os.listdir(DATA):
        if filename.endswith('.csv'):
            fullpath = Path.joinpath(DATA,filename)
            X = pd.read_csv(fullpath, encoding="utf-8", usecols=col_list)
    print(f'read all columns in info file')

    # 3. transform categorical columns
    col_means = {}
    X[cat_features] = X[cat_features].apply(pd.to_numeric, downcast="float")  # convert to numeric
    for col in X[cat_features].columns[X[cat_features].isna().any()]:         # if any value is missing
        print(f'processing categorical transformation of: {col}')                   
        X[col+'_1'] = X[col].isna().astype(int)                               # create new dummy column where value is 1 if missing in original column
        col_means[col]= X[col].mean()
        X[col].fillna(col_means[col], inplace=True)                           # fill missing value with mean in original column
    with open(Path.joinpath(RESULTS, 'column_means.pkl'), 'wb') as f: 
         pickle.dump(col_means, f)                                            # save column means for use in prediction.py    

    test_nofilter_idx = X[validation] == "Test"
    test_nofilter = [X[test_nofilter_idx], X.loc[test_nofilter_idx, target]]

    # 4. filter data as defined in data_info.txt file
    print(f'applying filter logic: {current_filter}')
    X.query(current_filter, inplace=True)

    # 5. remove rows with missing values in numeric features
    X.dropna(inplace=True)
    print(f"rows with missing values removed")

    # 6. prepare data for modelling and determine test, train and validation splits 
    data = X.copy(deep=True)
    X = X.drop([target]+[filter_col]+[validation], axis=1) 
    y = data[target]
    validation_idx = data[validation]
    print(f'final columns in the data used for training:\n{X.columns}')

    # 7. train models and save models and its scores    
    for scores in score_models(binary_models, X, y, split_idx=validation_idx, test_set=None, k=5, outpath=RESULTS):
        print(scores)
        with open(Path.joinpath(RESULTS, 'results'+current_filtername+'.json'), 'a') as f:
            f.write(json.dumps(scores) + '\n')
    

    
#########################################################################
# Part 8 perform clustering analysis on features defined in cluster
#        features in the data_info file
#########################################################################

    # test set without filter but nans removed
    test_nofilter_cluster = test_nofilter[0][cluster_features]
    test_nofilter_cluster.dropna(inplace=True)
    
    X_cluster = test_nofilter
    X_cluster = X[cluster_features]
    mbk_models = []
    clusters = []
    inertia = []
    silhouettes = []
    krange = range(2,21)
    for i in krange:
        mbk = MiniBatchKMeans(init="k-means++", n_clusters=i, n_init=10, max_no_improvement=10, verbose=0,)
        print(f"Fitting KMeans with {i} cluster")
        mbk.fit(X_cluster)
        labels = mbk.predict(X_cluster)
        sscore = silhouette_score(X_cluster, labels, sample_size=30000, n_jobs=-1)
        mbk_models.append(mbk)
        inertia.append(mbk.inertia_)
        clusters.append(labels)
        silhouettes.append(sscore)

    # plot
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(krange, silhouettes, marker='o')
    ax2.plot(krange, inertia, marker='o')
    plt.xlabel('Number of clusters')
    ax1.title.set_text('Silhoutte Score')
    ax2.title.set_text('Elbow method')
    fig.tight_layout()
    # plt.show()
    plt.savefig(Path.joinpath(RESULTS,'cluster_scores.png'))
    print(f"cluster inertia and sihoutte score plot saved")

    # save cluster info
    cluster_labels = pd.DataFrame(np.stack(clusters).T, columns=[str(i) for i in krange]) +1                     # cluster labels for models with all Ks
    cluster_means = pd.DataFrame(np.vstack([i.cluster_centers_ for i in mbk_models]), columns=X_cluster.columns) # cluster means for each clusters for models with all Ks
    k_idx = [item for sublist in [range(i) for i in krange]  for item in sublist]
    cluster_means['cluster_no'] = [item+1 for sublist in [range(i) for i in krange]  for item in sublist]
    cluster_means['k'] =  [item for sublist in [[i]*i for i in krange]  for item in sublist]

    cluster_labels.to_csv(Path.joinpath(RESULTS,'cluster_labels_allK.csv'))
    cluster_means.to_csv(Path.joinpath(RESULTS,'cluster_means_allK_'+current_filtername+'.csv'), index=False)
    with open(Path.joinpath(RESULTS,'cluster_info.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps({'cluster_k': [i for i in krange], 'inertia': inertia, 'silhoutte':silhouettes}))
    print(f"cluster labels, cluster centers, inertia and silhoutte scores saved to {RESULTS}")    
    
    # save cluster model for best k and its cluster labels on unfiltered test set (nans removed)
    best_k_idx = silhouettes.index(min(silhouettes)) 
    best_k = krange[best_k_idx]
    print(f"best number of clusters based on silhouttes score: {best_k}")
    
    best_k_model = mbk_models[best_k_idx]
    best_k_labels = best_k_model.predict(test_nofilter_cluster) +1

    with open(Path.joinpath(RESULTS, "best_K_model_"+str(best_k)+".pkl"), 'wb') as f:
        pickle.dump(best_k_model, f)
    with open(Path.joinpath(APP_RESULTS, "best_K_model_"+str(best_k)+".pkl"), 'wb') as f:
        pickle.dump(best_k_model, f)
    np.savetxt(Path.joinpath(RESULTS,'clusterlabels_k'+str(best_k)+'.csv'), best_k_labels.astype(int), delimiter=',')
    print(f"Cluster labels for a test set without {filter_col+filter_logic+filter_val} filter written out to {RESULTS}")

#########################################################################
# Part 9 among all models select the best one and save
#########################################################################
    
    # load all results
    all_scores = []
    with open(Path.joinpath(RESULTS, 'results'+current_filtername+'.json'), 'r') as f: 
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
    df.to_csv(Path.joinpath(RESULTS,'results'+current_filtername+'.csv'), index=False)
    
    # load the best model and test set
    best_model = all_scores[0]
    print('best_model is: ', best_model['name'])
    with open(Path.joinpath(RESULTS, (best_model['name']+current_filtername+'.pkl')), 'rb') as f: 
        model = pickle.load(f)
    with open(Path.joinpath(RESULTS, (best_model['name']+current_filtername+'_prob.pkl')), 'rb') as f:
        prob = pickle.load(f) 
    
    # save required info of best model in the heroku app folder

    with open(Path.joinpath(APP_RESULTS, "best_model_"+current_filtername+".pkl"), 'wb') as f:
        pickle.dump(model, f)
    with open(Path.joinpath(APP_RESULTS, "best_model_score_"+current_filtername+".pkl"), 'wb') as f:
        pickle.dump(best_model, f)
    with open(Path.joinpath(APP_RESULTS, "best_model_prob_"+current_filtername+".pkl"), 'wb') as f:
        pickle.dump(prob, f)
    with open(Path.joinpath(APP_RESULTS, "column_means_"+current_filtername+".pkl"), 'wb') as f: 
         pickle.dump(col_means, f)       
    with open(Path.joinpath(APP_RESULTS, "data_info_"+current_filtername+".pkl"), 'wb') as f: 
         pickle.dump(data_info, f)                                      


Z=np.random.rand(5,1)
c=np.random.rand(5,5)

n=len(Z)
avg=np.mean(Z,axis=0)
c_avg=c.mean(axis=1)

r=np.zeros(n)
for i in range(n):
    print(Z[i]-avg)
    r[i]=(Z[i]-avg)/c_avg[i]
    print(r)
# print(r)
# r.reshape(-1,1).shape

s = np.subtract(Z,Z.mean())
for i in s:
    print(i)
    s = i/c.mean(axis=1)
    # print(c.mean(axis=1))
    print(s)
print(s)
s.reshape(-1,1).shape