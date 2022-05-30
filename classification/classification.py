################################################################################
# Author : (c) Niti Mishra
# Date   : 2022.05.26
################################################################################

import warnings
warnings.filterwarnings('ignore') 

from pathlib import Path
import os
import ast
import json
from scipy import stats
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from build import binary_models
from build import score_models

from cluster import Best_Cluster_KMeans

# ROOT must be the main directory where a 'data' folder and 'info' folder exists 
# with respective files, so DATA and INFO variable need not be changed
ROOT = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests')
APP_RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_predictor')
DATA = Path.joinpath(ROOT, 'data')
INFO = Path.joinpath(ROOT, 'info') 

################################################################################
# 1. read info file that specifies the column names to select from data and
#    store these values in variables for later use
################################################################################

# read
f = open(Path.joinpath(INFO, 'data_info.txt'),'r')
contents = f.read()
data_info = ast.literal_eval(contents)
f.close() 
 
# store
cat_features = data_info['cat_features']
num_features = data_info['num_features']
target = data_info['target']
validation = data_info['validation']
filter_col = data_info['filter_feature']
cluster_features = data_info['cluster_features'] if type(data_info['cluster_features'])==list else list(data_info['cluster_features'])

################################################################################
# 2. read data as specified in the info file
################################################################################

col_list = cat_features + num_features + [validation] + [target] + [filter_col] + cluster_features    
for filename in os.listdir(DATA):
    if filename.endswith('.csv'):
        fullpath = Path.joinpath(DATA,filename)
        X = pd.read_csv(fullpath, encoding="utf-8", usecols=col_list)
print(f'\nread following columns specified in info file:\n{col_list}\n')

################################################################################
# 3. transform categorical columns
################################################################################

col_means = {}
X[cat_features] = X[cat_features].apply(pd.to_numeric, downcast="float")  # convert to numeric
for col in X[cat_features].columns[X[cat_features].isna().any()]:         # if any value is missing
    print(f'processing categorical transformation of: {col}')                   
    X[col+'_1'] = X[col].isna().astype(int)                               # create new dummy column where value is 1 if missing in original column
    col_means[col]= X[col].mean()
    X[col].fillna(col_means[col], inplace=True)                           # fill missing value with mean in original column
RESULTS_MAIN = Path.joinpath(ROOT, 'results')
RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
with open(Path.joinpath(RESULTS_MAIN, 'column_means.pkl'), 'wb') as f: 
    pickle.dump(col_means, f)                                             # save column means for use in prediction.py

################################################################################
# 4. save a dataset for each unique value in filter feature in a dictionary and
#    add the full dataset to this dictionary as well
#    value for each key should be tuple of three items:
#    (1) feature matrix, (2) target vector and (3) validation flag vector
################################################################################    

def data_splits(df):
    df.dropna(inplace=True)      
    target_vec = df[target]
    validation_flag = df[validation]
    feature_mat = df.drop([target]+[filter_col]+[validation], axis=1)
    return (feature_mat, target_vec, validation_flag) 

filters = [i for i in X[filter_col].unique()] + [None]                     # no. of filters in data including None filter
filtered_data = {f'{filter_col}_{str(i)}':None for i in filters }          # initialize dictionary to store filtered dataset
print(f'\nfollowing filters available:{filtered_data}')

for filter_val, key in zip(filters, filtered_data.keys()):
    print(filter_val, key)
    if filter_val is not None:                                             # if filter value is not None
        filtered = X[X[filter_col]==filter_val].copy(deep=True)            # apply the filter to the dataset and make a copy
    else:                                                             
        filtered = X.copy(deep=True)                                       # make a copy of whole data without any filter
    filtered_data[key] = data_splits(filtered)                             # add as value to this key:
                                                                           #      (1) feature matrix, (2) target vector and (3) validation flag as tuple

################################################################################
# 5. obtain index for test rows for unfiltered data set and 
#    each filtered data set
################################################################################   

all_idx = filtered_data[f'{filter_col}_{str(None)}'][-1]
all_test_idx = all_idx[all_idx=='Test'].index                              # index for all test rows in the data

filtered_set_idx = [v[0].index for k,v in filtered_data.items() if "None" not in k]
filtered_test_idx = [list(set(i).intersection(set(all_test_idx))) for i in filtered_set_idx] # index for test rows of each filtered data
print([len(i) for i in filtered_test_idx])

################################################################################
# 6. train the model for each data set in filtered_data
#    if data is unfiltered (meaning filter None) provide index for filter such  
#    that scores can be calculated on test set of each filtered set and 
#    not on the entire test
#    REFER TO build.py for model training pipeline
################################################################################   

filtered_model_scores = []
non_filtered_model_scores = []
filter_order = [k for k in filtered_data.keys()]
roc_scores = {}
importances = {}
for key,val in filtered_data.items():
    print(key)
    # create folder to store results for each filtered data set
    RESULTS = Path.joinpath(ROOT, 'results',key)
    RESULTS.mkdir(parents=True, exist_ok=True)
    
    # if working in entire dataset without filter
    if "None" in key:
        # each score on test set with each filter applied
        # i.e. if there are two filters to be applied, scores on two dataset should be calculated
        for scores in score_models(binary_models, val[0], val[1], split_idx=val[2], std_cols=num_features, filter_idx=filtered_test_idx , outpath=RESULTS):
            for score, test_filter in zip(scores,filter_order[:-1]):
                name = score['name']
                # save data for auc plot
                roc_data = pd.DataFrame([score['fpr'], score['tpr']], index=['fpr','tpr']).T
                roc_data.to_csv(Path.joinpath(RESULTS, f'ROC__{name}__{test_filter}.csv'), index=None)
                roc_scores[f'{name}__{test_filter}'] = roc_data
                importances[f'{name}__{test_filter}'] = score['importances']
                del score['importances']
                del score['fpr']
                del score['tpr']
                # # save remaining items in scores as json file
                # with open(Path.joinpath(RESULTS, f'scores__train{key}__test{test_filter}.json'), 'a') as f:
                #     f.write(json.dumps(score) + '\n')
            non_filtered_model_scores.append(scores)
    
    # if working with any of the filtered data set
    else:
        # each score for test set with corresponding filter only
        for scores in score_models(binary_models, val[0], val[1], split_idx=val[2], std_cols=num_features, outpath=RESULTS):
            # save data for auc plot
            name = scores['name']
            fpr, tpr = scores['fpr'], scores['tpr']
            roc_data = pd.concat([pd.Series(fpr), pd.Series(tpr)], axis=1) 
            roc_data.columns = ['fpr', 'tpr']
            roc_data.to_csv(Path.joinpath(RESULTS, f'ROC__{key}__{name}.csv'), index=None)
            roc_scores[f'{key}__{name}'] = roc_data
            importances[f'{key}__{name}'] = scores['importances']
            del scores['importances']
            del scores['fpr']
            del scores['tpr']
            # # save remaining items in scores as json file
            # with open(Path.joinpath(RESULTS, f'scores__{key}.json'), 'a') as f: 
            #     f.write(json.dumps(scores) + '\n')
            filtered_model_scores.append(scores)

################################################################################
# 7. create ROC Curve for each filtered set 
################################################################################  

for filtername in filter_order[:-1]:
    print(filtername)
    fig, ax = plt.subplots()
    for k,v in roc_scores.items():
        if filtername in k:
            plt.plot(v['fpr'], v['tpr'], label=k.replace(filtername,'').replace('_',''))            
    plt.title(f"ROC Curves \non test data with filter: {'='.join(filtername.split('_'))}")
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(Path.joinpath(RESULTS_MAIN, f'ROC_Curve__teston{filtername}.png')) 

for filtername in filter_order[:-1]:
    print(filtername)
    fig, ax = plt.subplots()
    for k,v in importances.items():
        if filtername in k:
            plt.plot([x for x in range(len(v))], v, label=k.replace(filtername,'').replace('_',''))            
    plt.title(f"Permutation feature importance \non test data with filter: {'='.join(filtername.split('_'))}")
    plt.legend(loc="upper left")
    # plt.show()
    plt.savefig(Path.joinpath(RESULTS_MAIN, f'feature_imp__teston{filtername}.png')) 

################################################################################
# 8. find the best model for each filtered set based on AUC and 
#    save as best model for that filter
################################################################################  

n_models = len(binary_models)
n_filters = len(filter_order[:-1])

#    for models trained on filtered data split scores for each filter 
filtered_model_scores = [list(i) for i in np.array_split(filtered_model_scores, n_filters)]

#    combine scores from all models (trained on filtered and unfiltered data) for each filter
for eachscore in non_filtered_model_scores:
    for eachmodel, m in zip(filtered_model_scores, range(n_models)):
        print(eachmodel)
        eachmodel.append(eachscore[m])

#    sort model scores based on AUC
filtered_model_scores = [sorted(i, key=lambda x:x['AUC'], reverse=True) for i in filtered_model_scores]

#    add rank key to each score dictionary and save scores as csv file for each filter
for eachfilter,filterid in zip(filtered_model_scores, filter_order[:-1]):
    for rank, eachmodel in enumerate(eachfilter):
        eachmodel['rank']=rank+1
    print(eachfilter)
    df = pd.DataFrame.from_dict(eachfilter)
    df = df[['rank', 'name',  'AUC', 'sensitivity', 'specificity', 'accuracy', 
            'precision', 'recall', 'f1_test', 'model', 'size', 'best_param', 'time']]
    df.to_csv(Path.joinpath(RESULTS_MAIN, f"results__teston{filterid}.csv"), index=False)

#    save the model with best performance on each filtered test data as best model for that data
#    save also in app folder
APP_RESULTS = Path(APP_RESULTS, 'model')
APP_RESULTS.mkdir(parents=True, exist_ok=True)
print(f"Model Results will be saved to {APP_RESULTS}")

best_models = {}
for idx, filterid in enumerate(filter_order[:-1]):

    # find the name of best model and the path where it is saved
    best_model_name = filtered_model_scores[idx][0]['name']
    if "trainedall" in best_model_name:
        best_model_name = best_model_name.replace('trainedall','',1)
        best_model_path = Path.joinpath(RESULTS_MAIN, filter_order[-1]) 
    else:
        best_model_path = Path.joinpath(RESULTS_MAIN, filterid)
    print(f'Best model for {filterid} is {best_model_name} in {best_model_path}')

    # load the best model and its scaler
    with open(Path.joinpath(best_model_path, best_model_name+".pkl"), 'rb') as f: 
        best_model = pickle.load(f)
    with open(Path.joinpath(best_model_path, "scaler.pkl"), 'rb') as f: 
        scaler_best_model = pickle.load(f)
    
    # add best model to best_models dictionary
    best_models[filterid] = (scaler_best_model, best_model)
    
    # save best model and its scaler in the results folder as best model
    modelfilename = f"bestmodel__{filterid}__{best_model_name}.pkl"
    scalerfilename = f"bestmodel_scaler_{filterid}_{best_model_name}.pkl"
    with open(Path.joinpath(RESULTS_MAIN, modelfilename), 'wb') as f:
        pickle.dump(best_model, f)
    with open(Path.joinpath(RESULTS_MAIN, scalerfilename), 'wb') as f:
        pickle.dump(best_model, f)
    # save best model and its scaler in the app folder as best model
    with open(Path.joinpath(APP_RESULTS, modelfilename), 'wb') as f:
        pickle.dump(best_model, f)
    with open(Path.joinpath(APP_RESULTS, scalerfilename), 'wb') as f:
        pickle.dump(best_model, f)


################################################################################
# 9. get model prediction and probabilities for each filtered test set  
#    from model with highest auc in that filtered test set and 
#    combine the test set
################################################################################  

test_nofilter = X[X[validation] == "Test"].copy(deep=True)
test_nofilter.dropna(inplace=True)  
featmat_divided = [test_nofilter.loc[idx] for idx in filtered_test_idx]

bestmodel_predprob = []
for idx,filterid in enumerate(filter_order[:-1]):
    print(idx, filterid)
    scaler, best_model = best_models[filterid]
    featmat = featmat_divided[idx].drop([target]+[filter_col]+[validation], axis=1)
    featmat[num_features] = scaler.transform(featmat[num_features])
    pred, prob = best_model.predict(featmat), best_model.predict_proba(featmat)[:,1]
    predprob = pd.DataFrame({'y_pred':pred,'y_prob':prob}, index=featmat.index)
    featmat_divided[idx] = pd.concat([featmat_divided[idx],predprob], axis=1)

test_nofilter = pd.concat(featmat_divided, axis=0)
print(f"\nmodel probabilities based on best performing model obtained for each filtered set and combined\n")

################################################################################
# 10. scale numeric features   
#     perform clustering analysis on cluster features only
#     REFER TO cluster.py for model training pipeline
################################################################################  

X_cluster = X.copy(deep=True)
#    scale numeric features
mean_val = X_cluster[num_features].mean()
max_val = X_cluster[num_features].max()
min_val = X_cluster[num_features].min()
X_cluster[num_features] = (X_cluster[num_features] - mean_val)/(max_val - min_val)

#    separate data to cluster on
X_cluster = X_cluster[cluster_features]
X_cluster_train = X_cluster.dropna(axis=0)

#    train mutiple KMeans and store their results
best_kmeans = Best_Cluster_KMeans(X=X_cluster_train, max_k=21, silhoutte_nsample=(30000,5), ballhall=False, result_path=RESULTS_MAIN)
best_k, best_kmeans_model = best_kmeans.best_k()

#    save labels on unfiltered test set (nans removed) from best k model  
X_cluster_test = X_cluster.iloc[test_nofilter.index]
test_nofilter['cluster'] = best_kmeans_model.predict(X_cluster_test)
np.savetxt(Path.joinpath(RESULTS_MAIN,'clusterlabels_k'+str(best_k)+'.csv'), test_nofilter['cluster'].astype(int), delimiter=',')
print(f"Cluster labels for a test set written out to {RESULTS_MAIN}\n")

#    save best k cluster model in results and app folder  
with open(Path.joinpath(RESULTS_MAIN, "best_K_model_"+str(best_k)+".pkl"), 'wb') as f:
    pickle.dump(best_kmeans_model, f)
with open(Path.joinpath(APP_RESULTS, "best_K_model_"+str(best_k)+".pkl"), 'wb') as f:
    pickle.dump(best_kmeans_model, f)

################################################################################
# 11. get probability percentile grouped by cluster from best model   
################################################################################  

cluster_percentiles = pd.DataFrame()
for k in test_nofilter['cluster'].unique():
    print(f"obtaining percentile for cluster {k}")
    subset = test_nofilter[test_nofilter['cluster']==k].copy(deep=True)
    prcntl = [stats.percentileofscore(subset['y_prob'], a) for a in subset['y_prob']]
    prcntl = pd.Series(prcntl) 
    prcntl.index=subset.index
    cluster_percentiles = pd.concat([cluster_percentiles,prcntl], axis=0)
test_nofilter['percentile'] = cluster_percentiles

#    save entire test data with prediction probability and cluster percentile as csv 
test_nofilter.to_csv(Path.joinpath(RESULTS_MAIN,'test_prob_prcntl.csv'), index=False)
print('\nmodel predictions probabilities and percentile based on clusters for test data saved\n')

