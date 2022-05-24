#########################################################################
# Part 8 perform clustering analysis on features defined in cluster
#        features in the data_info file
#########################################################################

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

class Cluster_KMeans(object):

    def __init__(self, X, n_clusters=5, n_init=10, max_iter=300, silhoutte_nsample=(30000,6), ballhall=False):
        self.X = X
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.n_sample = silhoutte_nsample[0] 
        self.ntimes = silhoutte_nsample[1] 
        self.ballhall = ballhall
    
    def fit_eval(self):
        mbk = MiniBatchKMeans(init="k-means++", n_clusters=self.n_clusters, n_init=self.n_init, max_iter=self.max_iter, 
                                max_no_improvement=10, verbose=0, batch_size=2048)
        print(f"Fitting KMeans with {self.n_clusters} cluster")
        mbk.fit(self.X)
        labels = mbk.labels_
        inertia = mbk.inertia_

        # calculate average silhoutte score on 5 sample of 30000 observation
        silhoutte_sample_scores = []
        print(f'length of random sample set: {self.n_sample}')
        for s in range(0,self.ntimes):
            print(f'calculating silhoutte for random sample set: {s+1}')
            idx = np.random.choice(range(0,self.X.shape[0]), self.n_sample, replace=False)
            X_sample, y_sample = self.X.iloc[idx], labels[idx]
            silhoutte_sample_scores.append(silhouette_score(X_sample, y_sample, n_jobs=-1))
        print(f'silhoutte score for each random sample {silhoutte_sample_scores}')
        silhoutte = np.mean(silhoutte_sample_scores)
        print(f'average silhoutte score from each random sample {silhoutte}')

        if self.ballhall:
            # np_bh = np.zeros(i)
            func_bh = np.zeros(self.n_clusters)
            for cluster in range(self.n_clusters):
                clusterdf = self.X[np.where(labels==cluster, True, False)]
                clusterdf = np.array(clusterdf)
                func_bh[cluster] = np.sum(clusterdf.var(axis=0))
                # np_bh[cluster] = np.sum(abs(clusterdf - clusterdf.mean(axis=0))**2)/clusterdf.shape[0] 
            # print(np_bh.mean())
            bhscore = func_bh.mean()
            return (mbk, labels, inertia, silhoutte, bhscore)
        else:
            return (mbk, labels, inertia, silhoutte)

# test1 = Cluster_KMeans(X=X_cluster_train, n_clusters=2, silhoutte_nsample=(300,5), ballhall=True)
# test1res = test1.fit_eval()

class Best_Cluster_KMeans(object):

    def __init__(self, X, max_k=3, n_init=10, max_iter=300, silhoutte_nsample=(30000,6), ballhall=False, result_path=None):
        self.X = X
        self.n_init = n_init
        self.max_iter = max_iter
        self.silhoutte_nsample = silhoutte_nsample
        self.ballhall = ballhall
        self.max_k = max_k
        self.result_path = result_path

    def best_k(self):            
        mbk_models, clusters, inertias, silhouettes, ballhalls = [], [], [], [], []
        krange = range(2, self.max_k+1)
        for k in krange:
            cluster_model = Cluster_KMeans(X=self.X, n_clusters=k, silhoutte_nsample=self.silhoutte_nsample, ballhall=self.ballhall)
            results = cluster_model.fit_eval()
            mbk_models.append(results[0])
            clusters.append(results[1])
            inertias.append(results[2])
            silhouettes.append(results[3])
            if self.ballhall:
                ballhalls.append(results[4])

        # find best k based on minimum silhouette score and the subsequent kmeans model
        best_k_idx = silhouettes.index(max(silhouettes)) 
        best_k = krange[best_k_idx]
        print(f"best number of clusters based on silhouttes score: {best_k}")
        best_k_model = mbk_models[best_k_idx]
        print(best_k_model)

        if self.result_path:
            # save plot of multiple KMeans
            if self.ballhall:
                fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
            else:
                fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            
            ax1.plot(krange, inertias, marker='o')
            ax2.plot(krange, silhouettes, marker='o')
            ax1.title.set_text('Elbow method')
            ax2.title.set_text('Silhoutte Score')
            if self.ballhall:
                ax3.plot(krange, ballhalls, marker='o')
                ax3.title.set_text('Ball-Hall Score')
            plt.xlabel('Number of clusters')
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            # plt.show()
            plt.savefig(Path.joinpath(self.result_path,'cluster_scores.png'))
            print(f"cluster inertia and sihoutte score plot saved")

            # save cluster scores for multiple KMeans
            clusters_info = pd.DataFrame.from_dict({
                'cluster_k': [i for i in krange], 
                'inertia': inertias, 
                'silhoutte':silhouettes,})
            if self.ballhall:
                clusters_info['ballhall']=ballhalls
            clusters_info.to_csv(Path.joinpath(self.result_path,'clusterinformation.csv'))
            # with open(Path.joinpath(self.result_path,'cluster_info.json'), 'w', encoding='utf-8') as f:
            #     f.write(json.dumps({
            #         'cluster_k': [i for i in krange], 
            #         'inertia': inertias, 
            #         'silhoutte':silhouettes, 
            #         'ballhall':ballhalls}))
            print(f"cluster labels, cluster centers, inertia, silhoutte and ball-hall scores saved to {self.result_path}") 
        
            # # save info of all clusters
            # cluster_labels = pd.DataFrame(np.stack(clusters).T, columns=[str(i) for i in krange]) +1                     # cluster labels for models with all Ks
            
            # cluster_means = pd.DataFrame(np.vstack([i.cluster_centers_ for i in mbk_models]), columns=self.X.columns) # cluster means for each clusters for models with all Ks
            # cluster_means['cluster_no'] = [item+1 for sublist in [range(i) for i in krange]  for item in sublist]
            # cluster_means['k'] =  [item for sublist in [[i]*i for i in krange]  for item in sublist]
            
            # cluster_labels.to_csv(Path.joinpath(RESULTS,'cluster_labels_allK.csv'))
            # cluster_means.to_csv(Path.joinpath(RESULTS,'cluster_means_allK_'+current_filtername+'.csv'), index=False)
        return (best_k , best_k_model)

# test= Best_Cluster_KMeans(X=X_cluster, max_k=5, silhoutte_nsample=(30,5), ballhall=True, result_path=RESULTS)
# testres = test.best_k()