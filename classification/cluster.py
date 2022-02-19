#########################################################################
# Part 8 perform clustering analysis on features defined in cluster
#        features in the data_info file
#########################################################################

class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
    self.n_clusters = n_clusters
    self.n_init = n_init
    self.max_iter = max_iter
    self.kmeans = None
    self.cluster_centers_ = None
    self.inertia_ = None
    
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

