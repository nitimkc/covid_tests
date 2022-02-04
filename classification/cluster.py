import faiss
import numpy as np


class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]


# X_cluster_test = np.ascontiguousarray(X_cluster)


# # using faiss lib
# X_cluster = np.ascontiguousarray(X_cluster)
# kmeans_clusters = []
# kmeans_inertia = []
# silhouette = []
# krange = range(2, 21)
# for i in krange:
#     kmeans = FaissKMeans(n_clusters=i)
#     kmeans.fit(X_cluster)
#     print(f"Fitting KMeans with {i} cluster")
#     cluster_labels = kmeans.predict(X_cluster)
#     # sscore = round(silhouette_score(X, np.ravel(cluster_labels, order="c")),2)
#     kmeans_clusters.append(cluster_labels)
#     kmeans_inertia.append(kmeans.inertia_)
#     # silhouette.append(sscore)
# best_k = kmeans_inertia.index(min(kmeans_inertia)) 
# best_k_labels = kmeans_clusters[best_k]
