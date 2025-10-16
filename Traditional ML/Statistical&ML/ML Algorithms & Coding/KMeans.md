```python
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_inters
        self.clusters = {}
    
    def distance(p1, p2):
    	return np.sqrt(np.sum((p1-p2)**2))
    
    def assign_clusters(X, clusters):
        for idx in range(X.shape[0]):
            dist = []
            x = X[idx]
            for i in range(k):
                dis = self.distance(x, clusters[i]['center'])
                dist.append(dis)
            new_cluster = np.argmin(dist)
            clusters[new_cluster]['points'].append(x)
        return clusters

    def update_clusters(X, clusters):
        for i in range(k):
            points = np.array(clusters[i]['points'])
            if points.shape[0] > 0:
                new_center = points.mean(axis=0)
                clusters[i]['center'] = new_center
                clusters[i]['points'] = []
        return clusters

    def fit(self, X: np.array):
        num_samples = X.shape[0]
        # Initialize clusters
        random_indices = np.random.permutation(num_samples)[:self.k]
        for i in range(len(random_indices)):
            self.cluster[i] = {
                'center' : random_indices[i],
        		'points' : []
            }
        
        for i in range(self.max_iters):
            assign_clusters(X, self.cluster)
            update_clusters(X, self.cluster)
    
    def pred_cluster(X, clusters):
        pred = []
        for i in range(X.shape[0]):
            dist = []
            for j in range(k):
                dist.append(distance(X[i], clusters[j]['center']))
            pred.append(np.argmin(dist))
        return pred
```

