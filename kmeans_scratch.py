import numpy as np

class KMeansScratch:
    def __init__(self, k=3, max_iters=300, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.k > n_samples:
            raise ValueError("k cannot be greater than number of samples")
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idx].astype(float)

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X, labels)

            # If any cluster lost all points, reinitialize that centroid
            for i, c in enumerate(new_centroids):
                if np.isnan(c).any():
                    new_centroids[i] = X[np.random.choice(n_samples)]

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                self.centroids = new_centroids
                break
            self.centroids = new_centroids

        return self

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else np.array([np.nan]*X.shape[1]) for i in range(self.k)])

    def predict(self, X):
        X = np.asarray(X)
        return self._assign_clusters(X)
