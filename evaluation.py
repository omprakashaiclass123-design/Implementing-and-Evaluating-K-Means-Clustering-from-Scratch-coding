import numpy as np

def inertia(X, labels, centroids):
    X = np.asarray(X)
    s = 0.0
    for i, c in enumerate(centroids):
        pts = X[labels == i]
        if pts.size == 0:
            continue
        s += np.sum(np.linalg.norm(pts - c, axis=1)**2)
    return s

def silhouette_score(X, labels):
    X = np.asarray(X)
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        return 0.0
    # precompute distance matrix
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    sil = []
    for i in range(n):
        own = labels[i]
        in_cluster = labels == own
        in_cluster[i] = False
        a = np.mean(dist[i][in_cluster]) if np.any(in_cluster) else 0.0
        b = np.inf
        for other in unique_labels:
            if other == own:
                continue
            other_mask = labels == other
            if not np.any(other_mask):
                continue
            b = min(b, np.mean(dist[i][other_mask]))
        if max(a, b) == 0:
            sil.append(0.0)
        else:
            sil.append((b - a) / max(a, b))
    return float(np.mean(sil))
