import matplotlib.pyplot as plt
import numpy as np
from kmeans_scratch import KMeansScratch
from evaluation import inertia, silhouette_score

def plot_clusters(X, labels, centroids, title='Clusters'):
    plt.figure()
    plt.scatter(X[:,0], X[:,1])
    # draw centroids
    plt.scatter(centroids[:,0], centroids[:,1], marker='x')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def plot_elbow(X, k_range=range(1,11), random_state=0):
    inertias = []
    for k in k_range:
        model = KMeansScratch(k=k, random_state=random_state)
        model.fit(X)
        labels = model.predict(X)
        inertias.append(inertia(X, labels, model.centroids))
    plt.figure()
    plt.plot(list(k_range), inertias)
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve')
    plt.show()
    return list(k_range), inertias

def plot_silhouette_scores(X, k_range=range(2,11), random_state=0):
    scores = []
    for k in k_range:
        model = KMeansScratch(k=k, random_state=random_state)
        model.fit(X)
        labels = model.predict(X)
        scores.append(silhouette_score(X, labels))
    plt.figure()
    plt.plot(list(k_range), scores)
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores by k')
    plt.show()
    return list(k_range), scores
