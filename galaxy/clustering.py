import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def run_clustering(features: np.ndarray, n_clusters: int = 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels


def run_tsne(features: np.ndarray):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(features)


def plot_tsne(tsne_results: np.ndarray, labels: np.ndarray, output: str = None):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels,
                          cmap="viridis", alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE projection of the galaxy features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if output:
        plt.savefig(output)
    else:
        plt.show()
