#!/usr/bin/env python3
"""Cluster extracted features and visualize with t-SNE."""
import argparse
import numpy as np
from galaxy.clustering import run_clustering, run_tsne, plot_tsne


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster image features")
    parser.add_argument("--features", default="features.npy", help="Path to features .npy file")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters")
    parser.add_argument("--labels-out", default="cluster_labels.npy", help="Output .npy for labels")
    parser.add_argument("--plot", default=None, help="Optional path for t-SNE plot image")
    return parser.parse_args()


def main():
    args = parse_args()
    features = np.load(args.features)
    labels = run_clustering(features, n_clusters=args.clusters)
    np.save(args.labels_out, labels)
    tsne_res = run_tsne(features)
    plot_tsne(tsne_res, labels, output=args.plot)


if __name__ == "__main__":
    main()
