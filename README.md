# Galaxy Classification

This project contains a small pipeline for training a galaxy morphology classifier and performing basic feature analysis using Keras and scikit-learn. The original coursework code has been refactored into modular scripts with simple command line interfaces.

## Requirements

* Python 3.8+
* TensorFlow 2.x and its dependencies
* scikit-learn
* pandas, matplotlib, numpy

Create a virtual environment and install the packages of your choice. GPU support is recommended but not required.

## Dataset

Download the Galaxy Zoo dataset and place the files as follows:

```
data/
├── images_training_rev1/      # directory of JPG images
└── training_solutions_rev1.csv
```

The CSV should contain a `GalaxyID` column followed by the 37 label columns. The scripts will automatically create a `filename` column by appending `.jpg` to `GalaxyID`.

## Training

Run the training script to create a classifier. The model is based on Xception and is saved in Keras format.

```bash
python train_model.py --csv data/training_solutions_rev1.csv \
                      --images data/images_training_rev1/ \
                      --epochs 50 --batch-size 32 \
                      --model-out best_model.keras --plot-out training.png
```

This will produce `best_model.keras` and an optional training plot.

## Feature Extraction

After training, extract intermediate features for all images:

```bash
python extract_features.py --model best_model.keras \
                           --csv data/training_solutions_rev1.csv \
                           --images data/images_training_rev1/ \
                           --out features.npy
```

## Clustering and Visualization

Cluster the extracted features with k-means and visualize them using t-SNE:

```bash
python cluster_features.py --features features.npy \
                           --clusters 10 \
                           --labels-out cluster_labels.npy \
                           --plot tsne.png
```

This will save cluster labels and a t-SNE plot.

## Repository Layout

- `train_model.py` – CLI entry point for training the model.
- `extract_features.py` – CLI for extracting features from images using a trained model.
- `cluster_features.py` – CLI for clustering features and creating a t-SNE plot.
- `galaxy/` – package containing reusable code for data loading, model construction, and analysis.
- `unsupervised.ipynb` – original exploratory notebook retained for reference.

Feel free to extend these scripts or integrate them into your own workflows.
