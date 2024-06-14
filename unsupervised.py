import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tensorflow.keras import Model

# Load the trained model
model_path = 'best_model.keras'
model = load_model(model_path, compile=False)

# Modify the model to remove the final layer and create a feature extractor
base_model = model.layers[-2].output  # assuming the last layer is a Dense layer
feature_extractor = Model(inputs=model.input, outputs=base_model)

# Setup the data generator for feature extraction
data_path = 'data/images_training_rev1/'
df = pd.read_csv('data/training_solutions_rev1.csv')  # This should have the filenames
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=data_path,
    x_col='filename',  # column in df that contains the filenames
    y_col=None,  # No labels
    batch_size=batch_size,
    shuffle=False,
    class_mode=None,  # No labels
    target_size=(299, 299)  # Ensure this matches the model's expected input
)

# Extract features
num_images = df.shape[0]
features = feature_extractor.predict(generator, steps=np.ceil(num_images/batch_size), verbose=1)

# Save extracted features if needed
np.save('features.npy', features)

# Apply k-means clustering
k = 10  # number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(features)

# Save cluster labels if needed
np.save('cluster_labels.npy', cluster_labels)

# Dimensionality reduction with t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features)

# Plotting
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE projection of the galaxy features, colored by cluster')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()
