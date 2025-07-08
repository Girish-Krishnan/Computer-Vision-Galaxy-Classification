import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .model_utils import load_feature_extractor
from .data_utils import prepare_dataframe


def extract_features(model_path: str, csv_path: str, image_dir: str,
                     batch_size: int = 32, output_path: str = "features.npy") -> np.ndarray:
    df = prepare_dataframe(csv_path)
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col="filename",
        y_col=None,
        batch_size=batch_size,
        shuffle=False,
        class_mode=None,
        target_size=(299, 299),
    )

    feature_extractor = load_feature_extractor(model_path)
    features = feature_extractor.predict(generator, steps=np.ceil(len(df)/batch_size), verbose=1)
    np.save(output_path, features)
    return features
