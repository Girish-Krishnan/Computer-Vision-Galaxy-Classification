import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    """Load labels CSV and add a filename column."""
    df = pd.read_csv(csv_path)
    df["filename"] = df["GalaxyID"].astype(str) + ".jpg"
    return df


def create_generators(df: pd.DataFrame, image_dir: str, batch_size: int = 32,
                      val_split: float = 0.2):
    """Create training and validation generators."""
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)

    label_cols = df.columns.tolist()[1:-1] if "filename" in df.columns else df.columns.tolist()[1:]

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="filename",
        y_col=label_cols,
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(299, 299),
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=image_dir,
        x_col="filename",
        y_col=label_cols,
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(299, 299),
    )

    return train_gen, val_gen
