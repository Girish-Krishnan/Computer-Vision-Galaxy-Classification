import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    """Load labels CSV and add a filename column."""
    df = pd.read_csv(csv_path)
    df["filename"] = df["GalaxyID"].astype(str) + ".jpg"
    return df


def create_generators(
    df: pd.DataFrame,
    image_dir: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> tuple:
    """Create training, validation, and test generators."""

    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be less than 1.0")

    train_df, temp_df = train_test_split(
        df, test_size=val_split + test_split, random_state=42
    )
    val_size = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_size, random_state=42)

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
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

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

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col="filename",
        y_col=label_cols,
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="raw",
        target_size=(299, 299),
    )

    return train_gen, val_gen, test_gen
