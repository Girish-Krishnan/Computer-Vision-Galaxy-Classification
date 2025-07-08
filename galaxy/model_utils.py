from typing import Tuple
import matplotlib.pyplot as plt
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def custom_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))


def build_model(input_shape: Tuple[int, int, int] = (299, 299, 3)) -> Model:
    base_model = Xception(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(37, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model


def train_model(model: Model, train_gen, val_gen, epochs: int = 50,
                lr: float = 0.001, output_path: str = "best_model.keras"):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=root_mean_squared_error,
        metrics=["mse", custom_accuracy],
    )

    checkpoint = ModelCheckpoint(output_path, monitor="val_loss", verbose=1,
                                 save_best_only=True, mode="min")
    early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="min")

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n // train_gen.batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.n // val_gen.batch_size,
        callbacks=[checkpoint, early_stop],
    )
    return history


def plot_history(history, output: str = None):
    plt.plot(history.history["loss"], label="train RMSE")
    plt.plot(history.history["val_loss"], label="validation RMSE")
    plt.plot(history.history["custom_accuracy"], label="train accuracy")
    plt.plot(history.history["val_custom_accuracy"], label="validation accuracy")
    plt.title("Model Performance")
    plt.ylabel("Metric")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    if output:
        plt.savefig(output)
    else:
        plt.show()


def load_feature_extractor(model_path: str) -> Model:
    model = load_model(model_path, compile=False)
    base_output = model.layers[-2].output
    feature_extractor = Model(inputs=model.input, outputs=base_output)
    return feature_extractor
