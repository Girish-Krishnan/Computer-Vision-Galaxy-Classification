import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf

# Define RMSE and custom accuracy
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def custom_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

# Load data
df = pd.read_csv('data/training_solutions_rev1.csv')
image_folder = 'data/images_training_rev1/'
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df['GalaxyID'] = train_df['GalaxyID'].astype(str) + '.jpg'
val_df['GalaxyID'] = val_df['GalaxyID'].astype(str) + '.jpg'

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=180, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df, directory=image_folder, x_col="GalaxyID", y_col=df.columns.tolist()[1:], batch_size=32, seed=42, shuffle=True, class_mode="raw", target_size=(299,299))
validation_generator = val_datagen.flow_from_dataframe(dataframe=val_df, directory=image_folder, x_col="GalaxyID", y_col=df.columns.tolist()[1:], batch_size=32, seed=42, shuffle=True, class_mode="raw", target_size=(299,299))

# Model setup
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(37, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss=root_mean_squared_error, metrics=['mse', custom_accuracy])

# Callbacks
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

# Train the model
history = model.fit(train_generator, steps_per_epoch=train_generator.n//train_generator.batch_size, epochs=50, validation_data=validation_generator, validation_steps=validation_generator.n//validation_generator.batch_size, callbacks=[checkpoint, early_stopping])

# Plot RMSE and custom accuracy
plt.plot(history.history['loss'], label='train RMSE')
plt.plot(history.history['val_loss'], label='validation RMSE')
plt.plot(history.history['custom_accuracy'], label='train accuracy')
plt.plot(history.history['val_custom_accuracy'], label='validation accuracy')
plt.title('Model Performance')
plt.ylabel('Metric')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
