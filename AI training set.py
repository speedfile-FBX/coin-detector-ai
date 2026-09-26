import os
import glob
import numpy as np
os.environ["KERAS_BACKEND"] = "torch"
import keras
dataset = keras.utils.image_dataset_from_directory(
    "coin_dataset",
    image_size=(530, 530),
    batch_size=16,
    validation_split=0.2,
    subset="both",
    seed=123
)

train_ds, validation_ds =dataset
print("CLASS NAMES:", train_ds.class_names)
model = keras.Sequential([
    keras.layers.Rescaling(1./255),

    keras.layers.RandomRotation(0.10),
    keras.layers.RandomZoom(0.3),

    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(256, 3, activation="relu"),
    keras.layers.MaxPooling2D(),

    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=30
)

model.save("coin_classifier.keras")
print("MODEL SAVED")