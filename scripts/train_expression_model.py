"""Train facial expression model from scratch.
Dataset layout:
  data/expressions/
    train/{angry,happy,sad,surprised}/*.jpg
    val/{angry,happy,sad,surprised}/*.jpg
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


IMG_SIZE = (48, 48)
BATCH_SIZE = 64


def build_model(num_classes: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(48, 48, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    base = Path("data/expressions")

    train = tf.keras.utils.image_dataset_from_directory(
        base / "train",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val = tf.keras.utils.image_dataset_from_directory(
        base / "val",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    num_classes = len(train.class_names)
    norm = layers.Rescaling(1.0 / 255)
    train = train.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    val = val.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)

    model = build_model(num_classes)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("models/expression_cnn.keras", monitor="val_loss", save_best_only=True),
    ]
    model.fit(train, validation_data=val, epochs=40, callbacks=callbacks)


if __name__ == "__main__":
    main()
