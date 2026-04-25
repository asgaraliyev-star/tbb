"""Train eye state model (open vs closed) from scratch.
Dataset layout:
  data/eye_state/
    train/open/*.jpg
    train/closed/*.jpg
    val/open/*.jpg
    val/closed/*.jpg
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


IMG_SIZE = (24, 24)
BATCH_SIZE = 64


def build_model() -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(24, 24, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC()])
    return model


def main() -> None:
    base = Path("data/eye_state")
    train = tf.keras.utils.image_dataset_from_directory(
        base / "train",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    val = tf.keras.utils.image_dataset_from_directory(
        base / "val",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    norm = layers.Rescaling(1.0 / 255)
    train = train.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    val = val.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)

    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("models/eye_state_cnn.keras", monitor="val_loss", save_best_only=True),
    ]
    model.fit(train, validation_data=val, epochs=25, callbacks=callbacks)


if __name__ == "__main__":
    main()
