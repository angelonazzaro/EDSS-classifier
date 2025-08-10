from typing import Literal, Tuple

import tensorflow as tf


class CNNModel(tf.keras.Model):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 units: int = 128,
                 n_conv_layers: int = 3,
                 n_dense_layers: int = 2,
                 dropout: float = 0.3,
                 data_format: Literal["channels_last", "channels_first"] = "channels_last",
                 task: Literal["binary", "multi-class"] = "binary",
                 n_classes: int = 1):
        super().__init__()

        if not (0 <= dropout <= 1):
            raise ValueError("`dropout` must be between 0 and 1")

        self.input_shape = input_shape
        self.task = task
        self.units = units
        self.n_conv_layers = n_conv_layers
        self.n_dense_layers = n_dense_layers
        self.data_format = data_format
        self.dropout = dropout
        self.n_classes = n_classes

        # Convolutional blocks - Feature Extractor
        self.conv_blocks = []

        for i in range(n_conv_layers):
            filters = 32 * (i + 1)
            block = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3),
                                       padding="same", activation="relu",
                                       data_format=data_format),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
            ])
            self.conv_blocks.append(block)

        # Dense blocks - Classifier
        self.flatten = tf.keras.layers.Flatten()
        self.dense_blocks = []
        for i in range(n_dense_layers):
            units = units // (2 ** i) if i > 0 else units
            dense_block = tf.keras.Sequential([
                tf.keras.layers.Dense(units, activation="relu"),
                tf.keras.layers.Dropout(self.dropout)
            ])
            self.dense_blocks.append(dense_block)

        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            n_classes,
            activation="sigmoid" if task == "binary" else "softmax"
        )

        if not isinstance(self.input_shape, Tuple):
            # fix 'TrackedList' conversion when loading from checkpoint
            self.input_shape = tuple(input_shape)

        # Build model with dummy forward pass for summary
        self(tf.random.uniform((1,) + self.input_shape))  # noqa

    def call(self, inputs, training: bool = False):
        x = inputs

        for i, block in enumerate(self.conv_blocks):
            x = block(x, training=training)

        x = self.flatten(x)

        for block in self.dense_blocks:
            x = block(x, training=training)

        return self.output_layer(x)

    def get_config(self):
        return {
            "input_shape": tuple(self.input_shape),
            "units": self.units,
            "n_conv_layers": self.n_conv_layers,
            "n_dense_layers": self.n_dense_layers,
            "dropout": self.dropout,
            "data_format": self.data_format,
            "task": self.task,
            "n_classes": self.n_classes
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    model = CNNModel((256, 256, 1), task="multi-class", n_classes=3,
                     n_conv_layers=4, n_dense_layers=3)
    model.summary()
    model.save("cnn_test.keras")

    loaded_model = tf.keras.models.load_model(
        "cnn_test.keras",
        custom_objects={"CNNModel": CNNModel}
    )
    loaded_model.summary()