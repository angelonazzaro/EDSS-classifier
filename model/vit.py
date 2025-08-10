import tensorflow as tf
from typing import Literal


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, mlp_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation="gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

        self(tf.random.uniform((1,) + (64, d_model)))  # noqa

    def call(self, x, training: bool = False):
        x_norm = self.layernorm1(x)

        attn_output, _ = self.mha(
            query=x_norm,
            value=x_norm,
            key=x_norm,
            return_attention_scores=True,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm, training=training)
        mlp_output = self.dropout2(mlp_output, training=training)

        return out1 + mlp_output

    def get_config(self):
        return {
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate,
        }


class VisionTransformer(tf.keras.Model):
    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 d_model: int,
                 mlp_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 n_layers: int = 6,
                 channels: int = 1,
                 data_format: Literal["channels_last", "channels_first"] = "channels_last",
                 task: Literal["binary", "multi-class"] = "binary",
                 n_classes: int = 1):
        super().__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"

        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        # Project flattened patches to d_model
        self.patch_proj = tf.keras.layers.Dense(d_model, trainable=True)

        # Class token + positional embedding
        self.cls_token = self.add_weight(
            shape=(1, 1, d_model),
            initializer="zeros",
            trainable=True,
            name="cls_token"
        )
        self.pos_embedding = self.add_weight(
            shape=(1, self.n_patches + 1, d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name="pos_embedding"
        )

        self.d_model = d_model
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.n_layers = n_layers
        self.channels = channels

        self.data_format = data_format
        self.n_classes = n_classes
        self.task = task

        self.enc_layers = [
            TransformerBlock(d_model, mlp_dim, num_heads, dropout)
            for _ in range(n_layers)
        ]

        self.dropout_layer = tf.keras.layers.Dropout(dropout)

        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(mlp_dim, activation="gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(n_classes, activation="sigmoid" if task == "binary" else "softmax"),
        ])

        # Build model with dummy forward pass for summary
        self(tf.random.uniform((1,) + (self.image_size, self.image_size, self.channels)))  # noqa

    # from https://dzlab.github.io/notebooks/tensorflow/vision/classification/2021/10/01/vision_transformer.html
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
        return patches

    def call(self, inputs, training: bool = False):
        batch_size = tf.shape(inputs)[0]

        # flatten patches to n_patches * (patch_size**2 * C)
        patches = self.extract_patches(inputs)
        x = self.patch_proj(patches)  # (B, n_patches, d_model)

        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.d_model])
        x = tf.concat([cls_tokens, x], axis=1)

        x = x + self.pos_embedding
        x = self.dropout_layer(x, training=training)

        for layer in self.enc_layers:
            x = layer(x, training=training) # noqa

        cls_output = x[:, 0]
        return self.mlp_head(cls_output, training=training)

    def get_config(self):
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "d_model": self.d_model,
            "mlp_dim": self.mlp_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "n_layers": self.n_layers,
            "data_format": self.data_format,
            "task": self.task,
            "n_classes": self.n_classes,
            "channels": self.channels
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


if __name__ == "__main__":
    model = VisionTransformer(224, 16, 768, 3072, 8, 0.3, 6)
    model.summary()

    x = tf.random.uniform((1, 224, 224, 1))

    model.predict(x)