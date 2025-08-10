from typing import Literal

import tensorflow as tf


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model: int, d_k: int, d_v: int, num_heads: int = 8):
        super().__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.num_heads = num_heads

        self.W_q = tf.keras.layers.Dense(num_heads * d_k, input_shape=(d_model,))
        self.W_k = tf.keras.layers.Dense(num_heads * d_k, input_shape=(d_model,))
        self.W_v = tf.keras.layers.Dense(num_heads * d_v, input_shape=(d_model,))

        self.W_o = tf.keras.layers.Dense(d_model, input_shape=(num_heads * d_k,))

    def split_heads(self, x, depth):
        # x: (batch_size, seq_len, num_heads * depth)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # reshape to (batch_size, seq_len, num_heads, depth)
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, depth))
        # transpose to (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention(self, query, key, value):
        # query/key/value shape: (batch_size, num_heads, seq_len, depth)
        scores = tf.matmul(query, key, transpose_b=True)
        scores = tf.divide(scores, tf.sqrt(tf.cast(self.d_k, tf.float32)))

        weights = tf.nn.softmax(scores, axis=-1)
        outputs = tf.matmul(weights, value)  # (batch_size, num_heads, seq_len, d_v)

        return outputs, weights

    def call(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size = tf.shape(x)[0]

        q = self.W_q(x)  # (batch_size, seq_len, num_heads * d_k)
        k = self.W_k(x)
        v = self.W_v(x)

        q = self.split_heads(q, self.d_k)
        k = self.split_heads(k, self.d_k)
        v = self.split_heads(v, self.d_v)

        attention_output, attention_weights = self.attention(q, k, v)

        # transpose and reshape back to (batch_size, seq_len, num_heads * d_v)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.num_heads * self.d_v))

        output = self.W_o(concat_attention)  # (batch_size, seq_len, d_model)

        return output, attention_weights

    def get_config(self):
        return {
            "num_heads": self.num_heads,
            "d_model": self.d_model,
            "d_k": self.d_k,
            "d_v": self.d_v,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_k: int, d_v: int, mlp_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.mha = MultiHeadSelfAttention(d_model, d_k, d_v, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(mlp_dim, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training: bool = False):
        x_norm = self.layernorm1(x)
        attn_output, _ = self.mha(x_norm)  # noqa
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm, training=training)
        mlp_output = self.dropout2(mlp_output, training=training)

        return out1 + mlp_output

    def get_config(self):
        return {
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class VisionTransformer(tf.keras.Model):
    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 d_model: int,
                 d_k: int,
                 d_v: int,
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

        # assuming only square images
        self.n_patches = (image_size // patch_size) ** 2

        self.patch_proj = tf.keras.layers.Dense(d_model, trainable=True)

        self.cls_token = self.add_weight(
            shape=(1, 1, d_model),
            initializer=tf.keras.initializers.Zeros(),
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
        self.d_k = d_k
        self.d_v = d_v
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.channels = channels

        self.data_format = data_format
        self.n_classes = n_classes
        self.task = task

        self.enc_layers = [
            TransformerBlock(d_model, d_k, d_v, mlp_dim, num_heads, dropout)
            for _ in range(n_layers)
        ]

        self.dropout_layer = tf.keras.layers.Dropout(dropout)

        # classifier
        self.mlp_head = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Dense(mlp_dim, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(n_classes, activation="sigmoid" if task == "binary" else "softmax"),
            ]
        )

        # Build model with dummy forward pass for summary
        self(tf.random.uniform((1,) + (self.image_size, self.image_size, 1)))  # noqa

    # from https://github.com/emla2805/vision-transformer
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_size])
        return patches

    def call(self, inputs, training: bool = False):
        batch_size = tf.shape(inputs)[0]
        # TODO: check patch extraction, shapes do not match with pos embeddings
        # flatten patches to n_patches * (patch_size**2 * C)
        patches = self.extract_patches(inputs)
        x = self.patch_proj(patches)  # (batch_size, d_model)

        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.d_model])
        x = tf.concat([cls_tokens, x], axis=1)  # (batch_size, num_patches + 1, d_model)

        x = x + self.pos_embedding

        x = self.dropout_layer(x, training=training)

        for layer in self.enc_layers:
            x = layer(x, training=training)  # noqa

        cls_output = x[:, 0]  # (batch_size, d_model)
        out = self.mlp_head(cls_output, training=training)

        return out

    def get_config(self):
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "d_model": self.d_model,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "mlp_dim": self.mlp_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
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
    model = VisionTransformer(256, 32, 768, 512, 512, 3072, 12)
    model.summary()

    x = tf.random.uniform((1, 256, 256, 1))

    model.predict(x)
