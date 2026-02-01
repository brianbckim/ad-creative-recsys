from typing import Sequence, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_copy_head(user_dim: int, theme_dim: int,
                    hidden_units: Sequence[int] = (128, 64),
                    dropout_rate: float = 0.2) -> keras.Model:
    u_in = keras.Input(shape=(user_dim,), name="user_emb")
    t_in = keras.Input(shape=(theme_dim,), name="theme_emb")
    c_in = keras.Input(shape=(theme_dim,), name="copy_emb")

    delta = layers.Subtract(name="delta")([c_in, t_in])
    x = layers.Concatenate(name="concat")([u_in, t_in, delta])

    h = x
    for i, units in enumerate(hidden_units):
        h = layers.Dense(units, activation="relu", name=f"dense_{i+1}")(h)
        if dropout_rate and dropout_rate > 0:
            h = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(h)

    logit = layers.Dense(1, activation=None, name="logit")(h)
    prob = layers.Activation("sigmoid", name="prob")(logit)

    model = keras.Model(inputs=[u_in, t_in, c_in], outputs=prob, name="copy_head")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model
