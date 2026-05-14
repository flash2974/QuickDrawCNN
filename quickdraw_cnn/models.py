from keras.models import Sequential
from keras.layers import (
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dropout,
)


def model_list(num_classes=10, input_shape=(28, 28, 1)):

    return [
        Sequential(
            name="BasicCNN",
            layers=[
                Conv2D(32, 4, activation="relu", input_shape=input_shape),
                Conv2D(64, 3, activation="relu"),
                Conv2D(128, 3, activation="relu"),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ],
        ),
        Sequential(
            name="MaxPooling",
            layers=[
                Conv2D(
                    32, 4, activation="relu", padding="same", input_shape=input_shape
                ),
                MaxPooling2D(),
                Conv2D(64, 3, activation="relu", padding="same"),
                MaxPooling2D(),
                Conv2D(128, 3, activation="relu", padding="same"),
                MaxPooling2D(),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ],
        ),
        Sequential(
            name="AVGPooling_Dropout",
            layers=[
                Conv2D(
                    32, 3, activation="relu", padding="same", input_shape=input_shape
                ),
                MaxPooling2D(),
                Conv2D(64, 3, activation="relu", padding="same"),
                MaxPooling2D(),
                Conv2D(128, 3, activation="relu", padding="same"),
                MaxPooling2D(),
                GlobalAveragePooling2D(),
                Dense(128, activation="relu"),
                Dropout(0.3),
                Dense(num_classes, activation="softmax"),
            ],
        ),
        Sequential(
            name="BasicMLP",
            layers=[
                Flatten(input_shape=(28, 28, 1)),
                Dense(256, activation="relu"),
                Dense(128, activation="relu"),
                Dense(10, activation="softmax"),
            ],
        ),
    ]
