from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, TimeDistributed, LSTM, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

np.random.seed(42)

os.makedirs("model", exist_ok=True)

model = Sequential([
    TimeDistributed(
        Conv2D(32, (3,3), activation='relu'),
        input_shape=(30, 224, 224, 3)
    ),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D(2,2)),

    TimeDistributed(
        Conv2D(64, (3,3), activation='relu')
    ),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D(2,2)),

    TimeDistributed(Flatten()),
    LSTM(32, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

X_train = np.random.rand(4, 30, 224, 224, 3)
y_train = np.array([0, 1, 0, 1])

model.fit(X_train, y_train, epochs=3)

model.save("model/deepfake_model.h5")

print("DeepFake model trained & saved successfully")

