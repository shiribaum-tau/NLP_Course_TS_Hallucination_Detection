import tensorflow as tf
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


plt.rc('font', size=16)
tfk = tf.keras
tfkl = tf.keras.layers


def generate_data(n_samples, n_features, n_timesteps):
    X = np.random.rand(n_samples, n_timesteps, n_features)
    y = np.random.randint(0, 2, n_samples)
    Y = np.eye(2)[y]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, Y_train, X_test, Y_test


class FeedForwardNN:
    def __init__(self, input_shape, seed=42):
        self.seed = seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.compat.v1.set_random_seed(seed)
        self.model = None
        self.build_ffnn(input_shape)

    def build_ffnn(self, input_shape):
        input_layer = tfkl.Input(shape=input_shape, name='input')

        hidden_layers = tfkl.Dense(units=4096, activation='relu', name='hidden_1')(input_layer)
        hidden_layers = tfkl.Dense(units=1024, activation='relu', name='hidden_2')(hidden_layers)
        hidden_layers = tfkl.Dense(units=512, activation='relu', name='hidden_3')(hidden_layers)
        hidden_layers = tfkl.Dense(units=256, activation='relu', name='hidden_4')(hidden_layers)
        hidden_layers = tfkl.Dense(units=128, activation='relu', name='hidden_5')(hidden_layers)
        hidden_layers = tfkl.Dense(units=64, activation='relu', name='hidden_6')(hidden_layers)
        gap = tfkl.GlobalAveragePooling1D(name='gap')(hidden_layers)

        output_layer = tfkl.Dense(units=2, activation='softmax', name='output')(gap)

        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')
        self.model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics=['accuracy'])

    def train(self, X_train, Y_train, X_val, Y_val, batch_size=128, epochs=10):
        history = self.model.fit(
            x=X_train,
            y=Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=[
                tfk.callbacks.EarlyStopping(monitor='val_accuracy', patience=150, mode='max', restore_best_weights=True),
                tfk.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=5, factor=0.5, min_lr=1e-5)
            ]
        ).history
        return history