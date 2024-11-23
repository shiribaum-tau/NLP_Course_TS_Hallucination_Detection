import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import keras
from predictor_utils import CustomAUPRC, CustomAUROC, CustomTP, CustomF1Score
from sklearn.utils import class_weight


plt.rc('font', size=16)
tfk = tf.keras
tfkl = tf.keras.layers


class ResNet:
    def __init__(self, input_shape, lr=0.001, seed=42, classes=2, output_bias=None):
        self.seed = seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.compat.v1.set_random_seed(seed)
        self.model = None
        self.build_resnet(input_shape, lr, classes, output_bias)

    def build_resnet(self, input_shape, lr, classes, output_bias):
        if output_bias is not None:
            output_bias = tfk.initializers.Constant(output_bias)

        input_layer = tfkl.Input(shape=input_shape, name='input')

        conv_x = tfkl.BatchNormalization()(input_layer)
        conv_x = tfkl.Conv1D(64, 8, 1, padding='same')(conv_x)
        conv_x = tfkl.BatchNormalization()(conv_x)
        conv_x = tfkl.Activation('relu')(conv_x)

        conv_y = tfkl.Conv1D(64, 5, 1, padding='same')(conv_x)
        conv_y = tfkl.BatchNormalization()(conv_y)
        conv_y = tfkl.Activation('relu')(conv_y)

        conv_z = tfkl.Conv1D(64, 3, 1, padding='same')(conv_y)
        conv_z = tfkl.BatchNormalization()(conv_z)

        shortcut_y = tfkl.Conv1D(64, 1, 1, padding='same')(input_layer)
        shortcut_y = tfkl.BatchNormalization()(shortcut_y)

        y = tfkl.Add()([shortcut_y, conv_z])
        y = tfkl.Activation('relu')(y)

        x1 = y
        conv_x = tfkl.Conv1D(64 * 2, 8, 1, padding='same')(x1)
        conv_x = tfkl.BatchNormalization()(conv_x)
        conv_x = tfkl.Activation('relu')(conv_x)

        conv_y = tfkl.Conv1D(64 * 2, 5, 1, padding='same')(conv_x)
        conv_y = tfkl.BatchNormalization()(conv_y)
        conv_y = tfkl.Activation('relu')(conv_y)

        conv_z = tfkl.Conv1D(64 * 2, 3, 1, padding='same')(conv_y)
        conv_z = tfkl.BatchNormalization()(conv_z)

        shortcut_y = tfkl.Conv1D(64 * 2, 1, 1, padding='same')(x1)
        shortcut_y = tfkl.BatchNormalization()(shortcut_y)

        y = tfkl.Add()([shortcut_y, conv_z])
        y = tfkl.Activation('relu')(y)

        x1 = y
        conv_x = tfkl.Conv1D(64 * 2, 8, 1, padding='same')(x1)
        conv_x = tfkl.BatchNormalization()(conv_x)
        conv_x = tfkl.Activation('relu')(conv_x)

        conv_y = tfkl.Conv1D(64 * 2, 5, 1, padding='same')(conv_x)
        conv_y = tfkl.BatchNormalization()(conv_y)
        conv_y = tfkl.Activation('relu')(conv_y)

        conv_z = tfkl.Conv1D(64 * 2, 3, 1, padding='same')(conv_y)
        conv_z = tfkl.BatchNormalization()(conv_z)

        shortcut_y = tfkl.Conv1D(64 * 2, 1, 1, padding='same')(x1)
        shortcut_y = tfkl.BatchNormalization()(shortcut_y)

        shortcut_y = tfkl.Dropout(0.2, seed=self.seed)(shortcut_y)

        y = tfkl.Add()([shortcut_y, conv_z])
        y = tfkl.Activation('relu')(y)

        gap = keras.layers.GlobalAveragePooling1D()(y)

        output_layer = tfkl.Dense(classes, activation='softmax', bias_initializer=output_bias)(gap)

        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name='ResNet')

        # Compile the model
        self.model.compile(loss=tfk.losses.BinaryCrossentropy(), optimizer=tfk.optimizers.Adam(learning_rate=lr), metrics=[
            'accuracy', keras.metrics.Precision(class_id=1), keras.metrics.Recall(class_id=1), keras.metrics.F1Score(),
            CustomAUROC(), CustomAUPRC(), CustomTP(), CustomF1Score()])


    def train(self, X_train, Y_train, X_val, Y_val, batch_size=128, epochs=10, callbacks=None):
        if callbacks is None:
            callbacks = [
                tfk.callbacks.EarlyStopping(monitor='val_accuracy', patience=150, mode='max', restore_best_weights=True),
                tfk.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=5, factor=0.5, min_lr=1e-5)
            ]

        # Weight the loss function during training to limit unbiased data effect
        class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(Y_train[:, 1]), y=Y_train[:, 1])
        d_class_weights = dict(enumerate(class_weights))

        history = self.model.fit(
            x=X_train,
            y=Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=callbacks,
            class_weight=d_class_weights
        ).history

        # history = self.model.fit(
        #     x=X_train,
        #     y=Y_train,
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     validation_data=(X_val, Y_val),
        #     callbacks=callbacks
        # ).history

        return history