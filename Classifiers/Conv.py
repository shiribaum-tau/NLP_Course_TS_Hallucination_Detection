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


class Conv:
    def __init__(self, input_shape, lr=0.001, seed=42, classes=2, output_bias=None):
        self.seed = seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.compat.v1.set_random_seed(seed)
        self.model = None
        self.build_conv(input_shape, lr, classes, output_bias)

    def build_conv(self, input_shape, lr, classes, output_bias):
        if output_bias is not None:
            output_bias = tfk.initializers.Constant(output_bias)

        input_layer = tfkl.Input(shape=input_shape, name='input')

        # cnn1 = tfkl.Conv1D(128, 3, padding='same', activation='relu')(input_layer)
        # b1 = tfkl.BatchNormalization()(cnn1)
        # cnn2 = tfkl.Conv1D(256, 3, padding='same', activation='relu')(cnn1)
        # b2 = tfkl.BatchNormalization()(cnn2)
        # cnn3 = tfkl.Conv1D(128, 3, padding='same', activation='relu')(b2)
        # cnn = tfkl.Add()([b1, b2, cnn3])
        # # cnn = tfkl.MaxPooling1D()(cnn)
        #
        # gap = tfkl.GlobalAveragePooling1D()(cnn)
        # # gap = tfkl.Dropout(0.1, seed=self.seed)(gap)
        #
        # classifier = tfkl.Dense(256, activation='relu')(gap)
        # classifier = tfkl.Dropout(0.4, seed=self.seed)(classifier)

        conv1 = tfkl.Conv1D(128, 8, 1, padding='same')(input_layer)
        conv1 = tfkl.BatchNormalization()(conv1)
        conv1 = tfkl.Activation('relu')(conv1)

        conv1 = tfkl.Dropout(0.1, seed=self.seed)(conv1)

        conv2 = tfkl.Conv1D(256, 5, 1, padding='same')(conv1)
        conv2 = tfkl.BatchNormalization()(conv2)
        conv2 = tfkl.Activation('relu')(conv2)

        conv2 = tfkl.Dropout(0.2, seed=self.seed)(conv2)

        conv3 = tfkl.Conv1D(128, 3, 1, padding='same')(conv2)
        conv3 = tfkl.BatchNormalization()(conv3)
        conv3 = tfkl.Activation('relu')(conv3)

        conv3 = tfkl.Dropout(0.3, seed=self.seed)(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = tfkl.Dense(classes, activation='softmax', bias_initializer=output_bias)(gap)

        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name='Conv')

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