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


class BiLSTM:
    def __init__(self, input_shape, lr=0.001, seed=42, classes=2, output_bias=None):
        self.seed = seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.compat.v1.set_random_seed(seed)
        self.model = None
        self.build_bilstm(input_shape, lr, classes, output_bias)

    def build_bilstm(self, input_shape, lr, classes, output_bias):
        if output_bias is not None:
            output_bias = tfk.initializers.Constant(output_bias)

        input_layer = tfkl.Input(shape=input_shape, name='input')

        bi_lstm = tfkl.Bidirectional(tfkl.LSTM(256, return_sequences=True))(input_layer)
        dropout = tfkl.Dropout(.4, seed=self.seed)(bi_lstm)

        bi_lstm = tfkl.Bidirectional(tfkl.LSTM(256))(dropout)
        dropout = tfkl.Dropout(.4, seed=self.seed)(bi_lstm)

        classifier = tfkl.Dense(256, kernel_regularizer=tfk.regularizers.L1L2(l1=1e-3, l2=1e-4))(dropout)
        output_layer = tfkl.Dense(classes, activation='softmax', bias_initializer=output_bias)(classifier)

        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name='biLSTM')

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