import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import tensorflow as tf
import keras

tfk = tf.keras
plt.rc('font', size=16)


def create_windows(entry_data, top_k, window_size, stride):
    logits = entry_data['top_k_probs'][:, -top_k:]
    # Cut labels to remove prompt
    labels = entry_data['labels'][-logits.shape[0]:]

    num_windows = ((logits.shape[0] - window_size) // stride) + 1

    # Create windows for the logits based on window size and stride
    logits_windows = np.lib.stride_tricks.sliding_window_view(logits, (window_size, logits.shape[1]))[::stride]
    logits_windows = logits_windows.squeeze(1)

    # Create windows for the labels based on window size and stride
    labels_windows = np.lib.stride_tricks.sliding_window_view(labels, window_size)[::stride]

    assert (num_windows == logits_windows.shape[0])
    assert (num_windows == labels_windows.shape[0])

    return logits_windows, labels_windows


def create_data_tensor(entity_list, data_dir, top_k, window_size, stride):
    logits_windows_list = []
    labels_windows_list = []

    for entity in entity_list:
        try:
            filename = f"{entity}_data.pkl"
            entry = pd.read_pickle(os.path.join(data_dir, filename))

            curr_logits_windows, curr_labels_windows = create_windows(entry, top_k, window_size, stride)

            # Determine the window label: 1 if any element in the window is 1, otherwise 0
            curr_window_labels = calculate_window_label(curr_labels_windows, True)

            # One-hot encode the labels (shape [num_windows, 2])
            one_hot_labels = np.eye(2)[curr_window_labels]

            logits_windows_list.append(curr_logits_windows)
            labels_windows_list.append(one_hot_labels)

        except Exception as e:
            print(f"----- Failed on entity {entity} -----")
            print(f"Reason for failure: {e}")
            print("moving to next entity")

    return logits_windows_list, labels_windows_list


def calculate_window_label(labels_windows, any_one):
    """Return one label for each labels window based on chosen condition

        Args:
            labels_windows (tensor): Tensor shape (num_windows, window_size), of the tokens'
            original labels divided to windows
            any_one (bool): True if the labeling method should be any one
        """
    if any_one:
        return labeling_any_one(labels_windows)
    else:
        return labeling_majority_rules(labels_windows)


def labeling_any_one(labels_windows):
    """Return 1 if the labels window contains one, otherwise return 0

        Args:
            labels_windows (tensor): Tensor shape (num_windows, window_size), of the tokens'
            original labels divided to windows
        """
    return np.any(labels_windows == 1, axis=1).astype(int)


def labeling_majority_rules(labels_windows):
    """Return 1 if the labels window are the majority of 1, otherwise return 0

        Args:
            labels_windows (tensor): Tensor shape (num_windows, window_size), of the tokens'
            original labels divided to windows
        """
    window_size = labels_windows.shape[1]
    return (np.sum(labels_windows, axis=1) >= (window_size // 2)).astype(int)


def setup_logger(save_dir, top_k, window_size, stride):
    logging.basicConfig(filename=f'{save_dir}/predictor_log_k={top_k}_w={window_size}_s={stride}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        filemode='w'
    )


def evaluate_val(X_val, Y_val, X_train, Y_train, history, model, output_dir, top_k, window_size, stride):
    best_accuracy_score = max(history['accuracy'])
    best_accuracy_score_val = max(history['val_accuracy'])
    print("Max train accuracy obtained with this model:", best_accuracy_score)
    print("Max val accuracy obtained with this model:", best_accuracy_score_val)
    logging.info(f'Max train accuracy obtained with this model: {best_accuracy_score}')
    logging.info(f'Max val accuracy obtained with this model: {best_accuracy_score_val}')

    best_auprc_score = max(history['custom_auprc'])
    best_auprc_score_val = max(history['val_custom_auprc'])
    print("Max train AUPRC obtained with this model:", best_auprc_score)
    print("Max val AUPRC obtained with this model:", best_auprc_score_val)
    logging.info(f'Max train AUPRC obtained with this model: {best_auprc_score}')
    logging.info(f'Max val AUPRC obtained with this model: {best_auprc_score_val}')

    best_auroc_score = max(history['custom_auroc'])
    best_auroc_score_val = max(history['val_custom_auroc'])
    print("Max train AUROC obtained with this model:", best_auroc_score)
    print("Max val AUROC obtained with this model:", best_auroc_score_val)
    logging.info(f'Max train AUROC obtained with this model: {best_auroc_score}')
    logging.info(f'Max val AUROC obtained with this model: {best_auroc_score_val}')

    best_f1_score = max(history['custom_f1_score'])
    best_f1_score_val = max(history['val_custom_f1_score'])
    print("Max train F1 score obtained with this model:", best_f1_score)
    print("Max val F1 score obtained with this model:", best_f1_score_val)
    logging.info(f'Max train F1 score obtained with this model: {best_f1_score}')
    logging.info(f'Max val F1 score obtained with this model: {best_f1_score_val}')

    best_epoch_accuracy = np.argmax(history['val_accuracy'])
    best_epoch_f1_score = np.argmax(history['val_custom_f1_score'])
    best_epoch_auprc = np.argmax(history['val_custom_auprc'])
    best_epoch_auroc = np.argmax(history['val_custom_auroc'])


    print("Best epoch - f1 score:", best_epoch_f1_score)
    print("Best epoch - AUPRC:", best_epoch_auprc)
    print("Best epoch - AUROC:", best_epoch_auroc)
    logging.info(f'Best epoch - f1 score: {best_epoch_f1_score}')
    logging.info(f'Best epoch - AUPRC: {best_epoch_auprc}')
    logging.info(f'Best epoch - AUROC: {best_epoch_auroc}')

    plt.figure(figsize=(17, 6))
    plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch_accuracy, label='Best epoch accuracy', alpha=.3, ls='--', color='#5a9aa5')
    plt.axvline(x=best_epoch_f1_score, label='Best epoch f1 score', alpha=.3, ls='--', color='#ff7f0e')
    plt.axvline(x=best_epoch_auprc, label='Best epoch AUPRC', alpha=.3, ls='--', color='#eb330e')
    plt.axvline(x=best_epoch_auroc, label='Best epoch AUROC', alpha=.3, ls='--', color='#61db16')
    plt.title('Binary Crossentropy loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.grid(alpha=.3)
    plt.savefig(f"{output_dir}/Binary_Crossentropy_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()

    plt.figure(figsize=(17, 6))
    plt.plot(range(10, len(history['loss'])), history['loss'][10:], label='Training loss', alpha=.8,
             color='#ff7f0e')
    plt.plot(range(10, len(history['val_loss'])), history['val_loss'][10:], label='Validation loss', alpha=.9,
             color='#5a9aa5')
    plt.axvline(x=best_epoch_accuracy, label='Best epoch accuracy', alpha=.3, ls='--', color='#5a9aa5')
    plt.axvline(x=best_epoch_f1_score, label='Best epoch f1 score', alpha=.3, ls='--', color='#ff7f0e')
    plt.axvline(x=best_epoch_auprc, label='Best epoch AUPRC', alpha=.3, ls='--', color='#eb330e')
    plt.axvline(x=best_epoch_auroc, label='Best epoch AUROC', alpha=.3, ls='--', color='#61db16')
    plt.title('Binary Crossentropy loss (Zoomed in after 10 epochs)')
    plt.legend()
    plt.xlabel('Epoch')
    plt.xlim(10, len(history['loss']) - 1)  # Set the x-axis limit to start from 10
    plt.grid(alpha=.3)
    plt.savefig(f"{output_dir}/Binary_Crossentropy_zoomed_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()


    plt.figure(figsize=(17, 6))
    plt.plot(history['accuracy'], label='Training accuracy', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_accuracy'], label='Validation accuracy', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch_accuracy, label='Best epoch accuracy', alpha=.3, ls='--', color='#5a9aa5')
    plt.axvline(x=best_epoch_f1_score, label='Best epoch f1 score', alpha=.3, ls='--', color='#ff7f0e')
    plt.axvline(x=best_epoch_auprc, label='Best epoch AUPRC', alpha=.3, ls='--', color='#eb330e')
    plt.axvline(x=best_epoch_auroc, label='Best epoch AUROC', alpha=.3, ls='--', color='#61db16')
    plt.title('Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.grid(alpha=.3)
    plt.savefig(f"{output_dir}/Accuracy_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()

    plt.figure(figsize=(17, 6))
    plt.plot(history['learning_rate'], label='Learning Rate', alpha=.8, color='#ff7f0e')
    plt.legend()
    plt.title('Learning rate')
    plt.xlabel('Epoch')
    plt.grid(alpha=.3)
    plt.savefig(f"{output_dir}/Learning_rate_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()

    plt.figure(figsize=(17, 6))
    plt.plot(history['precision'], label='Training precision', alpha=.8, color='#ff7f0e')
    plt.plot(history['recall'], label='Training recall', alpha=.9, color='#5a9aa5')
    plt.plot(history['custom_f1_score'], label='Training F1 score', alpha=.9, color='darkgreen')
    plt.legend()
    plt.title('Precision, Recall, F1 Training')
    plt.xlabel('Epoch')
    plt.grid(alpha=.3)
    plt.savefig(f"{output_dir}/Precision_Recall_F1_Training_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()

    plt.figure(figsize=(17, 6))
    plt.plot(history['val_precision'], label='Validation precision', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_recall'], label='Validation recall', alpha=.9, color='#5a9aa5')
    plt.plot(history['val_custom_f1_score'], label='Validation F1 score', alpha=.9, color='darkgreen')
    plt.axvline(x=best_epoch_accuracy, label='Best epoch accuracy', alpha=.3, ls='--', color='#5a9aa5')
    plt.axvline(x=best_epoch_f1_score, label='Best epoch f1 score', alpha=.3, ls='--', color='#ff7f0e')
    plt.axvline(x=best_epoch_auprc, label='Best epoch AUPRC', alpha=.3, ls='--', color='#eb330e')
    plt.axvline(x=best_epoch_auroc, label='Best epoch AUROC', alpha=.3, ls='--', color='#61db16')
    plt.legend()
    plt.title('Precision, Recall, F1 Validation')
    plt.xlabel('Epoch')
    plt.grid(alpha=.3)
    plt.savefig(f"{output_dir}/Precision_Recall_F1_Validation_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()

    plt.figure(figsize=(17, 6))
    plt.scatter(history['recall'], history['precision'], label='Training', alpha=.8, color='#ff7f0e')
    plt.scatter(history['val_recall'], history['val_precision'], label='Validation', alpha=.9, color='#5a9aa5')
    plt.legend()
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(alpha=.3)
    plt.savefig(f"{output_dir}/PR_Curve_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()

    plt.figure(figsize=(17, 6))
    plt.plot(history['custom_auroc'], label='Training auroc', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_custom_auroc'], label='Validation auroc', alpha=.8, color='#5a9aa5')
    plt.axvline(x=best_epoch_accuracy, label='Best epoch accuracy', alpha=.3, ls='--', color='#5a9aa5')
    plt.axvline(x=best_epoch_f1_score, label='Best epoch f1 score', alpha=.3, ls='--', color='#ff7f0e')
    plt.axvline(x=best_epoch_auprc, label='Best epoch AUPRC', alpha=.3, ls='--', color='#eb330e')
    plt.axvline(x=best_epoch_auroc, label='Best epoch AUROC', alpha=.3, ls='--', color='#61db16')
    plt.legend()
    plt.title('Area Under ROC Curve')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC value')
    plt.grid(alpha=.3)
    plt.savefig(f"{output_dir}/AUROC_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()

    plt.figure(figsize=(17, 6))
    plt.plot(history['custom_auprc'], label='Training auprc', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_custom_auprc'], label='Validation auprc', alpha=.8, color='#5a9aa5')
    plt.axvline(x=best_epoch_accuracy, label='Best epoch accuracy', alpha=.3, ls='--', color='#5a9aa5')
    plt.axvline(x=best_epoch_f1_score, label='Best epoch f1 score', alpha=.3, ls='--', color='#ff7f0e')
    plt.axvline(x=best_epoch_auprc, label='Best epoch AUPRC', alpha=.3, ls='--', color='#eb330e')
    plt.axvline(x=best_epoch_auroc, label='Best epoch AUROC', alpha=.3, ls='--', color='#61db16')
    plt.legend()
    plt.title('Area Under Precision-Recall Curve')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC value')
    plt.grid(alpha=.3)
    plt.savefig(f"{output_dir}/AUPRC_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()

    predictions_val = model.model.predict(X_val)
    print(f"predictions shape: {predictions_val.shape}")

    predictions_train = model.model.predict(X_train)

    cm_val = confusion_matrix(np.argmax(Y_val, axis=-1), np.argmax(predictions_val, axis=-1))
    cm_val = cm_val.T

    col_sum = cm_val.sum(axis=0)
    cm_prob_val = cm_val / col_sum

    cm_train = confusion_matrix(np.argmax(Y_train, axis=-1), np.argmax(predictions_train, axis=-1))
    cm_train = cm_train.T

    col_sum = cm_train.sum(axis=0)
    cm_prob_train = cm_train / col_sum

    accuracy = accuracy_score(np.argmax(Y_val, axis=-1), np.argmax(predictions_val, axis=-1))
    precision = precision_score(np.argmax(Y_val, axis=-1), np.argmax(predictions_val, axis=-1),
                                zero_division=0)
    recall = recall_score(np.argmax(Y_val, axis=-1), np.argmax(predictions_val, axis=-1))
    f1 = f1_score(np.argmax(Y_val, axis=-1), np.argmax(predictions_val, axis=-1))
    fpr, tpr, thresholds = roc_curve(np.argmax(Y_val, axis=-1), np.argmax(predictions_val, axis=-1))
    roc_auc = auc(fpr, tpr)

    print('Accuracy:', '{:.4f}'.format(accuracy))
    print('Precision:', '{:.4f}'.format(precision))
    print('Recall:', '{:.4f}'.format(recall))
    print('F1:', '{:.4f}'.format(f1))
    print('AUROC:', '{:.4f}'.format(roc_auc))

    logging.info(f'Accuracy: {"{:.4f}".format(accuracy)}')
    logging.info(f'Precision: {"{:.4f}".format(precision)}')
    logging.info(f'Recall: {"{:.4f}".format(recall)}')
    logging.info(f'F1: {"{:.4f}".format(f1)}')
    logging.info(f'AUROC: {"{:.4f}".format(roc_auc)}')

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_prob_val, cmap='Blues', annot=True, fmt='.2f', vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .25, .5, .75, 1])
    plt.title('Confusion Matrix Validation')
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.savefig(f"{output_dir}/Confusion_Matrix_val_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_prob_train, cmap='Blues', annot=True, fmt='.2f', vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .25, .5, .75, 1])
    plt.title('Confusion Matrix Train')
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.savefig(f"{output_dir}/Confusion_Matrix_train_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()

    plt.figure(figsize=(17, 6))
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.4f)' % roc_auc, alpha=.8, color='#ff7f0e')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.legend()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(alpha=.3)
    plt.savefig(f"{output_dir}/ROC_Curve_k={top_k}_w={window_size}_s={stride}.png")
    plt.show()


def evaluate_test(X_test, Y_test, model_file, model_name, output_dir, data_gen_model):
    predictions_test = model_file.predict(X_test)

    cm_test = confusion_matrix(np.argmax(Y_test, axis=-1), np.argmax(predictions_test, axis=-1))
    cm_test = cm_test.T

    col_sum = cm_test.sum(axis=0)
    cm_prob_test = cm_test / col_sum

    accuracy = accuracy_score(np.argmax(Y_test, axis=-1), np.argmax(predictions_test, axis=-1))
    precision = precision_score(np.argmax(Y_test, axis=-1), np.argmax(predictions_test, axis=-1),
                                zero_division=0)
    recall = recall_score(np.argmax(Y_test, axis=-1), np.argmax(predictions_test, axis=-1))
    f1 = f1_score(np.argmax(Y_test, axis=-1), np.argmax(predictions_test, axis=-1))
    fpr, tpr, thresholds = roc_curve(np.argmax(Y_test, axis=-1), np.argmax(predictions_test, axis=-1))
    roc_auc = auc(fpr, tpr)

    print('Accuracy:', '{:.4f}'.format(accuracy))
    print('Precision:', '{:.4f}'.format(precision))
    print('Recall:', '{:.4f}'.format(recall))
    print('F1:', '{:.4f}'.format(f1))
    print('AUROC:', '{:.4f}'.format(roc_auc))

    logging.info(f'Accuracy: {"{:.4f}".format(accuracy)}')
    logging.info(f'Precision: {"{:.4f}".format(precision)}')
    logging.info(f'Recall: {"{:.4f}".format(recall)}')
    logging.info(f'F1: {"{:.4f}".format(f1)}')
    logging.info(f'AUROC: {"{:.4f}".format(roc_auc)}')

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_prob_test, cmap='Blues', annot=True, fmt='.2f', vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .25, .5, .75, 1])
    plt.title('Confusion Matrix Test')
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.savefig(f"{output_dir}/Confusion_Matrix_test_model={model_name}_data={data_gen_model}.png")
    plt.show()

    plt.figure(figsize=(17, 6))
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.4f)' % roc_auc, alpha=.8, color='#ff7f0e')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.legend()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(alpha=.3)
    plt.savefig(f"{output_dir}/ROC_Curve_test_model={model_name}_data={data_gen_model}.png")
    plt.show()
    print()


def create_checkpoint_callback(ckpt_dir, steps_per_epoch, top_k, window_size, stride, t=10):
    # Saves the model every t epochs
    periodic_checkpoint = tfk.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, f'model_k={top_k}_w={window_size}_s={stride}_epoch_{{epoch:02d}}.keras'),
        save_freq=t * steps_per_epoch,  # `steps_per_epoch` == number of batches per epoch
        save_weights_only=False
    )

    # Saves the best model separately
    best_model_checkpoint = tfk.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, f'best_model_k={top_k}_w={window_size}_s={stride}.keras'),
        monitor='val_custom_auroc',
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )

    # Stops training when a monitored metric has stopped improving
    early_stopping = tfk.callbacks.EarlyStopping(
        monitor='val_custom_auroc',
        patience=150,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    # Reduce learning rate when a metric has stopped improving
    reduce_lr_on_plateau = tfk.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        patience=10,
        factor=0.2,
        min_lr=1e-5,
        verbose=1
    )

    return [periodic_checkpoint, best_model_checkpoint, early_stopping, reduce_lr_on_plateau]


class CustomAUROC(keras.metrics.Metric):
    def __init__(self, name='custom_auroc', **kwargs):
        super(CustomAUROC, self).__init__(name=name, **kwargs)
        self.auc = keras.metrics.AUC(curve="ROC")  # AUC metric instantiated once

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract probabilities for the positive class (class 1)
        pos_probs = tf.reshape(tf.argmax(y_pred, axis=-1), (-1, 1))
        # Extract the actual labels for the positive class
        true_labels = tf.reshape(tf.argmax(y_true, axis=-1), (-1, 1))

        # Update the state of the internal AUC metric
        self.auc.update_state(true_labels, pos_probs, sample_weight)

    def result(self):
        return self.auc.result()

    def reset_states(self):
        # Reset the state of the internal AUC metric
        self.auc.reset_states()


class CustomAUPRC(keras.metrics.Metric):
    def __init__(self, name='custom_auprc', **kwargs):
        super(CustomAUPRC, self).__init__(name=name, **kwargs)
        self.auc = keras.metrics.AUC(curve="PR")  # AUC metric instantiated once

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract probabilities for the positive class (class 1)
        pos_probs = tf.reshape(tf.argmax(y_pred, axis=-1), (-1, 1))
        # Extract the actual labels for the positive class
        true_labels = tf.reshape(tf.argmax(y_true, axis=-1), (-1, 1))

        # Update the state of the internal AUC metric
        self.auc.update_state(true_labels, pos_probs, sample_weight)

    def result(self):
        return self.auc.result()

    def reset_states(self):
        # Reset the state of the internal AUC metric
        self.auc.reset_states()


class CustomTP(keras.metrics.Metric):
    def __init__(self, name='custom_tp', **kwargs):
        super(CustomTP, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract probabilities for the positive class (class 1)
        pos_probs = tf.reshape(tf.argmax(y_pred, axis=-1), (-1, 1))
        # Extract the actual labels for the positive class
        true_labels = tf.reshape(tf.argmax(y_true, axis=-1), (-1, 1))

        # Update the state of the internal AUC metric
        self.tp.update_state(true_labels, pos_probs)

    def result(self):
        return self.tp.result()

    def reset_states(self):
        # Reset the state of the internal AUC metric
        self.tp.reset_states()


class CustomF1Score(keras.metrics.Metric):
    def __init__(self, name='custom_f1_score', **kwargs):
        super(CustomF1Score, self).__init__(name=name, **kwargs)
        self.f1 = keras.metrics.F1Score()  # AUC metric instantiated once

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract probabilities for the positive class (class 1)
        pos_probs = tf.reshape(tf.argmax(y_pred, axis=-1), (-1, 1))
        # Extract the actual labels for the positive class
        true_labels = tf.reshape(tf.argmax(y_true, axis=-1), (-1, 1))

        # Update the state of the internal AUC metric
        self.f1.update_state(true_labels, pos_probs, sample_weight)

    def result(self):
        return self.f1.result()

    def reset_states(self):
        # Reset the state of the internal AUC metric
        self.f1.reset_states()
