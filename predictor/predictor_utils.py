import numpy as np
import pandas as pd
import os
import logging


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


def setup_logger():
    logging.basicConfig(filename='predictor_log.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        filemode='w'
    )