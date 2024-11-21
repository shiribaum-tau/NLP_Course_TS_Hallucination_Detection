import pickle
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from collections import Counter
import glob


def create_windows(entry_data, top_k, window_size, stride, logit_key_name = "top_k_probs", label_key_name = "labels"):
    logits = entry_data[logit_key_name][:, -top_k:]
    # Cut labels to remove prompt
    labels = entry_data[label_key_name][-logits.shape[0]:]

    num_windows = ((logits.shape[0] - window_size) // stride) + 1

    # Create windows for the logits based on window size and stride
    logits_windows = np.lib.stride_tricks.sliding_window_view(logits, (window_size, logits.shape[1]))[::stride]
    logits_windows = logits_windows.squeeze(1)

    # Create windows for the labels based on window size and stride
    labels_windows = np.lib.stride_tricks.sliding_window_view(labels, window_size)[::stride]

    assert (num_windows == logits_windows.shape[0])
    assert (num_windows == labels_windows.shape[0])

    return logits_windows, labels_windows


def stitichintime_metric(probs, concepts):
    relevant_probs = probs[concepts == 1]
    if len(relevant_probs) == 0:
        return 1
    return np.min(relevant_probs)


def do_min_of_max(probs, concepts):
    relevant_probs = probs[concepts == 1]
    if len(relevant_probs) == 0:
        return 1
    return np.min(np.max(relevant_probs, axis=1))


threshold_array = np.linspace(0.2, 1, 100)
accuracies = np.zeros(len(threshold_array))
recalls = np.zeros(len(threshold_array))
precisions = np.zeros(len(threshold_array))
winsize = 5
stride = 1
topk = 20

DATA_PATH = os.path.join("data_for_paper_comparison", "openai_chosen_token_and_concepts")

# Generate Dataset

probs_SIT_metric = []
probs_our_metric = []
all_labels = []
for file_path in glob.glob(os.path.join(DATA_PATH, "*.pkl")):
    with open(file_path, "rb") as f:
        data = pandas.read_pickle(f)

    logit_windows, labels_windows = create_windows(data, topk, winsize, stride)

    data['chosen_token_prob2'] = data['chosen_token_prob'].unsqueeze(1)
    prob_windows, concept_windows = create_windows(data, topk, winsize, stride, logit_key_name="chosen_token_prob2",
                                                    label_key_name='concept_words')
    prob_windows = prob_windows.squeeze(2)
    for probs, labels, concepts, logits in zip(prob_windows, labels_windows, concept_windows, logit_windows):
        probs_SIT_metric.append(stitichintime_metric(probs, concepts))
        probs_our_metric.append(do_min_of_max(logits, concepts))
        all_labels.append(1 if 1 in labels else 0)

# Undersampling for balance
np.random.seed(42)
probs_SIT_metric = np.array(probs_SIT_metric)
probs_our_metric = np.array(probs_our_metric)
all_labels = np.array(all_labels)

counter = Counter(all_labels)
min_class_size = min(counter.values())

# Separate truth from hallucination
true_probs_SIT = probs_SIT_metric[all_labels == 0]
hall_probs_SIT = probs_SIT_metric[all_labels == 1]
true_probs_our_metric = probs_our_metric[all_labels == 0]
hall_probs_our_metric = probs_our_metric[all_labels == 1]

# Undersample
true_chosen_indices = np.random.choice(len(true_probs_SIT), min_class_size, replace=False)
hall_chosen_indices = np.random.choice(len(hall_probs_SIT), min_class_size, replace=False)

chosen_true_probs_SIT = true_probs_SIT[true_chosen_indices]
chosen_hall_probs_SIT = hall_probs_SIT[hall_chosen_indices]

chosen_true_probs_our_metric = true_probs_our_metric[true_chosen_indices]
chosen_hall_probs_our_metric = hall_probs_our_metric[hall_chosen_indices]

chosen_true_labels = np.zeros(min_class_size)
chosen_hall_labels = np.ones(min_class_size)


probs_SIT_metric = np.concatenate((chosen_true_probs_SIT, chosen_hall_probs_SIT), axis=0)
probs_our_metric = np.concatenate((chosen_true_probs_our_metric, chosen_hall_probs_our_metric), axis=0)
all_labels = np.concatenate((chosen_true_labels, chosen_hall_labels), axis=0)


# Calculate AUC

for i, thresh in enumerate(threshold_array):
    hallucinations = probs_SIT_metric < thresh

    accuracy = np.sum(hallucinations == all_labels) / len(all_labels)
    accuracies[i] = accuracy
    if np.sum(all_labels) != 0:
        recall = np.sum(hallucinations * all_labels) / np.sum(all_labels)
        recalls[i] = recall
    if np.sum(hallucinations) != 0:
        precision = np.sum(hallucinations * all_labels) / np.sum(hallucinations)
        precisions[i] = precision


plt.plot(threshold_array, recalls, label='Recall')
plt.plot(threshold_array, precisions, label = 'Precision')
plt.xlabel('Probability Threshold')
plt.ylabel('Recall / Precision')
plt.legend()
plt.title("Precision + Recall")
plt.show()
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
auc_pr = auc(recalls, precisions)
print(auc_pr)
plt.text(0.95, 0.95, f'AUC-PR: {auc_pr:.2f}',
         fontsize=15, ha='right', va='top', transform=plt.gca().transAxes) 
plt.show()


