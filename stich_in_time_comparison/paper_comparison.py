import pickle
import os
import pandas
import torch
import numpy as np
from ts_hallucination.predictor.predictor_utils import create_windows
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def do_thing(probs, concepts):
    relevant_probs = probs[concepts == 1]
    if len(relevant_probs) == 0:
        return 1
    return np.min(relevant_probs)

# def do_min_of_max(logits):
#     return np.min(np.max(logits,axis=2), axis = 1)


threshold_array = np.linspace(0.2, 1, 100)
accuracies = np.zeros(len(threshold_array))
recalls = np.zeros(len(threshold_array))
precisions = np.zeros(len(threshold_array))
winsize = 5
stride = 1
topk = 20

noob_person_absolute_path = r"C:\Users\Arik Drori\Desktop\Year3+\NLP\FinalProject\ts_hallucination\data_for_paper_comparison\openai_chosen_token_and_concepts"



for i, thresh in enumerate(threshold_array):
    all_probs = []
    all_labels = []
    for root, dirs, files in os.walk(noob_person_absolute_path):
        for file in files:
            file_path = os.path.join(root, file)
            data = None

            with open(file_path, "rb") as f:
                data = pandas.read_pickle(f)

            logit_windows, labels_windows = create_windows(data, topk, winsize, stride)

            # entity = os.path.basename(file_path).replace("_data.pkl", "")
            # with open(os.path.join(noob_person_absolute_path, f"{entity}_gpt-4o-mini-2024-07-18_300_w_chosen_token_and_concepts.pkl"), "rb") as f:
            #     dat2 = pickle.load(f)
            data['chosen_token_prob2'] = data['chosen_token_prob'].unsqueeze(1)
            prob_windows, concept_windows = create_windows(data, topk, winsize, stride, logit_key_name="chosen_token_prob2",
                                                            label_key_name='concept_words')
            prob_windows = prob_windows.squeeze(2)
            for probs, labels, concepts in zip(prob_windows, labels_windows, concept_windows):
                all_probs.append(do_thing(probs, concepts))
                all_labels.append(1 if 1 in labels else 0)


            # new_labels = np.where(labels_windows.any(axis=1), 1, 0)


            # min_of_max = do_min_of_max(logit_windows)

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        hallucinations = all_probs < thresh

        accuracy = np.sum(hallucinations == all_labels) / len(all_labels)
        accuracies[i] = accuracy
        if np.sum(all_labels)!= 0:
            recall = np.sum(hallucinations * all_labels) / np.sum(all_labels)
            recalls[i] = recall
        if np.sum(hallucinations) != 0:
            precision = np.sum(hallucinations * all_labels) / np.sum(hallucinations)
            precisions[i] = precision




plt.plot(threshold_array, recalls)
plt.xlabel('Probability Threshold')
plt.ylabel('Recall')
plt.show()
plt.plot(threshold_array, recalls, label='Recall')
plt.plot(threshold_array, precisions, label = 'Precision')
plt.xlabel('Probability Threshold')
plt.ylabel('Recall / Precision')
plt.legend()
plt.show()
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
auc_pr = auc(recalls, precisions)
print(auc_pr)
plt.text(0.9, 0.9, f'AUC-PR: {auc_pr:.2f}', fontsize=6)
plt.show()


# def do_thing(probs, concepts):
#     relevant_probs = probs[concepts == 1]
#     if len(relevant_probs) == 0:
#         return 1
#     return np.min(relevant_probs)
#
# for winsize in [5, 10, 50]:
#     for stride in [winsize]:
#         print(f"Doing {winsize}:{stride}")
#         all_probs = []
#         all_labels = []
#         for path in glob.glob(os.path.join('mini_GPT', "*.pkl")):
#             with open(path, "rb") as f:
#                 dat = pickle.load(f)
#                 #
#             try:
#                 logits_windows, labels_windows = create_windows(dat, winsize, stride)
#                 entity = os.path.basename(path).replace("_data.pkl", "")
#                 with open(os.path.join(paper_comp_data_loc, f"{entity}_gpt-4o-mini-2024-07-18_300_w_chosen_token_and_concepts.pkl"), "rb") as f:
#                     dat2 = pickle.load(f)
#                     dat2['chosen_token_prob2'] = dat2['chosen_token_prob'].unsqueeze(1)
#                 prob_windows, concept_windows = create_windows(dat2, winsize, stride, logit_key_name="chosen_token_prob2", label_key_name='concept_words')
#                 prob_windows = prob_windows.squeeze(2)
#                 for probs, labels, concepts in zip(prob_windows, labels_windows, concept_windows):
#                     all_probs.append(do_thing(probs, concepts))
#                     all_labels.append(1 if 1 in labels else 0)
#             except Exception as e:
#                 print(f"Skipping {path}: {e}")

