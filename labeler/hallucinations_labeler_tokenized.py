import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPTNeoXForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-2.8b")
tokenizer.pad_token = tokenizer.eos_token

auxiliary_verbs = [
    "be", "have", "do", "can", "could", "may", "might",
    "must", "shall", "should", "will", "would",
    "am", "is", "are", "was", "were",
    "has", "have", "had",
    "do", "does", "did",
    "can", "could",
    "may", "might",
    "must",
    "shall", "should",
    "will", "would"]

auxiliary_verbs_tokens = [tokenizer(aux_v)[0].ids[0] for aux_v in auxiliary_verbs]

punctuations_extended = r"""!"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"""

punctuations_extended_tokens = [tokenizer(c)[0].ids[0] for c in punctuations_extended]


def remove_auxiliary_verbs_tokens(tokens: list, remove_punctuation = False):
    for t in auxiliary_verbs_tokens:
        while t in tokens:
            tokens.remove(t)
    if remove_punctuation:
        for t in punctuations_extended_tokens:
            while t in tokens:
                tokens.remove(t)


def reduce_facts(atoms, method):
    if len(atoms) == 1:
        return [tokenizer(atoms[0]['text'])[0].ids], [atoms[0]['label']]
    cut_atoms = []

    if method == 'Weak Uniqueness':
        unique_words = set(tokenizer(atoms[0]['text'])[0].ids)
        for atom in atoms[1:]:
            current_unique_words = set(tokenizer(atom['text'])[0].ids)
            unique_words = unique_words.intersection(current_unique_words)
        for atom in atoms:
            atom_sentence = tokenizer(atom['text'])[0].ids
            for word in unique_words:
                atom_sentence = [token for token in atom_sentence if token != word]
            cut_atoms.append(atom_sentence)
        labels = [atom['label'] for atom in atoms]
        return cut_atoms, labels

    if method == "Cascading Deletion":
        atoms_df = pd.DataFrame(atoms)['text']
        atoms_num = len(atoms_df)
        atoms_df = [tokenizer(atoms_df[i])[0].ids for i in range(atoms_num)]
        labels = [pd.DataFrame(atoms)['label'][i] for i in range(atoms_num)]

        atoms_to_labels = {}
        for i in range(atoms_num):
            atoms_to_labels[str(atoms_df[i])] = labels[i]

        merged_sorted_atoms = sorted(atoms_df, key=len)
        sorted_atoms_by_length = merged_sorted_atoms
        sorted_labels_by_length = [atoms_to_labels[str(merged_sorted_atoms[i])] for i in range(atoms_num)]

        for atom_ind, atom_text in enumerate(sorted_atoms_by_length):
            for word in atom_text:
                for next_atom_ind in range(atom_ind+1, atoms_num):
                    next_atom = sorted_atoms_by_length[next_atom_ind]
                    next_atom_updated = [token for token in next_atom if token != word]
                    sorted_atoms_by_length[next_atom_ind] = next_atom_updated

        for i in range(len(sorted_atoms_by_length)):
            atom = sorted_atoms_by_length[i]
            merged_atom = atom
            cut_atoms.append(merged_atom)
        return cut_atoms, sorted_labels_by_length

    if method == 'Strong Uniqueness':
      #  Should only make sure values unique to a single sentence remain.
      #  Should be implemented with a dictionary and count
        raise ValueError('Strong Uniqueness not yet implemented')
    else:
        raise ValueError('method arg is invalid')


def find_index_of_fact_in_sentence(fact, sentence):
    ind_list = []
    for i, fact_word in enumerate(fact):
        if fact_word not in sentence:
            continue
        word_ind = sentence.index(fact_word)
        ind_list.append(word_ind)
    ind_array = np.array(ind_list)
    n = len(ind_array)
    # Given an exponentially decaying weight to the indices, to find the proper index where the fact leads.
    # sum of all weight should be 1.
    if n == 0:
        return -1, None
    a = (0.5/(1-(0.5)**n))
    weights_array_exp = np.array([a/(2**i) for i in range(n-1, -1, -1)])
    return int(ind_array @ weights_array_exp), ind_list


def read_our_json(filename):
    """Parses a JSONL file and extracts the numeric data.

    Args:
        filename: The path to the JSONL file.

    Returns:
        A Pandas DataFrame containing the extracted numeric data.
    """

    data = []
    with open(filename, 'r') as f:
        for line in f:
            question_dictionary = json.loads(line)
            for d in question_dictionary:
                data.append(d)

    df = pd.DataFrame(data)
    return df


def read_jsonl(filename):
    """Reads a JSONL file into a Pandas DataFrame.

    Args:
      filename: The path to the JSONL file.

    Returns:
      A Pandas DataFrame containing the data from the JSONL file.
    """

    with open(filename, 'r') as f:
      lines = [json.loads(line) for line in f]

    df = pd.DataFrame(lines)
    return df


def find_missing_elements(list1, list2):
    missing_elements = []
    missed_values = 0
    for i, element in enumerate(list1):
        if i - missed_values == len(list2):
            list2.append(element)
        if list2[i-missed_values] != element:
            missing_elements.append(element)
            missed_values += 1
    return missing_elements


def find_pre_sentence(paragraph: str, index):
    sub_paragraph = paragraph[:index]
    # last_dot_index = sub_paragraph.rfind('.')
    # last_colun_index = sub_paragraph.rfind(':')
    # dot_semi_colun = paragraph[max(last_dot_index, last_colun_index) + 1: index]

    if index == 0:
        return "", -1
    backwards_indices = 1
    while sub_paragraph[-backwards_indices] in [" ", "\n"]:
        backwards_indices += 1
        if backwards_indices == len(sub_paragraph) + 1:
            break
    beginning_index = index - backwards_indices + 1
    x = paragraph[beginning_index : index]
    return x, beginning_index


def get_full_sentence_start_and_end_index(partial_sentence, generation, previous_sentence_end_index):
    sentence_text_beginning_ind = generation.find(partial_sentence, previous_sentence_end_index + 1)
    if sentence_text_beginning_ind == -1:
        # Should never happen.
        raise Exception("FUUUUCCCCKKKKKK, I ASSUMED THE ENTIRE SENTENCE IS INSIDE THE ORIGINAL, FUCK")
    pre_sentence, beginning_ind = find_pre_sentence(generation, sentence_text_beginning_ind)
    full_sentence = f"{pre_sentence}{partial_sentence}"
    end_index = previous_sentence_end_index + len(partial_sentence) - 1
    return full_sentence, end_index, beginning_ind


def rebuild_original_paragraph_tokenized(so_far_rebuilt_paragraph, sentence_to_rebuild):
    for t in sentence_to_rebuild:
        so_far_rebuilt_paragraph.append(t)


def find_hallucination_indices_in_sentence(facts, labels, tokenized_sentence):
    sentence_hallucinations_indices = []
    for rfact, label in zip(facts, labels):
        if label != "NS":
            continue
        remove_auxiliary_verbs_tokens(rfact, remove_punctuation_from_facts)
        rrfact = rfact
        # print(rrfact)

        if len(rrfact) != 0:
            fact_ind, fact_indices_list = find_index_of_fact_in_sentence(rrfact, tokenized_sentence)
            if fact_ind == -1 or fact_indices_list is None:
                continue
            for ind in fact_indices_list:
                sentence_hallucinations_indices.append(ind)
    return sentence_hallucinations_indices

# Example usage:
show_print = True
remove_punctuation_from_facts = True
example_output_2 = r"C:\Users\Arik Drori\Desktop\Year3+\NLP\FinalProject\ts_hallucination\fact_checked_data\pythia_2.8_deterministic_fact_checked_bm25.json"
df = read_our_json(example_output_2)
generations = df['output']
annotations = df['annotations']
for annotation, generation in zip(annotations, generations):
    if annotation is None:
        continue
    total_annotation_length = 0
    hallucination_indices = []
    rebuilt_paragraph = []
    original_paragraph = tokenizer(generation)[0].ids
    last_sentence_end_index = -1
    for sentence in annotation:
        sentence_text_original, this_sentence_end_ind, this_sentence_start_ind = get_full_sentence_start_and_end_index(sentence['text'], generation,
                                                                                          last_sentence_end_index)

        last_sentence_start_ind = this_sentence_end_ind
        sentence_text = sentence_text_original
        relevancy = sentence['is-relevant']
        atomic_facts = sentence['human-atomic-facts']

        sentence_text_tokenized = tokenizer(sentence_text_original, padding=True, truncation=True)[0].ids
        rebuild_original_paragraph_tokenized(rebuilt_paragraph, sentence_text_tokenized)

        if atomic_facts is None or len(atomic_facts) == 0:
            continue

        # Reduce facts to weak-uniqueness (Words that appear in ALL atoms are delteted), or Cascading Deletion (Every
        # word seen in a fact will be deleted in the following ones)
        reduced_facts, labels = reduce_facts(atomic_facts, method='Cascading Deletion')
        sentence_hallucinations = find_hallucination_indices_in_sentence(reduced_facts, labels, sentence_text_tokenized)
        for ind in sentence_hallucinations:
            hallucination_indices.append(ind + total_annotation_length)
        total_annotation_length += len(sentence_text_tokenized)

    if show_print:
        marked_hallucinations_sentence = []
        missing_tokens = find_missing_elements(original_paragraph, rebuilt_paragraph)
        if len(missing_tokens) != 0:
            print("Missing tokens " + str(tokenizer.decode(missing_tokens)))
        for i, t in enumerate(rebuilt_paragraph):
            if i in hallucination_indices:
                marked_hallucinations_sentence.append(60)  # token for [
                marked_hallucinations_sentence.append(t)  # token
                marked_hallucinations_sentence.append(62)  # token for ]
            else:
                marked_hallucinations_sentence.append(t)
        marked_hallucinations_sentence_decoded = tokenizer.decode(marked_hallucinations_sentence)
        print(marked_hallucinations_sentence_decoded)
