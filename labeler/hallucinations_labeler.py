import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

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

punctuations_extended = r"""!"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"""


def delete_word_from_atom_sentence_and_join(word: str, atom_sentence: list):
    for i in range(len(atom_sentence)):
        if atom_sentence[i] == word:
            atom_sentence[i] = ''
    return " ".join(atom_sentence[i] for i in range(len(atom_sentence)) if atom_sentence[i] != "")


def remove_auxiliary_verbs(string: str):
    split_string = string.split(" ")
    for s in auxiliary_verbs:
        while s in split_string:
            split_string.remove(s)
    return ' '.join(split_string)


def reduce_facts(atoms, method):
    if len(atoms) == 1:
        return [atoms[0]['text'].lower()], [atoms[0]['label']]
    cut_atoms = []
    labels = []

    if method == 'Weak Uniqueness':
        unique_words = set(atoms[0]['text'].lower().split(' '))
        for atom in atoms[1:]:
            current_unique_words = set(atom['text'].lower().split(' '))
            unique_words = unique_words.intersection(current_unique_words)
        for atom in atoms:
            atom_sentence = atom['text'].lower()
            for word in unique_words:
                atom_sentence = atom_sentence.split(" ")
                atom_sentence = delete_word_from_atom_sentence_and_join(word, atom_sentence)
            cut_atoms.append(atom_sentence)
        labels = [atom['label'] for atom in atoms]
        return cut_atoms, labels

    if method == "Cascading Deletion":
        atoms_df = pd.DataFrame(atoms)['text']
        atoms_num = len(atoms_df)
        atoms_df = [atoms_df[i].lower() for i in range(atoms_num)]
        labels = [pd.DataFrame(atoms)['label'][i] for i in range(atoms_num)]

        atoms_to_labels = {}
        for i in range(atoms_num):
            atoms_to_labels[atoms_df[i]] = labels[i]

        sorted_atoms_by_length = sorted([atoms_df[i].split(' ') for i in range(atoms_num)], key=len)
        merged_sorted_atoms = [" ".join(sorted_atoms_by_length[i]) for i in range(atoms_num)]
        sorted_labels_by_length = [atoms_to_labels[merged_sorted_atoms[i]] for i in range(atoms_num)]

        for atom_ind, atom_text in enumerate(sorted_atoms_by_length):
            for word in atom_text:
                for next_atom_ind in range(atom_ind+1, atoms_num):
                    next_atom = sorted_atoms_by_length[next_atom_ind]
                    next_atom_updated = delete_word_from_atom_sentence_and_join(word, next_atom).split(" ")
                    sorted_atoms_by_length[next_atom_ind] = next_atom_updated
        for i in range(len(sorted_atoms_by_length)):
            atom = sorted_atoms_by_length[i]
            merged_atom = " ".join(atom)
            cut_atoms.append(merged_atom)
        return cut_atoms, sorted_labels_by_length


    if method == 'Strong Uniqueness':
      #  Should only make sure values unique to a single sentence remain.
      #  Should be implemented with a dictionary and count
        raise ValueError('Strong Uniqueness not yet implemented')
    else:
        raise ValueError('method arg is invalid')


def find_index_of_fact_in_sentence(fact: str, sentence: str):
    # Option 1 - Look
    split_fact = fact.translate(str.maketrans(punctuations_extended, ' '*len(punctuations_extended))).split(" ")
    split_fact = [split_fact[i] for i in range(len(split_fact)) if split_fact[i] != ""]
    split_sentence = sentence.translate(str.maketrans(punctuations_extended, ' '*len(punctuations_extended))).split(" ")
    split_sentence = [split_sentence[i] for i in range(len(split_sentence)) if split_sentence[i] != ""]
    ind_array = []
    for i, fact_word in enumerate(split_fact):
        if fact_word not in split_sentence:
            continue
        word_ind = split_sentence.index(fact_word)
        ind_array.append(word_ind)
    ind_array = np.array(ind_array)
    n = len(ind_array)
    # Given an exponentially decaying weight to the indices, to find the proper index where the fact leads.
    # sum of all weight should be 1.
    if n == 0:
        return -1
    a = (0.5/(1-(0.5)**n))
    weights_array_exp = np.array([a/(2**i) for i in range(n-1, -1, -1)])
    return int(ind_array @ weights_array_exp)


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


# Example usage:
#jsonl_file = r"C:\Users\Arik Drori\Desktop\Year3+\NLP\FinalProject\ts_hallucination\answers_gpt4_bio_test_addtional.jsonl"
jsonl_gpt = r"C:\Users\Arik Drori\Desktop\Year3+\NLP\FinalProject\ts_hallucination\labeler\ChatGPT.jsonl"
df = read_jsonl(jsonl_gpt)
generations = df['output']
annotations = df['annotations']
for annotation, generation in zip(annotations, generations):
    if annotation is None:
        continue
    total_annotation_length = 0
    hallucination_indices = []
    rebuilt_sentence = []
    for sentence in annotation:
        sentence_text = sentence['text'].lower()
        relevancy = sentence['is-relevant']
        atomic_facts = sentence['human-atomic-facts']

        if atomic_facts is None:
            continue

        split_sentence = sentence_text.translate(str.maketrans(punctuations_extended, ' ' * len(punctuations_extended))).split(" ")
        split_sentence = [split_sentence[i] for i in range(len(split_sentence)) if split_sentence[i] != ""]
        for word in split_sentence:
            rebuilt_sentence.append(word)

        # Reduce facts to weak-uniqueness (Words that appear in ALL atoms are delteted)
        reduced_facts, labels = reduce_facts(atomic_facts, method='Cascading Deletion')
        for rfact, label in zip(reduced_facts, labels):
            if label != "NS":
                continue
            rrfact = remove_auxiliary_verbs(rfact)
            print(rrfact)
            if rrfact != "":
                fact_ind = find_index_of_fact_in_sentence(rrfact, sentence_text)
                print(fact_ind + total_annotation_length)
                hallucination_indices.append(fact_ind + total_annotation_length)
        total_annotation_length += len(sentence_text.split(" "))
    for i in hallucination_indices:
        rebuilt_sentence[i] = f"[{rebuilt_sentence[i]}]"
    marked_generations = " ".join(rebuilt_sentence)
    print(marked_generations)







#print(df)
