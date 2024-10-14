import pandas as pd
import json
import numpy as np
import re
import matplotlib.pyplot as plt
from transformers import GPTNeoXForCausalLM, AutoTokenizer

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

PUNCT_EXTRA = "<PUNC>"


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
        return [atoms[0]['text']], [atoms[0]['label']]
    cut_atoms = []

    if method == 'Weak Uniqueness':
        unique_words = set(atoms[0]['text'].split(' '))

        for atom in atoms[1:]:
            current_unique_words = set(atom['text'].split(' '))
            unique_words = unique_words.intersection(current_unique_words)
        for atom in atoms:
            atom_sentence = atom['text']
            for word in unique_words:
                atom_sentence = atom_sentence.split(" ")
                atom_sentence = delete_word_from_atom_sentence_and_join(word, atom_sentence)
            cut_atoms.append(atom_sentence)
        labels = [atom['label'] for atom in atoms]
        return cut_atoms, labels

    if method == "Cascading Deletion":
        atoms_df = pd.DataFrame(atoms)['text']
        atoms_num = len(atoms_df)
        atoms_df = [atoms_df[i] for i in range(atoms_num)]
        labels = [pd.DataFrame(atoms)['label'][i] for i in range(atoms_num)]

        atoms_to_labels = {}
        for i in range(atoms_num):
            atoms_to_labels[str(atoms_df[i])] = labels[i]

        sorted_atoms_by_length = sorted([atoms_df[i].split(' ') for i in range(atoms_num)], key=len)
        merged_sorted_atoms = [" ".join(sorted_atoms_by_length[i]) for i in range(atoms_num)]
        sorted_labels_by_length = [atoms_to_labels[str(merged_sorted_atoms[i])] for i in range(atoms_num)]

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


def split_string(string: str):
    split = string.translate(str.maketrans(punctuations_extended, ' '*len(punctuations_extended))).split(" ")
    return [split[i] for i in range(len(split)) if split[i] != ""]


def find_index_of_fact_in_sentence(fact, sentence):
    split_fact = split_string(fact)

    ind_list = []
    for fact_word in split_fact:
        appearances_list = [(m.start(), m.end()) for m in re.finditer(rf'\b{fact_word}\b', sentence)]
        for couple in appearances_list:
            ind_list.append(couple)
    return ind_list



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


def get_full_sentence_start_and_end_index(partial_sentence, generation, previous_sentence_end_index):
    beginning_ind = generation.find(partial_sentence, previous_sentence_end_index + 1)
    if beginning_ind == -1:
        # Should never happen.
        raise Exception("FUUUUCCCCKKKKKK, I ASSUMED THE ENTIRE SENTENCE IS INSIDE THE ORIGINAL, FUCK")
    end_index = beginning_ind + len(partial_sentence) - 1
    return end_index, beginning_ind


def rebuild_original_paragraph(so_far_rebuilt_paragraph, sentence_to_rebuild):
    split_sentence = sentence_to_rebuild.translate(
        str.maketrans(punctuations_extended, ' ' * len(punctuations_extended))).split(" ")
    split_sentence = [split_sentence[i] for i in range(len(split_sentence)) if split_sentence[i] != ""]
    for word in split_sentence:
        so_far_rebuilt_paragraph.append(word)


def find_hallucination_indices_in_sentence(facts, labels, sentence):
    sentence_hallucinations_indices = []
    for rfact, label in zip(facts, labels):
        if label != "NS":
            continue
        rrfact = remove_auxiliary_verbs(rfact)
        # print(rrfact)

        if rrfact != "":
            fact_indices_couple_list = find_index_of_fact_in_sentence(rrfact, sentence)
            if len(fact_indices_couple_list) == 0:
                continue
            for couple in fact_indices_couple_list:
                sentence_hallucinations_indices.append(couple)
    return sentence_hallucinations_indices


def replace_punctuation(text):
    # Create a regex pattern that matches any punctuation character
    pattern = f"[{re.escape(punctuations_extended)}]"
    # Replace each punctuation mark with " {PUNCT_EXTRA}P{PUNCT_EXTRA} "
    replaced_text = re.sub(pattern, f" {PUNCT_EXTRA}\g<0>{PUNCT_EXTRA} ", text)
    return replaced_text


def insert_letter(string, letter, index):
    return string[:index] + letter + string[index:]


def chatgpt_finds_couples(atomic_facts, sentence_text):
    raise NotImplementedError("lol noob get not implemented")


ROM_ALGO_BEST_ALGO = True
show_print = True
remove_punctuation_from_facts = True
example_output_2 = r"C:\Users\Arik Drori\Desktop\Year3+\NLP\FinalProject\ts_hallucination\fact_checked_data\pythia_2.8_deterministic_fact_checked_gtr.json"
df = read_our_json(example_output_2)
generations = df['output']
annotations = df['annotations']

for annotation, generation in zip(annotations, generations):
    if annotation is None:
        continue
    hallucination_indices_couples = []
    last_sentence_end_index = -1
    for sentence in annotation:
        sentence_text = sentence['text']
        last_sentence_end_index, this_sentence_start_ind = get_full_sentence_start_and_end_index(sentence_text, generation,
                                                                                               last_sentence_end_index)

        relevancy = sentence['is-relevant']
        atomic_facts = sentence['human-atomic-facts']

        if atomic_facts is None or len(atomic_facts) == 0:
            continue

        if ROM_ALGO_BEST_ALGO:
            # Reduce facts to weak-uniqueness (Words that appear in ALL atoms are delteted), or Cascading Deletion (Every
            # word seen in a fact will be deleted in the following ones)
            # And then find where those hallucinataions are in the given sentence
            reduced_facts, labels = reduce_facts(atomic_facts, method='Cascading Deletion')
            sentence_hallucinations_couples = find_hallucination_indices_in_sentence(reduced_facts, labels, sentence_text)
        else:
            sentence_hallucinations_couples = chatgpt_finds_couples(atomic_facts, sentence_text)

        for couple in sentence_hallucinations_couples:
            s, e = couple
            hallucination_indices_couples.append((s + this_sentence_start_ind, e + this_sentence_start_ind))

    if show_print:
        hallucination_indices_couples_from_end = sorted(list(set(hallucination_indices_couples)), key=lambda x: -x[1])
        generation_to_change = generation
        for couple in hallucination_indices_couples_from_end:
            s, e = couple
            generation_to_change = insert_letter(generation_to_change, "]", e)
            generation_to_change = insert_letter(generation_to_change, "[", s)
        print(generation_to_change)
        # what we want to return is hallucinations indices and sentence text tokenized => Only need to notice that /n are
        # cut in the whole ordeal.


    


#print(df)
