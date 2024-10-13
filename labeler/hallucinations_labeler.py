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


def remove_auxiliary_verbs_tokens(tokens: list, remove_punctuation = False):
    for t in auxiliary_verbs_tokens:
        while t in tokens:
            tokens.remove(t)
    if remove_punctuation:
        for t in punctuations_extended_tokens:
            while t in tokens:
                tokens.remove(t)


def reduce_facts(atoms, method, tokenize_atoms):
    if len(atoms) == 1:
        if not tokenize_atoms:
            return [atoms[0]['text'].lower()], [atoms[0]['label']]
        else:
            return [tokenizer(atoms[0]['text'])[0].ids], [atoms[0]['label']]
    cut_atoms = []
    labels = []

    if method == 'Weak Uniqueness':
        if not tokenize_atoms:
            unique_words = set(atoms[0]['text'].lower().split(' '))
        else:
            unique_words = set(tokenizer(atoms[0]['text'])[0].ids)

        for atom in atoms[1:]:
            if not tokenize_atoms:
                current_unique_words = set(atom['text'].lower().split(' '))
            else:
                current_unique_words = set(tokenizer(atom['text'])[0].ids)
            unique_words = unique_words.intersection(current_unique_words)
        if not tokenize_atoms:
            for atom in atoms:
                atom_sentence = atom['text'].lower()
                for word in unique_words:
                    atom_sentence = atom_sentence.split(" ")
                    atom_sentence = delete_word_from_atom_sentence_and_join(word, atom_sentence)
                cut_atoms.append(atom_sentence)
        else:
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
        if not tokenize_atoms:
            atoms_df = [atoms_df[i].lower() for i in range(atoms_num)]
        else:
            atoms_df = [tokenizer(atoms_df[i])[0].ids for i in range(atoms_num)]
        labels = [pd.DataFrame(atoms)['label'][i] for i in range(atoms_num)]

        atoms_to_labels = {}
        for i in range(atoms_num):
            atoms_to_labels[str(atoms_df[i])] = labels[i]

        if not tokenize_atoms:
            sorted_atoms_by_length = sorted([atoms_df[i].split(' ') for i in range(atoms_num)], key=len)
            merged_sorted_atoms = [" ".join(sorted_atoms_by_length[i]) for i in range(atoms_num)]
        else:
            merged_sorted_atoms = sorted(atoms_df, key=len)
            sorted_atoms_by_length = merged_sorted_atoms
        sorted_labels_by_length = [atoms_to_labels[str(merged_sorted_atoms[i])] for i in range(atoms_num)]

        for atom_ind, atom_text in enumerate(sorted_atoms_by_length):
            for word in atom_text:
                for next_atom_ind in range(atom_ind+1, atoms_num):
                    next_atom = sorted_atoms_by_length[next_atom_ind]
                    if not tokenize_atoms:
                        next_atom_updated = delete_word_from_atom_sentence_and_join(word, next_atom).split(" ")
                    else:
                        next_atom_updated = [token for token in next_atom if token != word]
                    sorted_atoms_by_length[next_atom_ind] = next_atom_updated

        for i in range(len(sorted_atoms_by_length)):
            atom = sorted_atoms_by_length[i]
            if not tokenize_atoms:
                merged_atom = " ".join(atom)
            else:
                merged_atom = atom
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


def find_index_of_fact_in_sentence(fact, sentence, used_tokenizer):
    # Option 1 - Look
    if not used_tokenizer:
        split_fact = split_string(fact)
        split_sentence = split_string(sentence)
    else:
        split_fact = fact
        split_sentence = sentence

    ind_list = []
    for i, fact_word in enumerate(split_fact):
        if fact_word not in split_sentence:
            continue
        word_ind = split_sentence.index(fact_word)
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
        if list2[i-missed_values] != element:
            missing_elements.append(element)
            missed_values += 1
    return missing_elements


def find_pre_sentence(paragraph: str, index):
    sub_paragraph = paragraph[:index]

    ## TODO: Needs refining to include where the other one works
    # if index == 0:
    #     return ""
    # backwards_indices = 1
    # while sub_paragraph[-backwards_indices] in [" ", "\n"]:
    #     backwards_indices += 1
    #     if backwards_indices == len(sub_paragraph):
    #         break
    # x = paragraph[index-backwards_indices: index]
    # return x

    last_dot_index = sub_paragraph.rfind('.')
    last_colun_index = sub_paragraph.rfind(':')
    return paragraph[max(last_dot_index, last_colun_index) + 1: index]



# Example usage:
use_tokenizer = True
show_print = True
remove_punctuation_from_facts = True
#jsonl_file = r"C:\Users\Arik Drori\Desktop\Year3+\NLP\FinalProject\ts_hallucination\answers_gpt4_bio_test_addtional.jsonl"
#jsonl_gpt = r"C:\Users\Arik Drori\Desktop\Year3+\NLP\FinalProject\ts_hallucination\labeler\ChatGPT.jsonl"
#example_output = r"C:\Users\Arik Drori\Desktop\Year3+\NLP\FinalProject\ts_hallucination\labeler\example_output-1.json"
example_output_2 = r"C:\Users\Arik Drori\Desktop\Year3+\NLP\FinalProject\ts_hallucination\fact_checked_data\pythia_2.8_deterministic_fact_checked_bm25.json"
#df_old = read_jsonl(jsonl_gpt)
df = read_our_json(example_output_2)
generations = df['output']
annotations = df['annotations']
for annotation, generation in zip(annotations, generations):
    if annotation is None:
        continue
    total_annotation_length = 0
    hallucination_indices = []
    rebuilt_sentence = []
    original_paragraph = tokenizer(generation)[0].ids
    for sentence in annotation:
        sentence_text_original = sentence['text']


        ## My notes:
        # The split paragraph and the recontructed one from splits ARENT THE SAME, because the splits sometimes change
        # The tokens, or delete some.
        # However, I noticed that the sentence is always found within the text. Good.
        # If this can now turn from letter index into the appropriate token's index, it would be super simple
        # barely an inconvinience
        # to mark the original indices.
        ## My plan: Look backwards from the original sentence, until I find a period, and I will switch the first token
        # in the split sentence with the token that includes all characters since the note
        ## I did notice that sometimes it Misses a letter, and numbers as well.
        ## Well, idc.
        sentence_text_beginning_ind = generation.find(sentence_text_original)
        if sentence_text_beginning_ind == -1:
            # Should never happen.
            raise Exception("FUUUUCCCCKKKKKK, I ASSUMED THE ENTIRE SENTENCE IS INSIDE THE ORIGINAL, FUCK")
        pre_sentence = find_pre_sentence(generation, sentence_text_beginning_ind)
        sentence_text_original = f"{pre_sentence}{sentence_text_original}"

        sentence_text_tokenized = tokenizer(sentence_text_original, padding=True, truncation=True)[0].ids

        sentence_text = sentence_text_original.lower()
        relevancy = sentence['is-relevant']
        atomic_facts = sentence['human-atomic-facts']

        if not use_tokenizer:
            split_sentence = sentence_text.translate(str.maketrans(punctuations_extended, ' ' * len(punctuations_extended))).split(" ")
            split_sentence = [split_sentence[i] for i in range(len(split_sentence)) if split_sentence[i] != ""]
            for word in split_sentence:
                rebuilt_sentence.append(word)
        else:
            for t in sentence_text_tokenized:
                rebuilt_sentence.append(t)

        if atomic_facts is None or len(atomic_facts) == 0:
            continue

        # Reduce facts to weak-uniqueness (Words that appear in ALL atoms are delteted)
        reduced_facts, labels = reduce_facts(atomic_facts, method='Cascading Deletion', tokenize_atoms=use_tokenizer)
        for rfact, label in zip(reduced_facts, labels):
            if label != "NS":
                continue
            if not use_tokenizer:
                rrfact = remove_auxiliary_verbs(rfact)
            else:
                remove_auxiliary_verbs_tokens(rfact, remove_punctuation_from_facts)
                rrfact = rfact
            #print(rrfact)

            if (not use_tokenizer and rrfact != "") or (use_tokenizer and len(rrfact) != 0):
                my_sentence = sentence_text if not use_tokenizer else sentence_text_tokenized
                fact_ind, fact_indices_list = find_index_of_fact_in_sentence(rrfact, my_sentence, use_tokenizer)
                if fact_ind == -1 or fact_indices_list is None:
                    continue
                for ind in fact_indices_list:
                    hallucination_indices.append(ind + total_annotation_length)
                # hallucination_indices.append(fact_ind + total_annotation_length)
        if not use_tokenizer:
            total_annotation_length += len(split_string(sentence_text))
        else:
            total_annotation_length += len(sentence_text_tokenized)

    if show_print:
        if use_tokenizer:
            marked_hallucinations_sentence = []
            missing_tokens = find_missing_elements(original_paragraph, rebuilt_sentence)
            if len(missing_tokens) != 0:
                print("Missing tokens " + str(tokenizer.decode(missing_tokens)))
            for i, t in enumerate(rebuilt_sentence):
                if i in hallucination_indices:
                    marked_hallucinations_sentence.append(60)  # token for [
                    marked_hallucinations_sentence.append(t)  # token
                    marked_hallucinations_sentence.append(62)  # token for ]
                else:
                    marked_hallucinations_sentence.append(t)
            marked_hallucinations_sentence_decoded = tokenizer.decode(marked_hallucinations_sentence)
            print(marked_hallucinations_sentence_decoded)
        else:
            for i in hallucination_indices:
                rebuilt_sentence[i] = f"[{rebuilt_sentence[i]}]"
            marked_generations = " ".join(rebuilt_sentence)
            print(marked_generations)
        # what we want to return is hallucinations indices and sentence text tokenized => Only need to notice that /n are
        # cut in the whole ordeal.








#print(df)
