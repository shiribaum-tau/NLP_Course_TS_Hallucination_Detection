import pickle
import json
import torch
import os
import re
import glob
import numpy as np
from text_to_token import get_hallucination_labels
from nltk.tokenize import sent_tokenize


MODEL_NAME = "pythia"
ALL_GENS_DIR = f"../../data_for_paper_comparison/{MODEL_NAME}/Test/"

OUT_DIR = f"../../data_for_paper_comparison/{MODEL_NAME}/Test_w_chosen_token_and_concepts"
os.makedirs(OUT_DIR, exist_ok=True)

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences

reponse_fp = open(f"{MODEL_NAME}_concept_word_response.jsonl", "rb")
raw_resp = reponse_fp.readline().strip()
out = {}
all_concepts = 0
missed = 0
out = {}

while len(raw_resp) != 0:
    resp = json.loads(raw_resp)
    entity_name = resp['custom_id'].split("_")[-1]
    numbers = re.findall(r'\d+', resp['custom_id'])
    entity_idx, sentence_idx = map(int, numbers)
    # print(f"{entity_idx}: {sentence_idx}")
    with open(os.path.join(ALL_GENS_DIR, f"{entity_name}_data.pkl"), "rb") as f:
        data = pickle.load(f)
        generation = data['generation']

    initials = detect_initials(generation)
    curr_sentences = sent_tokenize(generation)
    # curr_sentences = fix_sentence_splitter(curr_sentences, initials)

    our_sent = curr_sentences[sentence_idx]
    base_idx = len(" ".join(curr_sentences[:sentence_idx]))
    sentence_base = generation.find(our_sent, max(0, base_idx-1))
    try:
        assert sentence_base != -1, "Sentence not found. Life in garbage."
    except:
        import ipdb;ipdb.set_trace()

    response_text = resp['response']['body']['choices'][0]['message']['content']
    concepts = [i.strip() for i in response_text.split(",")]
    pairs = []
    for concept in concepts:
        all_concepts += 1
        concept_base = our_sent.lower().find(concept.lower().rstrip('.'))
        if concept_base == -1:
            missed += 1
            print(f"[[{concept}]] not found in \"{our_sent}\". Skipping.")
        else:
            pairs.append((sentence_base+concept_base,sentence_base+concept_base+len(concept)))
    
    # import ipdb;ipdb.set_trace()
    current_entity_pairs = out.get(entity_name, [])
    current_entity_pairs.extend(pairs)
    out[entity_name] = current_entity_pairs
    raw_resp = reponse_fp.readline().strip()
    
print(f"Missed {missed}/{all_concepts} in total")

with open(f"{MODEL_NAME}_concept_indices.json", "w") as f:
    json.dump(out, f)


for entity_name, pairs in out.items():
    with open(os.path.join(ALL_GENS_DIR, f"{entity_name}_data.pkl"), "rb") as f:
        data = pickle.load(f)


    concept_words = get_hallucination_labels(data['generation'], data['tokens'], pairs)
    assert len(concept_words) == len(data['tokens']), f"mismatch on {entity_name}"
    
    data['concept_words'] = concept_words
    data['top_k_logits'] = data['top_k_logits'][:, -20:]
    data['top_k_probs'] = data['top_k_probs'][:, -20:]
    data['chosen_token_prob'] = data['top_k_probs'][:, -1]

    print(f"writing {entity_name}")
    with open(os.path.join(OUT_DIR, f"{entity_name}_{MODEL_NAME}_300_w_chosen_token_and_concepts.pkl"), "wb") as f:
        pickle.dump(data, f)