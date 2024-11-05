import pickle
import json
import torch
import os
import re
import glob
import numpy as np
from text_to_token import get_hallucination_labels
from nltk.tokenize import sent_tokenize

ALL_GENS_DIR = "../../data_exploration/mini_GPT/"
MODEL_NAME = "gpt-4o-mini-2024-07-18"
OUT_DIR = "openai_chosen_token_and_concepts"
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

with open("index2.json", "r") as f:
    ent_index = json.load(f)

reponse_fp = open("response.jsonl", "rb")
raw_resp = reponse_fp.readline().strip()
out = {}
all_concepts = 0
missed = 0
out = {}

while len(raw_resp) != 0:
    resp = json.loads(raw_resp)
    numbers = re.findall(r'\d+', resp['custom_id'])
    entity_idx, sentence_idx = map(int, numbers)
    # print(f"{entity_idx}: {sentence_idx}")
    ent_path = ent_index[str(entity_idx)]
    with open(os.path.join(ALL_GENS_DIR, ent_path), "rb") as f:
        data = pickle.load(f)
        generation = data['generation']

    initials = detect_initials(generation)
    curr_sentences = sent_tokenize(generation)
    # curr_sentences = fix_sentence_splitter(curr_sentences, initials)

    our_sent = curr_sentences[sentence_idx]
    sentence_base = generation.find(our_sent)
    try:
        assert sentence_base != -1, "Sentence not found. Life in garbage."
    except:
        import ipdb;ipdb.set_trace()

    response_text = resp['response']['body']['choices'][0]['message']['content']
    concepts = [i.strip() for i in response_text.split(",")]
    pairs = []
    for concept in concepts:
        all_concepts += 1
        concept_base = our_sent.find(concept)
        if concept_base == -1:
            missed += 1
            # print(f"[[{concept}]] not found in \"{our_sent}\". Skipping.")
        pairs.append((sentence_base+concept_base,sentence_base+concept_base+len(concept)))
    
    entity_name = ent_path.replace("_data.pkl", "")
    # import ipdb;ipdb.set_trace()
    current_entity_pairs = out.get(entity_name, [])
    current_entity_pairs.extend(pairs)
    out[entity_name] = current_entity_pairs
    raw_resp = reponse_fp.readline().strip()
    
print(f"Missed {missed}/{all_concepts} in total")

with open("concept_indices.json", "w") as f:
    json.dump(out, f)
    
    
GENS_WITH_TOKEN_DIR = "../create_openai_batch/openai_with_chosen_token/"

for entity_name, pairs in out.items():
    with open(os.path.join(ALL_GENS_DIR, f"{entity_name}_data.pkl"), "rb") as f:
        data = pickle.load(f)
        generation1 = data['generation']

    with open(os.path.join(GENS_WITH_TOKEN_DIR, f"{entity_name}_gpt-4o-mini-2024-07-18_300_w_chosen_token.pkl"), "rb") as f:
        data = pickle.load(f)

    assert generation1 == data['generated_text'], f"Text doesn't match for {entity_name}"
    
    concept_words = get_hallucination_labels(data['generated_text'], data['tokens'], pairs)
    assert len(concept_words) == len(data['tokens']), f"mismatch on {entity_name}"
    data['concept_words'] = concept_words

    print(f"writing {entity_name}")
    with open(os.path.join(OUT_DIR, f"{entity_name}_{MODEL_NAME}_300_w_chosen_token_and_concepts.pkl"), "wb") as f:
        pickle.dump(data, f)