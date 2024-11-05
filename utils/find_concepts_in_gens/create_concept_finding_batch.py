import os
import glob
import pickle
import re
import numpy as np
import json
from nltk.tokenize import sent_tokenize


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


# with open("example_batch.jsonl", "r") as f:
#     ex = f.read().strip()

ALL_GENS = glob.glob("../../data_exploration/mini_GPT/*.pkl")

all_requests = []

for gen_id, genpath in enumerate(ALL_GENS):
    with open(genpath, "rb") as f:
        data = pickle.load(f)
        generation = data['generation']

    initials = detect_initials(generation)

    curr_sentences = sent_tokenize(generation)

    curr_sentences = fix_sentence_splitter(curr_sentences, initials)

    for sent_id, sent in enumerate(curr_sentences):
        json_line = {
            "custom_id": f"req_{gen_id}_{sent_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": f"{sent}\nIdentify all the important keyphrases from the above sentence and return a comma separated list."
                    }
                ],
                "max_completion_tokens": 300
            }
        }

        all_requests.append(json.dumps(json_line))

joined_data = "\n".join(all_requests)

with open('big_concept_batch.jsonl', 'w', encoding='utf-8') as file:
    file.write(joined_data)