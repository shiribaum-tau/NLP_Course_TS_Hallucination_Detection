from factscore.factscorer import FactScorer
import nltk
import os
import glob
import pickle
import json

nltk.download('punkt_tab')

KEY = ".key"
DATA_ROOT = os.path.join("..", "gen_data") #"/home/joberant/NLP_2324b/kr/output"
REMOVE_PREFIX = True

def remove_prefix(s, pref):
    if s.startswith(pref):
        return s[len(pref):]
    return s

def create_annotation(text, fact_list):
    annot = dict(text=text)
    annot['is-relevant'] = True
    annot['human-atomic-facts'] = fact_list
    return annot

def main():
    fs = FactScorer(openai_key=KEY)
    dat_files = glob.glob(os.path.join(DATA_ROOT, "*.pkl"))
    out_data = []  # Final json output
    generations = []  # Generations as sent to factscore (no prompt)
    for dat_file in dat_files:
        print(f"Reading {os.path.basename(dat_file)}")
        with open(dat_file, "rb") as f:
            generation_data = pickle.load(f)
        stripped_generation = remove_prefix(generation_data['generated_text'],
                                            generation_data['prompt'])
        full_output = stripped_generation if REMOVE_PREFIX else generation_data['generated_text']
        generations.append(stripped_generation)
        out_data.append(dict(input=generation_data['prompt'],
                             output=full_output,
                             topic=generation_data['entity'],
                             cat=["N/A", "N/A"]))
    topics = [i['topic'] for i in out_data]  # topic that will be sent to factscore

    checked_facts = fs.get_score(topics, generations, gamma=10)
    # Now iterate over the topics
    for out_dat, decisions in zip(out_data, checked_facts['decisions']):
        print(f"Annotating {out_dat['topic']}")
        annotations = []
        # If the generation starts with the prompt,
        # create an atom with it marked as "supported"
        prompt = out_dat['input']
        if (not REMOVE_PREFIX) and out_dat['output'].startswith(prompt):
            annot = create_annotation(prompt, [dict(text=prompt, label="S")])
            annotations.append(annot)
        sent_to_atoms = checked_facts['sentences_to_facts'][out_dat['topic']]
        atom_to_verdict = {i['atom']: i['is_supported'] for i in decisions}
        for sentence, atoms in sent_to_atoms:
            annot = create_annotation(sentence, [{ "text": i, "label": "S" if atom_to_verdict[i] else "NS" } for i in atoms])
            annotations.append(annot)
        out_dat['annotations'] = annotations

        try:
            print("Backing up...")
            with open("pythia_2.8_deterministic_fact_checked.json", "w") as f:
                json.dump(out_data, f)
        except:
            print("Failed to output. Moving on...")

    with open("pythia_2.8_deterministic_fact_checked.json", "w") as f:
        json.dump(out_data, f)

if __name__ == "__main__":
    main()
