import pickle
import json
import torch
import os

PROMPT = "Question: Tell me a bio of {}."
OUT_DIR = "openai_bios"
os.makedirs(OUT_DIR, exist_ok=True)

with open("Aaron Burr_M=2.8b_RS=42_D=True_T=1.0_TP=0.9_len=300.pkl", "rb") as f:
    orig = pickle.load(f)

bio_fp = open("all_300_bios.jsonl", "rb")
raw_bio = bio_fp.readline().strip()
while len(raw_bio) != 0:
    out = {}
    bio = json.loads(raw_bio)
    out['entity'] = bio['custom_id']
    out['prompt'] = PROMPT.format(out['entity'])
    model_name = bio['response']['body']['model']

    response_obj = bio['response']['body']['choices'][0]
    out['generated_text'] = response_obj['message']['content']
    token_data = response_obj['logprobs']['content']
    tokens = []
    logits = []
    for token in token_data:
        tokens.append(token['token'])
        logits.append([i['logprob'] for i in token['top_logprobs']])
    out['tokens'] = tokens
    out['logits'] = torch.tensor(logits)
    print(f"writing {out['entity']}")
    with open(os.path.join(OUT_DIR, f"{out['entity']}_{model_name}_300.pkl"), "wb") as f:
        pickle.dump(out, f)
    raw_bio = bio_fp.readline().strip()
