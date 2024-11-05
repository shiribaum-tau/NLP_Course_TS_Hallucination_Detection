import pickle
import json
import torch
import os
import numpy as np

PROMPT = "Question: Tell me a bio of {}."
OUT_DIR = "openai_with_chosen_token"
os.makedirs(OUT_DIR, exist_ok=True)

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
    chosen_token_logprob = []
    for token in token_data:
        tokens.append(token['token'])
        chosen_token_logprob.append(token['logprob'])
        logits.append([i['logprob'] for i in token['top_logprobs']])
    chosen_token_logprob = np.array(chosen_token_logprob)
    assert len(chosen_token_logprob) == len(tokens), f"wtf {out['entity']}"
    out['tokens'] = tokens
    out['logits'] = torch.tensor(logits)
    out['chosen_token_logit'] = torch.tensor(chosen_token_logprob)
    out['chosen_token_prob'] = torch.tensor(np.exp(chosen_token_logprob))
    
    print(f"writing {out['entity']}")
    with open(os.path.join(OUT_DIR, f"{out['entity']}_{model_name}_300_w_chosen_token.pkl"), "wb") as f:
        pickle.dump(out, f)
    raw_bio = bio_fp.readline().strip()
