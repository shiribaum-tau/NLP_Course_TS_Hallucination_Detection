import numpy as np
from transformers import AutoTokenizer, GPT2Tokenizer

GENERATION = "Irina Shayk, born Irina Shaykhlislamova on January 6, 1986, in Yemanzhelinsk, Russia, is a renowned Russian supermodel and actress. She rose to fame after being featured in the Sports Illustrated Swimsuit Issue and was named its Rookie of the Year in 2007. Shayk has graced the covers of numerous fashion magazines, including Vogue, Harper's Bazaar, and Elle, and has worked with high-profile brands such as Victoria's Secret, Givenchy, and Versace.\n\n"
TOKENS = ["Ir","ina"," Shay","k",","," born"," Ir","ina"," Shay","k","hl","is","lam","ova"," on"," January"," ","6",","," ","198","6",","," in"," Yem","anz","hel","insk",","," Russia",","," is"," a"," renowned"," Russian"," super","model"," and"," actress","."," She"," rose"," to"," fame"," after"," being"," featured"," in"," the"," Sports"," Illustrated"," Sw","ims","uit"," Issue"," and"," was"," named"," its"," Rookie"," of"," the"," Year"," in"," ","200","7","."," Shay","k"," has"," gr","aced"," the"," covers"," of"," numerous"," fashion"," magazines",","," including"," Vogue",","," Harper","'s"," Bazaar",","," and"," Elle",","," and"," has"," worked"," with"," high","-profile"," brands"," such"," as"," Victoria","'s"," Secret",","," Given","chy",","," and"," Vers","ace",".\n\n"]
# GENERATION = 'Question: Tell me a bio of Aaron Burr.Q: Aaron Burr was born in 1756 in Virginia. He was the son of a wealthy planter. He was a brilliant student and was admitted to the College of New Jersey in 1770. He was a member of the Continental Congress and was a'
# TOKENS = [23433,    27, 19906,   479,   247,  9015,   273, 22234,  7634,    83,
#             15,     0,     0,     0,     0,     0,     0,     0,    50,    27,
#          22234,  7634,    83,   369,  5686,   275,  1722,  3208,   275,  9385,
#             15,   754,   369,   253,  3347,   273,   247, 20193,  2098,   350,
#             15,   754,   369,   247, 15925,  5974,   285,   369,  8176,   281,
#            253,  6822,   273,  1457,  8911,   275,  1722,  1967,    15,   754,
#            369,   247,  3558,   273,   253, 33911,  5759,   285,   369,   247]

PAIRS = [(3,12), (20, 23), (64, 68)]


tokenizer = AutoTokenizer.from_pretrained(
"facebook/opt-2.7b")
#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
tokenizer.pad_token = tokenizer.eos_token


def intervals_intersect(pair1, pair2):
    a1, b1 = pair1
    a2, b2 = pair2
    return a1 <= b2 and a2 <= b1


def get_hallucation_indices(generation, tokens, pairs):
    if type(tokens[0]) is str:
        decoded_tokens = tokens
    else:
        decoded_tokens = [tokenizer.decode([token], skip_special_tokens = True) for token in tokens]

    assert "".join(decoded_tokens) == generation, "Tokens do not make the text."

    text_idx = 0
    pairs_idx = 0

    hallucation_tokens = []

    if len(pairs) == 0:
        return hallucation_tokens

    # re returns the index after the end of the word.
    pairs = [(a, b - 1) for a, b in pairs]

    for token_idx, dec_token in enumerate(decoded_tokens):
        idx_at_start = text_idx
        text_idx += len(dec_token)
        current_span = (idx_at_start, text_idx - 1)


        while pairs_idx < len(pairs) - 1 and pairs[pairs_idx][1] < current_span[0]:
            # we've gone past the current hallucination. Go to the next one.
            pairs_idx += 1

        current_pair = pairs[pairs_idx]
        if intervals_intersect(current_pair, current_span):
            hallucation_tokens.append(token_idx)
        # If we didn't enter the if then the word is before the hallucination. Move to the next word.

        if pairs_idx >= len(pairs): # No more pairs
            break

    return hallucation_tokens

def get_hallucination_labels(generation, tokens, pairs):
    indices = get_hallucation_indices(generation, tokens, pairs)
    labels = np.zeros(len(tokens))
    labels[indices] = 1
    return labels


if __name__ == "__main__":
    hallucination_tokens = get_hallucation_indices(GENERATION, TOKENS, PAIRS)
    if type(TOKENS[0]) is str:
        out = TOKENS.copy()
    else:
        out = [tokenizer.decode([token], skip_special_tokens = True) for token in TOKENS]

    for hal in hallucination_tokens:
        out[hal] = f"[{out[hal]}]"

    print("".join(out))

    labels = get_hallucination_labels(GENERATION, TOKENS, PAIRS)
    print(labels)
