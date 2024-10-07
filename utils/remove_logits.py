import os
import glob
import pickle

DATA_ROOT = "/home/joberant/NLP_2324b/kr/output"
OUTPUT_ROOT = "/home/joberant/NLP_2324b/shirabaum/pythia_nologits"

def main():
    dat_files = glob.glob(os.path.join(DATA_ROOT, "*.pkl"))
    for dat_file in dat_files:
        old_basename = os.path.basename(dat_file)
        print(f"Reading {old_basename}")
        new_basename = old_basename.replace(".pkl", "_nologits.pkl")
        new_path = os.path.join(OUTPUT_ROOT, new_basename)
        if os.path.exists(new_path):
            print(f"Skipping {old_basename}. Already done.")
        with open(dat_file, "rb") as f:
            gen_data = pickle.load(f)

        gen_data.pop("logits", None)
        with open(new_path, 'wb') as f:
            pickle.dump(gen_data, f)
        print(f"New file written to {new_path}")



if __name__ == "__main__":
    main()
