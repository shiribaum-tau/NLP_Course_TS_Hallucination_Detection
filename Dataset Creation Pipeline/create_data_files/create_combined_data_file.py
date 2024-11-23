import os
import argparse
import logging
import pickle
import pandas as pd
import torch.nn.functional as F
import torch
import glob
import numpy as np


def setup_logger():
    logging.basicConfig(filename='combine_data_log.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        filemode='w'
    )


def main(args):
    setup_logger()

    # Loop through all the biography files with the logits
    for i, filename in enumerate(glob.glob(os.path.join(args.logits_dir, "*.pkl"))):

        print(f"----- Working on file {filename} number {i+1} -----")
        logging.info(f"----- Working on file {filename} number {i+1} -----")

        try:
            logits_file = pd.read_pickle(filename)
            entity = logits_file['entity']

            labels_filename = f"{entity}.pickle"
            labels_file = pd.read_pickle(os.path.join(args.labels_dir, labels_filename))

            # Sort the logits and get the top k (500) sorted
            sorted_logits = torch.sort(logits_file['logits'], dim=1)[0]
            # For Pythia and OPT
            #sorted_probs = F.softmax(sorted_logits, dim=1)

            # For GPT
            sorted_probs = np.exp(sorted_logits)

            top_k_logits = sorted_logits[:, -500:]
            top_k_probs = sorted_probs[:, -500:]


            combined_data = {
                "model": "GPT-4o-mini",
                "tokens": logits_file["tokens"],
                "generation": logits_file["generated_text"],
                "top_k_logits": top_k_logits,
                "top_k_probs": top_k_probs,
                "labels": labels_file["labels"]
            }

            # Save to a pickle file
            output_file = f'{args.output_dir}/{entity}_data.pkl'
            with open(output_file, 'wb') as out_f:
                pickle.dump(combined_data, out_f)

            logging.info("------Saved output to %s-----", output_file)

        except Exception as e:
            print(f"----- Failed on file {filename} -----")
            print(f"Reason for failure: {e}")
            print("moving to next entity")

            logging.info(f"----- Failed on file {filename} -----")
            logging.info(f"Reason for failure: {e}")
            logging.info("moving to next entity")

    print("----- Finished all files -----")
    logging.info("----- Finished all files -----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--logits_dir", type=str, default="./logits_dir")
    parser.add_argument("--labels_dir", type=str, default="./labels_dir")

    args = parser.parse_args()

    main(args)
