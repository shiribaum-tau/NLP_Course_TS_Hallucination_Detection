import argparse
import random
import torch
import numpy as np
from pythia_utils import *
import pickle


def main(args):

    setup_logger()
    logging.info("--------Starting script with the following arguments--------")
    logging.info(f"Model: {args.model}")
    logging.info(f"Deterministic: {args.deterministic}")
    if not args.deterministic:
        logging.info(f"Temperature: {args.temperature}")
        logging.info(f"top_p: {args.top_p}")

    print("--------Starting script with the following arguments--------")
    print(f"Model: {args.model}")
    print(f"Deterministic: {args.deterministic}")
    if not args.deterministic:
        print(f"Temperature: {args.temperature}")
        print(f"top_p: {args.top_p}")

    # only torch 2.4 and upper support mps, change when running on different versions
    if args.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    print("Using device: ", device)
    logging.info(f"Using device: {device}")

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Create model
    model = load_model(args.model, device)
    tokenizer = load_tokenizer(args.model)

    logging.info("Loaded model successfully")

    # Load entities and create prompts
    with open('entities.txt', 'r') as f:
        entities = f.readlines()

    prompts = [entity_to_prompt(entity.strip()) for entity in entities]
    prompts_tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    for i, prompt in enumerate(prompts):
        print(f"----- Working on current prompt {prompt} number {i+1} -----")
        logging.info(f"----- Working on current prompt {prompt} number {i+1} -----")

        try:
            with torch.no_grad():
                prompt_ids = prompts_tokenized["input_ids"][i].unsqueeze(0).to(device)
                attention_mask = prompts_tokenized["attention_mask"][i].unsqueeze(0).to(device)

                if args.deterministic:
                    outputs = model.generate(
                        prompt_ids,
                        attention_mask=attention_mask,  # Pass the attention mask
                        max_length=args.gen_len,  # Maximum length of generated text
                        num_return_sequences=1,  # Number of outputs to generate
                        do_sample=False,  # Enable sampling for non-deterministic outputs
                        output_scores=True,  # Return logits (scores) for each generated token
                        output_logits=True,  # Return logits (probs) for each generated token
                        return_dict_in_generate=True,  # Return a dictionary with additional data
                        pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
                    )
                else:
                    outputs = model.generate(
                        prompt_ids,
                        attention_mask=attention_mask,  # Pass the attention mask
                        max_length=args.gen_len,  # Maximum length of generated text
                        num_return_sequences=1,  # Number of outputs to generate
                        temperature=args.temperature,  # Balanced temperature to avoid extreme values
                        top_p=args.top_p,  # Nucleus sampling parameter for diversity
                        do_sample=True,  # Enable sampling for non-deterministic outputs
                        output_scores=True,  # Return logits (scores) for each generated token
                        output_logits=True,  # Return logits (probs) for each generated token
                        return_dict_in_generate=True,  # Return a dictionary with additional data
                        pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
                    )

                # Extract generated output tokens and decode to text and save to file
                generated_ids = outputs['sequences']
                output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Extract generated logits and tokens and save to file
                logits = torch.stack(outputs.logits, dim=1)[0]
                logits = logits.detach().half().to('cpu')

                tokens = generated_ids[0].detach().to(torch.int32).to('cpu')

                # Save generated text, logits, prompt, and entity as pickle
                output_data = {'entity': entities[i].strip(),
                               'prompt': prompt,
                               'generated_text': output_text,
                               'tokens': tokens,
                               'logits': logits}


                # Save to a pickle file
                output_file = f'{args.output_dir}/{entities[i].strip()}_M={args.model.split("-")[1]}_RS={args.random_seed}_D={args.deterministic}_T={args.temperature}_TP={args.top_p}_len={args.gen_len}.pkl'
                with open(output_file, 'wb') as out_f:
                    pickle.dump(output_data, out_f)

                logging.info("------Saved output to %s-----", output_file)

        except Exception as e:
            print(f"----- Failed on prompt {prompt} -----")
            print(f"Reason for failure: {e}")
            print("moving to next entity")

            logging.info(f"----- Failed on prompt {prompt} -----")
            logging.info(f"Reason for failure: {e}")
            logging.info("moving to next entity")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--model", type=str, default="facebook/opt-2.7b")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--gen_len", type=int, default=300)

    args = parser.parse_args()

    main(args)


