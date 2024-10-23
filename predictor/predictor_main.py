import argparse
import torch
import random
from sklearn.model_selection import train_test_split
from predictor_utils import *


def main(args):
    setup_logger()

    logging.info("--------Starting script with the following arguments--------")
    logging.info(f"Predictor model: {args.model}")
    logging.info(f"The data was generated using the model: {args.data_gen_model}")
    logging.info(f"Top_k: {args.top_k}")
    logging.info(f"Window size: {2 * args.window_l}")
    logging.info(f"Stride: {args.stride}")

    print("--------Starting script with the following arguments--------")
    print(f"Predictor model: {args.model}")
    print(f"The data was generated using the model: {args.data_gen_model}")
    print(f"Top_k: {args.top_k}")
    print(f"Window size: {2 * args.window_l}")
    print(f"Stride: {args.stride}")

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

    with open(f'entities_train_{args.data_gen_model}.txt', 'r') as f:
        entities = f.readlines()

    entities_train, entities_val = train_test_split(entities, test_size=0.2, random_state=args.random_seed)
    entities_train = [entity.strip() for entity in entities_train]
    entities_val = [entity.strip() for entity in entities_val]

    train_logit_windows, train_window_labels = create_data_tensor(
        entities_train, args.train_data_dir, args.top_k, args.window_l * 2, args.stride)

    all_logits_train = np.vstack(train_logit_windows)  # Shape: [total_num_windows, window_size, top_k_logits]
    all_labels_train = np.hstack(train_window_labels)  # Shape: [total_num_windows]

    val_logit_windows, val_window_labels = create_data_tensor(
        entities_val, args.train_data_dir, args.top_k, args.window_l * 2, args.stride)

    all_logits_val = np.vstack(val_logit_windows)  # Shape: [total_num_windows, window_size, top_k_logits]
    all_labels_val = np.hstack(val_window_labels)  # Shape: [total_num_windows]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='.',
                        help='directory to save to')
    parser.add_argument('--train_data_dir', type=str, default='./train_dir',
                        help='directory to get the train data files from')
    parser.add_argument('--model', type=str, default='FFN',
                        help='the predictor model architecture')
    parser.add_argument('--data_gen_model', type=str, default='pythia',
                        help='the model that generated the data')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=5000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--window_l', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)

    args = parser.parse_args()

    main(args)
