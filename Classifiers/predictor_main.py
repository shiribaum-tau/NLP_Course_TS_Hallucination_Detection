import argparse
import torch
import random
import pickle
from sklearn.model_selection import train_test_split
from predictor_utils import *
from ffnn import FeedForwardNN
from bidir_LSTM import BiLSTM
from Conv import Conv
from ResNet import ResNet
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from scipy.special import logit


def main(args):
    # Create an output folder that includes parameters of the run
    model_output_dir = os.path.join(args.output_dir, args.data_gen_model, args.model.lower())

    if not os.path.exists(path=model_output_dir):
        from pathlib import Path
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)

    setup_logger(model_output_dir, args.top_k, 2*args.window_l, args.stride)

    logging.info("--------Starting script with the following arguments--------")
    logging.info(f"Predictor model: {args.model}")
    logging.info(f"The data was generated using the model: {args.data_gen_model}")
    logging.info(f"Top_k: {args.top_k}")
    logging.info(f"Window size: {2 * args.window_l}")
    logging.info(f"Stride: {args.stride}")
    logging.info(f"Start learning rate: {args.learning_rate}")

    print("--------Starting script with the following arguments--------")
    print(f"Predictor model: {args.model}")
    print(f"The data was generated using the model: {args.data_gen_model}")
    print(f"Top_k: {args.top_k}")
    print(f"Window size: {2 * args.window_l}")
    print(f"Stride: {args.stride}")
    print(f"Start learning rate: {args.learning_rate}")

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

    model_dict = {
        'FFN': FeedForwardNN,
        'BiLSTM': BiLSTM,
        'Conv': Conv,
        'ResNet': ResNet
    }

    with open(f'entities_train_{args.data_gen_model}.txt', 'r') as f:
        entities = f.readlines()

    entities_train, entities_val = train_test_split(entities, test_size=0.2, random_state=args.random_seed)
    entities_train = [entity.strip() for entity in entities_train]
    entities_val = [entity.strip() for entity in entities_val]

    # entities_train = ['Jonathan Roy', 'Mateo Correa Magallanes', 'Idris Elba', 'George Eacker', 'George Washington',
    #                   'Tracy Somerset, Duchess of Beaufort', 'Avraham Eilam-Amzallag', 'Winston Churchill',
    #                   'Michael Collins (astronaut)', 'Amr Diab', 'Xi Jinping']

    # entities_train = ["PewDiePie", "Liam Neeson", "Abdul Halik Hudu", "Zhang Yaokun", "Zewde Gebre-Sellassie",
    #                   "Zelma Wilson", "Zeca Pagodinho", "Zamfir Arbore", "Yuu Watase", "Yordanka Donkova",
    #                   "Ylona Garcia", "Yehuda Lindell", "Xi Jinping"]
    # entities_val = ["Aaron Burr", "A. K. Ramanujan", "Abdou Diallo"]

    print("Finished train and val split, num val:", len(entities_val))
    logging.info(f"Finished train and val split, num val: {len(entities_val)}")

    train_logit_windows, train_window_labels = create_data_tensor(
        entities_train, args.train_data_dir, args.top_k, args.window_l * 2, args.stride)

    all_logits_train = np.vstack(train_logit_windows)  # Shape: [total_num_windows, window_size, top_k_logits]
    all_labels_train = np.vstack(train_window_labels)  # Shape: [total_num_windows, 2]

    print("Finished train windows, logits windows shape:", all_logits_train.shape)
    print("Finished train windows, labels windows shape:", all_labels_train.shape)
    logging.info(f"Finished train windows, logits windows shape: {all_logits_train.shape}")
    logging.info(f"Finished train windows, labels windows shape: {all_labels_train.shape}")

    val_logit_windows, val_window_labels = create_data_tensor(
        entities_val, args.train_data_dir, args.top_k, args.window_l * 2, args.stride)

    all_logits_val = np.vstack(val_logit_windows)  # Shape: [total_num_windows, window_size, top_k_logits]
    all_labels_val = np.vstack(val_window_labels)  # Shape: [total_num_windows, 2]

    print("Finished val windows, logits windows shape:", all_logits_val.shape)
    print("Finished val windows, labels windows shape:", all_labels_val.shape)
    logging.info(f"Finished val windows, logits windows shape: {all_logits_val.shape}")
    logging.info(f"Finished val windows, labels windows shape: {all_labels_val.shape}")

    num_hall_windows_train = sum(all_labels_train[:, 1])
    num_hall_windows_val = sum(all_labels_val[:, 1])

    print("Number of hallucination windows in training:", num_hall_windows_train)
    print("Number of non-hallucination windows in training:", all_labels_train.shape[0] - num_hall_windows_train)
    print("Number of hallucination windows in validation:", num_hall_windows_val)
    print("Number of non-hallucination windows in validation:", all_labels_val.shape[0] - num_hall_windows_val)
    logging.info(f"Number of hallucination windows in training: {num_hall_windows_train}")
    logging.info(
        f"Number of non-hallucination windows in training: {all_labels_train.shape[0] - num_hall_windows_train}")
    logging.info(f"Number of hallucination windows in validation: {num_hall_windows_val}")
    logging.info(
        f"Number of non-hallucination windows in validation: {all_labels_val.shape[0] - num_hall_windows_val}")

    input_shape = all_logits_train.shape[1:]

    # Scale data test

    # scaler = FunctionTransformer(logit)
    # all_logits_train = scaler.transform(all_logits_train)
    # all_logits_val = scaler.transform(all_logits_val)

    # all_logits_train = np.log(all_logits_train)
    # all_logits_val = np.log(all_logits_val)
    #
    # for i in range(all_logits_train.shape[0]):
    #     scaler = StandardScaler()
    #     scaler = scaler.fit(all_logits_train[i])
    #     all_logits_train[i] = scaler.transform(all_logits_train[i])
    #
    # for i in range(all_logits_val.shape[0]):
    #     scaler = StandardScaler()
    #     scaler = scaler.fit(all_logits_val[i])
    #     all_logits_val[i] = scaler.transform(all_logits_val[i])

    # print("Finished scaling")
    # logging.info("Finished scaling")

    model_architecture = model_dict[args.model]

    # model_obj = model_architecture(input_shape)
    # model_obj.model.summary()

    steps_per_epoch = int(np.ceil(all_logits_train.shape[0] / args.batch_size))
    callbacks = create_checkpoint_callback(
        model_output_dir, steps_per_epoch, args.top_k, 2 * args.window_l, args.stride, args.saving_interval)

    # Changes - test maybe remove

    pos = num_hall_windows_train + num_hall_windows_val
    neg = (all_labels_train.shape[0] - num_hall_windows_train) + (all_labels_val.shape[0] - num_hall_windows_val)
    initial_bias = np.log([neg / pos, pos / neg])
    initial_weights = f'{model_output_dir}/initial.weights.h5'

    model_obj = model_architecture(input_shape, lr=args.learning_rate, output_bias=initial_bias)
    # model_obj.model.save_weights(initial_weights)

    # model_obj = model_architecture(input_shape, lr=args.learning_rate)
    # model_obj.model.load_weights(initial_weights)
    # model_obj.model.layers[-1].bias.assign([0.0, 0.0])
    model_obj.model.summary()


    print("---- Start training ----")
    logging.info("---- Start training ----")

    history = model_obj.train(
        all_logits_train, all_labels_train, all_logits_val, all_labels_val, args.batch_size, args.epochs, callbacks)

    print("---- Finished training ----")
    logging.info("---- Finished training ----")

    evaluate_val(
        all_logits_val, all_labels_val, all_logits_train, all_labels_train, history, model_obj, model_output_dir, args.top_k, 2 * args.window_l, args.stride)


    # Saving the history of training
    history_file = f'{model_output_dir}/history_{args.model}_k={args.top_k}_w={2*args.window_l}_s={args.stride}.pkl'
    with open(history_file, 'wb') as out_f:
        pickle.dump(history, out_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='directory to save to')
    parser.add_argument('--train_data_dir', type=str, default='./train_dir',
                        help='directory to get the train data files from')
    parser.add_argument('--model', type=str, default='BiLSTM',
                        help='the predictor model architecture')
    parser.add_argument('--data_gen_model', type=str, default='pythia',
                        help='the model that generated the data')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='number of windows per update')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of iterations to train for')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--window_l', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--saving_interval', type=int, default=25)

    args = parser.parse_args()

    main(args)
