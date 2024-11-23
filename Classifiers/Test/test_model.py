import tensorflow as tf
import keras
import argparse
import logging
from predictor_utils import *

tfk = tf.keras

def setup_logger(save_dir, model):
    logging.basicConfig(filename=f'{save_dir}/test_predictor_log_model={model}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        filemode='w'
    )


def main(args):
    setup_logger(args.output_dir, args.model_name)

    logging.info("--------Starting script with the following arguments--------")
    logging.info(f"Predictor model: {args.model_name}")
    logging.info(f"The data was generated using the model: {args.data_gen_model}")
    logging.info(f"Top_k: {args.top_k}")
    logging.info(f"Window size: {2 * args.window_l}")
    logging.info(f"Stride: {args.stride}")

    print("--------Starting script with the following arguments--------")
    print(f"Predictor model: {args.model_name}")
    print(f"The data was generated using the model: {args.data_gen_model}")
    print(f"Top_k: {args.top_k}")
    print(f"Window size: {2 * args.window_l}")
    print(f"Stride: {args.stride}")

    with open(f'entities_test_{args.data_gen_model}.txt', 'r') as f:
        entities_test = f.readlines()

    entities_test = [entity.strip() for entity in entities_test]

    # entities_test = ['Jonathan Roy', 'Mateo Correa Magallanes', 'Idris Elba', 'George Eacker', 'George Washington',
    #                 'Tracy Somerset, Duchess of Beaufort', 'Avraham Eilam-Amzallag', 'Winston Churchill',
    #                     'Michael Collins (astronaut)', 'Amr Diab', 'Xi Jinping']

    print("Finished loading test data, num test:", len(entities_test))
    logging.info(f"Finished loading test data, num test: {len(entities_test)}")

    test_logit_windows, test_window_labels = create_data_tensor(
        entities_test, args.test_data_dir, args.top_k, args.window_l * 2, args.stride)

    all_logits_test = np.vstack(test_logit_windows)  # Shape: [total_num_windows, window_size, top_k_logits]
    all_labels_test = np.vstack(test_window_labels)  # Shape: [total_num_windows, 2]

    print("Finished test windows, logits windows shape:", all_logits_test.shape)
    print("Finished test windows, labels windows shape:", all_labels_test.shape)
    logging.info(f"Finished test windows, logits windows shape: {all_logits_test.shape}")
    logging.info(f"Finished test windows, labels windows shape: {all_labels_test.shape}")

    num_hall_windows_test = sum(all_labels_test[:, 1])

    print("Number of hallucination windows in test:", num_hall_windows_test)
    print("Number of non-hallucination windows in test:", all_labels_test.shape[0] - num_hall_windows_test)
    logging.info(f"Number of hallucination windows in test: {num_hall_windows_test}")
    logging.info(
        f"Number of non-hallucination windows in test: {all_labels_test.shape[0] - num_hall_windows_test}")

    model_obj = keras.models.load_model(args.model_file)
    model_obj.summary()

    print("---- Start evaluating ----")
    logging.info("---- Start evaluating ----")

    evaluate_test(all_logits_test, all_labels_test, model_obj, args.model_name, args.output_dir, args.data_gen_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='directory to save to')
    parser.add_argument('--test_data_dir', type=str, default='./train_dir',
                        help='directory to get the test data files from')
    parser.add_argument('--model_file', type=str, default='./output/best models/bilstm_best_model_k=20_w=6_s=1.keras',
                        help='the trained predictor model file')
    parser.add_argument('--model_name', type=str, default='BiLSTM',
                        help='the trained predictor model name')
    parser.add_argument('--data_gen_model', type=str, default='pythia',
                        help='the model that generated the data')
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--window_l', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)

    args = parser.parse_args()

    main(args)