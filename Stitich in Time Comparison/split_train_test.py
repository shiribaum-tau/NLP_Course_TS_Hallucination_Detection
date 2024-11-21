import pickle
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from collections import Counter
import scipy as sp
import glob
import shutil

DATA_PATH = os.path.join("..", "data_for_paper_comparison", "minigpt", "openai_fixed")
TEST_ENT_PATH = os.path.join("..", "predictor", "entities_test_openai.txt")
TRAIN_ENT_PATH = os.path.join("..", "predictor", "entities_train_openai.txt")
OUT_DIR_TRAIN = os.path.join("..", "data_for_paper_comparison", "minigpt", "train")
OUT_DIR_TEST = os.path.join("..", "data_for_paper_comparison", "minigpt", "test")
os.makedirs(OUT_DIR_TRAIN, exist_ok=True)
os.makedirs(OUT_DIR_TEST, exist_ok=True)

with open(TEST_ENT_PATH, "rb") as f:
    test_people = set([i.decode('utf-8').strip() for i in f.readlines()])

with open(TRAIN_ENT_PATH, "rb") as f:
    train_people = set([i.decode('utf-8').strip() for i in f.readlines()])


for i, file_path in enumerate(glob.glob(os.path.join(DATA_PATH, "*.pkl"))):
    with open(file_path, "rb") as f:
        data = pandas.read_pickle(f)


    if data['entity'] in test_people:
        shutil.move(file_path, OUT_DIR_TEST)
        print(f"Moved {data['entity']} to test")
    elif data['entity'] in train_people:
        shutil.move(file_path, OUT_DIR_TRAIN)
        print(f"Moved {data['entity']} to train")
    else:
        import ipdb;ipdb.set_trace()