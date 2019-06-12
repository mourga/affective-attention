import csv
import glob
import html
import json
import os
import pickle
import random
from collections import Counter

import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from sys_config import DATA_DIR
from utils.smt import smt_dataset


def read_amazon(file):
    reviews = []
    summaries = []
    labels = []

    with open(file) as f:
        for line in f:
            entry = json.loads(line)
            reviews.append(entry["reviewText"])
            summaries.append(entry["summary"])
            labels.append(entry["overall"])

    return reviews, summaries, labels


def read_semeval():
    def read_dataset(d):
        with open(os.path.join(DATA_DIR, "semeval", "E-c",
                               "E-c-En-{}.txt".format(d))) as f:
            reader = csv.reader(f, delimiter='\t')
            labels = next(reader)[2:]

            _X = []
            _y = []
            for row in reader:
                _X.append(row[1])
                _y.append([int(x) for x in row[2:]])
            return _X, _y

    X_train, y_train = read_dataset("train")
    X_dev, y_dev = read_dataset("dev")
    X_test, y_test = read_dataset("test")

    X_train = X_train + X_test
    y_train = y_train + y_test

    return X_train, numpy.array(y_train), X_dev, numpy.array(y_dev)


def read_emoji(split=0.1, min_freq=100, max_ex=1000000, top_n=None):
    X = []
    y = []
    with open(os.path.join(DATA_DIR, "emoji", "emoji_1m.txt")) as f:
        for i, line in enumerate(f):
            if i > max_ex:
                break
            emoji, text = line.rstrip().split("\t")
            X.append(text)
            y.append(emoji)

    counter = Counter(y)
    top = set(l for l, f in counter.most_common(top_n) if f > min_freq)

    data = [(_x, _y) for _x, _y in zip(X, y) if _y in top]

    total = len(data)

    data = [(_x, _y) for _x, _y in data if
            random.random() > counter[_y] / total]

    X = [x[0] for x in data]
    y = [x[1] for x in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=split,
                                                        stratify=y,
                                                        random_state=0)

    lb = LabelBinarizer()
    lb.fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)

    return X_train, y_train, X_test, y_test


def clean_text(text):
    """
    Remove extra quotes from text files and html entities
    Args:
        text (str): a string of text

    Returns: (str): the "cleaned" text

    """
    text = text.rstrip()

    if '""' in text:
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        text = text.replace('\\""', '"')
        text = text.replace('""', '"')

    text = text.replace('\\""', '"')

    text = html.unescape(text)
    text = ' '.join(text.split())
    return text


def parse_file(file):
    """
    Read a file and return a dictionary of the data, in the format:
    tweet_id:{sentiment, text}
    """

    data = {}
    lines = open(file, "r", encoding="utf-8").readlines()
    for line_id, line in enumerate(lines):
        columns = line.rstrip().split("\t")
        tweet_id = columns[0]
        sentiment = columns[1]
        text = columns[2:]
        text = clean_text(" ".join(text))
        data[tweet_id] = (sentiment, text)
    return data


def load_data_from_dir(path):
    FILE_PATH = os.path.dirname(__file__)
    files_path = os.path.join(FILE_PATH, path)

    files = glob.glob(files_path + "/**/*.tsv", recursive=True)
    files.extend(glob.glob(files_path + "/**/*.txt", recursive=True))

    data = {}  # use dict, in order to avoid having duplicate tweets (same id)
    for file in files:
        file_data = parse_file(file)
        data.update(file_data)
    return list(data.values())

def load_irony_dataset(file):
    data = []
    lines = open(file, "r", encoding="utf-8").readlines()
    for line_id, line in enumerate(lines):
        if line_id != 0:
            columns = line.rstrip().split("\t")
            tweet_id = columns[0]
            label = columns[1]
            text = columns[2:]
            text = clean_text(" ".join(text))
            data.append((label, text))
    return data

def load_sst_dataset(fine_grained):
    print("Loading SST dataset...")

    train_path = os.path.join(DATA_DIR, 'sst_fine_grained', 'train.pickle')
    dev_path = os.path.join(DATA_DIR, 'sst_fine_grained', 'dev.pickle')
    test_path = os.path.join(DATA_DIR, 'sst_fine_grained', 'test.pickle')

    # train = load_pickle(train_path)
    # dev = load_pickle(dev_path)
    # test = load_pickle(test_path)

    train = smt_dataset(train=True, fine_grained=fine_grained)
    dev = smt_dataset(dev=True, fine_grained=fine_grained)
    test = smt_dataset(test=True, fine_grained=fine_grained)

    X_train, y_train = load_sst_X_y(train)
    X_dev, y_dev = load_sst_X_y(dev)
    X_test, y_test = load_sst_X_y(test)

    # tuple =  torchtext.datasets.SST.splits(path, text_field, label_field, subtrees=False, fine_grained=True)

    # return X_train+X_dev, y_train+y_dev, X_test, y_test
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_sst_X_y(data):
    X = []
    y = []
    for i in range(0, len(data)):
        text = data[i]['text']
        text = clean_text(text)
        label = data[i]['label']
        X.append(text)
        y.append(label)
    return X, y

def load_subj_or_mr_dataset(file, label):
    data = []
    lines = open(file, "r", encoding = "ISO-8859-1").readlines()
    for line_id, line in enumerate(lines):
        text = line.rstrip()
        label = label
        data.append((label, text))
    return data


def load_scv1(path):
    files = glob.glob(path+"/*.txt")
    data = []

    for file in files:
        text = open(file, "r", encoding="utf-8").readlines()
        data.append(text[0])
    return data

def load_affective(path):

    return

def load_trec(file):
    data = []
    lines = open(file, "rb").readlines()
    for line_id, line in enumerate(lines):
        if line_id != 0:
            line = str(line)
            label = line[2]
            text = line[3:-3]
            data.append((label, text))
    return data





