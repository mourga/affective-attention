import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sys_config import DATA_DIR
from utils.data_parsing import load_data_from_dir, load_irony_dataset, load_sst_dataset, \
    load_subj_or_mr_dataset, load_scv1, load_pickle, load_trec
from sklearn.model_selection import train_test_split

def semeval_2017A():
    train = load_data_from_dir(os.path.join(DATA_DIR, 'semeval_2017_4A/train'))
    X = [obs[1] for obs in train]
    y = [obs[0] for obs in train]

    test = load_data_from_dir(os.path.join(DATA_DIR, 'semeval_2017_4A/test'))
    X_test = [obs[1] for obs in test]
    y_test = [obs[0] for obs in test]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    # transform labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def irony():
    train_file = os.path.join(DATA_DIR, 'task3/train/SemEval2018-T3-train-taskB_emoji.txt')
    train = load_irony_dataset(train_file)
    X = [obs[1] for obs in train]
    y = [obs[0] for obs in train]

    test_file = os.path.join(DATA_DIR, 'task3/gold/SemEval2018-T3_gold_test_taskB_emoji.txt')
    test = load_irony_dataset(test_file)
    X_test = [obs[1] for obs in test]
    y_test = [obs[0] for obs in test]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    # transform labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def sst(fine_grained):
    X_train, y_train, X_val, y_val, X_test, y_test = load_sst_dataset(fine_grained)

    # transform labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    # return X_train, y_train, X_test, y_test
    return X_train, y_train, X_val, y_val, X_test, y_test

def subj():
    subj_path = os.path.join(DATA_DIR, 'subj/quote.tok.gt9.5000')
    obj_path = os.path.join(DATA_DIR, 'subj/plot.tok.gt9.5000')

    obj_data = load_subj_or_mr_dataset(file=obj_path, label="obj")
    subj_data = load_subj_or_mr_dataset(file=subj_path, label="subj")
    data = obj_data + subj_data

    X = [obs[1] for obs in data]
    y = [obs[0] for obs in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # transform labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test

def mr():
    pos_path = os.path.join(DATA_DIR, 'rt-polaritydata/rt-polarity.pos')
    neg_path = os.path.join(DATA_DIR, 'rt-polaritydata/rt-polarity.neg')

    pos_data = load_subj_or_mr_dataset(file=pos_path, label="positive")
    neg_data = load_subj_or_mr_dataset(file=neg_path, label="negative")
    data = pos_data + neg_data

    X = [obs[1] for obs in data]
    y = [obs[0] for obs in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, stratify=y)

    # transform labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test

def scv1():
    not_sarc_path = os.path.join(DATA_DIR, 'SCV1', 'notsarc')
    sarc_path = os.path.join(DATA_DIR, 'SCV1', 'sarc')

    not_sarc = load_scv1(not_sarc_path)
    sarc = load_scv1(sarc_path)

    not_sarc_label = ['not_sarcastic' for x in not_sarc]
    sarc_label = ['sarcastic' for x in sarc]

    X = not_sarc + sarc
    y = not_sarc_label + sarc_label

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,
                                                        test_size=995, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                        test_size=0.2, stratify=y_train_val)
    # transform labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def affective():
    # train_path = os.path.join(DATA_DIR, 'AffectiveTextSemeval2007', 'AffectiveText.trial')
    # test_path = os.path.join(DATA_DIR, 'AffectiveTextSemeval2007', 'AffectiveText.test')

    path = os.path.join(DATA_DIR, 'AffectiveTextSemeval2007', 'raw.pickle')
    data = load_pickle(path)
    train = [(data['texts'][x], data['info'][x]) for x in data['train_ind']]
    val = [(data['texts'][x], data['info'][x]) for x in data['val_ind']]
    test = [(data['texts'][x], data['info'][x]) for x in data['test_ind']]

    y_train = [train[x][1]['label'] for x in range(len(train))]
    return

def scv2_gen():
    path = os.path.join(DATA_DIR, 'SCv2-GEN', 'raw.pickle')
    data = load_pickle(path)

    train = [(data['texts'][x], data['info'][x]["label"]) for x in data['train_ind']]
    val = [(data['texts'][x], data['info'][x]["label"]) for x in data['val_ind']]
    test = [(data['texts'][x], data['info'][x]["label"]) for x in data['test_ind']]

    X_train = [x[0] for x in train]
    X_val = [x[0] for x in val]
    X_test = [x[0] for x in test]

    y_train = [x[1] for x in train]
    y_val = [x[1] for x in val]
    y_test = [x[1] for x in test]

    return X_train, y_train, X_val, y_val, X_test, y_test

def psych_exp(test=False):
    path = os.path.join(DATA_DIR, 'PsychExp', 'raw.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    train = [(data['texts'][x], data['info'][x]["label"]) for x in data['train_ind']]
    val = [(data['texts'][x], data['info'][x]["label"]) for x in data['val_ind']]
    test = [(data['texts'][x], data['info'][x]["label"]) for x in data['test_ind']]

    X_train = [x[0] for x in train]
    X_val = [x[0] for x in val]
    X_test = [x[0] for x in test]

    y_train_list = [x[1] for x in train]
    y_val_list = [x[1] for x in val]
    y_test_list = [x[1] for x in test]

    y_train = [int(np.where(y_train_list[i]==1.)[0]) for i in range(len(y_train_list))]
    y_val = [int(np.where(y_val_list[i]==1.)[0]) for i in range(len(y_val_list))]
    y_test = [int(np.where(y_test_list[i]==1.)[0]) for i in range(len(y_test_list))]

    # transform labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    # if test:
    #     return X_train, y_train, X_test, y_test
    # else:
    #     return X_train, y_train, X_val, y_val

    return X_train, y_train, X_val, y_val, X_test, y_test

def olympic():
    path = os.path.join(DATA_DIR, 'Olympic', 'raw.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    train = [(data['texts'][x], data['info'][x]["label"]) for x in data['train_ind']]
    val = [(data['texts'][x], data['info'][x]["label"]) for x in data['val_ind']]
    test = [(data['texts'][x], data['info'][x]["label"]) for x in data['test_ind']]

    X_train = [x[0] for x in train]
    X_val = [x[0] for x in val]
    X_test = [x[0] for x in test]

    y_train_list = [x[1] for x in train]
    y_val_list = [x[1] for x in val]
    y_test_list = [x[1] for x in test]

    y_train = [int(np.where(y_train_list[i]==True)[0]) for i in range(len(y_train_list))]
    y_val = [int(np.where(y_val_list[i]==True)[0]) for i in range(len(y_val_list))]
    y_test = [int(np.where(y_test_list[i]==True)[0]) for i in range(len(y_test_list))]

    # transform labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)
    return X_train, y_train, X_val, y_val

def ss_youtube(test_set=False):
    path = os.path.join(DATA_DIR, 'SS-Youtube', 'raw.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    train = [(data['texts'][x], data['info'][x]["label"]) for x in data['train_ind']]
    val = [(data['texts'][x], data['info'][x]["label"]) for x in data['val_ind']]
    test = [(data['texts'][x], data['info'][x]["label"]) for x in data['test_ind']]

    X_train = [x[0] for x in train]
    X_val = [x[0] for x in val]
    X_test = [x[0] for x in test]

    y_train = [x[1] for x in train]
    y_val = [x[1] for x in val]
    y_test = [x[1] for x in test]

    # transform labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)
    if test_set:
        return X_test, y_test
    else:
        return X_train, y_train, X_val, y_val

def mbti_1():
    path = os.path.join(DATA_DIR, 'MBTI_1', 'mbti_1.csv')
    df = pd.read_csv(path)
    X = df.posts.values
    y = df.type.values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, stratify=y)

    # transform labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test

def trec():
    train_file = os.path.join(DATA_DIR, 'TREC', 'TREC.train.all')
    val_file = os.path.join(DATA_DIR, 'TREC', 'TREC.test.all')

    train = load_trec(train_file)
    val = load_trec(val_file)

    X_train = [obs[1] for obs in train]
    y_train = [obs[0] for obs in train]
    X_test = [obs[1] for obs in val]
    y_test = [obs[0] for obs in val]

    # transform labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test

def load_dataset(name, test=False):
    if name == "semeval2017A":
        return semeval_2017A()
    elif name == "irony":
        return irony()
    elif name == "sst_fine_grained":
        return  sst(fine_grained=True)
    elif name == "sst":
        return sst(fine_grained=False)
    elif name == "subj":
        return subj()
    elif name == "mr":
        return mr()
    elif name == "scv1":
        return scv1()
    elif name == "affective":
        return affective()
    elif name == "scv2_gen":
        return scv2_gen()
    elif name == "psych_exp":
        return psych_exp(test)
    elif name == "olympic":
        return olympic()
    elif name == "ss_youtube":
        return ss_youtube()
    elif name == "mbti_1":
        return mbti_1()
    elif name == "trec":
        return trec()
    else:
        raise ValueError(f"The dataset:'{name}' is not yet supported!")

def load_test_file(name):
    if name == "ss_youtube":
        return ss_youtube(test_set=True)
    else:
        raise ValueError(f"The test set of the dataset:'{name}' is not yet supported!")

