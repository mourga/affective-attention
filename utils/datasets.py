import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy
import torch
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from sys_config import BASE_DIR


def vectorize(tokens, vocab):
    """
    Covert array of tokens, to array of ids
    Args:
        tokens (list): list of tokens
        vocab (dict):
    Returns:  list of ids
    """
    ids = []
    for token in tokens:
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab["<unk>"])
    return ids


def features_mapping(indexes, idx2feat, features):
    """
    Covert array of tokens, to array of ids
    Args:
        tokens (list): list of tokens
        vocab (dict):
    Returns:  list of ids
    """
    feature_list = []
    for index in indexes:
        if index in idx2feat:
            feature_list.append(features[index])
        else:
            feature_list.append(numpy.zeros(len(features[0])))
    return feature_list


def map_from_features_dict(tokens, features_dict, feat_length):
    feature_list = []
    for token in tokens:
        if token in features_dict.keys():
            feature_list.append(numpy.array(features_dict[token]))
        else:
            feature_list.append(numpy.zeros(feat_length))

    return feature_list


def hist_dataset(data, seq_len):
    lengths = [len(x) for x in data]
    plt.hist(lengths, density=1, bins=20)
    plt.axvline(seq_len, color='k', linestyle='dashed', linewidth=1)
    plt.show()


class ClfCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, sort=False, batch_first=True):
        self.sort = sort
        self.batch_first = batch_first

    def pad_collate(self, batch):
        inputs = pad_sequence([torch.LongTensor(x[0]) for x in batch],
                              self.batch_first)
        labels = torch.LongTensor([x[1] for x in batch])
        lengths = torch.LongTensor([x[2] for x in batch])

        return inputs, labels, lengths

    def __call__(self, batch):
        return self.pad_collate(batch)


class ClfCollate_withFeatures:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, sort=False, batch_first=True):
        self.sort = sort
        self.batch_first = batch_first

    def pad_collate(self, batch):
        inputs = pad_sequence([torch.LongTensor(x[0]) for x in batch],
                              self.batch_first)
        labels = torch.LongTensor([x[1] for x in batch])
        # featrues need padding too
        features = pad_sequence([torch.FloatTensor(x[2]) for x in batch],
                                self.batch_first)
        lengths = torch.LongTensor([x[3] for x in batch])

        return inputs, labels, features, lengths

    def __call__(self, batch):
        return self.pad_collate(batch)


class SortedSampler(Sampler):
    """
    Defines a strategy for drawing samples from the dataset,
    in ascending or descending order, based in the sample lengths.
    """

    def __init__(self, lengths, descending=False):
        self.lengths = lengths
        self.desc = descending

    def __iter__(self):

        if self.desc:
            return iter(numpy.flip(numpy.array(self.lengths).argsort(), 0))
        else:
            return iter(numpy.array(self.lengths).argsort())

    def __len__(self):
        return len(self.lengths)


class BucketBatchSampler(Sampler):
    """
    Defines a strategy for drawing batches of samples from the dataset,
    in ascending or descending order, based in the sample lengths.
    """

    def __init__(self, lengths, batch_size, shuffle=False):
        sorted_indices = numpy.array(lengths).argsort()
        num_sections = math.ceil(len(lengths) / batch_size)
        self.batches = numpy.array_split(sorted_indices, num_sections)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter(self.batches[i]
                        for i in torch.randperm(len(self.batches)))
        else:
            return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class BaseDataset(Dataset):
    """
    This is a Base class which extends pytorch's Dataset, in order to avoid
    boilerplate code and equip our datasets with functionality such as
    caching.
    """

    def __init__(self, X, y,
                 name=None,
                 label_transformer=None,
                 verbose=True,
                 preprocess=None):
        """

        Args:
            X (): List of training samples
            y (): List of training labels
            name (str): the name of the dataset. It is needed for caching.
                if None then caching is disabled.
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            label_transformer (LabelTransformer):
        """
        self.data = X
        self.labels = y
        self.name = name
        self.label_transformer = label_transformer
        self.verbose = verbose

        if preprocess is not None:
            self.preprocess = preprocess

        self.data = self.load_preprocessed_data()

    def preprocess(self, name, X):
        """
        Preprocessing pipeline
        Args:
            X (list): list of training examples

        Returns: list of processed examples

        """
        raise NotImplementedError

    @staticmethod
    def _check_cache():
        cache_dir = os.path.join(BASE_DIR, "_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_filename(self):
        return os.path.join(BASE_DIR, "_cache",
                            "preprocessed_{}.p".format(self.name))

    def _write_cache(self, data):
        self._check_cache()

        cache_file = self._get_cache_filename()

        with open(cache_file, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    def load_preprocessed_data(self):

        # NOT using cache
        if self.name is None:
            print("cache deactivated!")
            return self.preprocess(self.name, self.data)

        # using cache
        cache_file = self._get_cache_filename()

        if os.path.exists(cache_file):
            print("Loading {} from cache!".format(self.name))
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("No cache file for {} ...".format(self.name))
            data = self.preprocess(self.name, self.data)
            self._write_cache(data)
            return data


class ClfDataset(BaseDataset):

    def __init__(self, X, y, word2idx,
                 features_dict=None,
                 feat_length=None,
                 name=None,
                 label_transformer=None,
                 verbose=True,
                 preprocess=None):
        """
        A PyTorch Dataset
        What we have to do is to implement the 2 abstract methods:

            - __len__(self): in order to let the DataLoader know the size
                of our dataset and to perform batching, shuffling and so on...
            - __getitem__(self, index): we have to return the properly
                processed data-item from our dataset with a given index

        Args:
            X (): list of training samples
            y (): list of training labels
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            word2idx (dict): a dictionary which maps words to indexes
            label_transformer (LabelTransformer):
        """
        self.word2idx = word2idx

        self.features_dict = features_dict
        self.feat_length = feat_length

        BaseDataset.__init__(self, X, y, name, label_transformer,
                             verbose, preprocess)

    def preprocess(self, name, dataset):
        preprocessor = TextPreProcessor(
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                       'time',
                       'date', 'number'],
            annotate={"hashtag", "elongated", "allcaps", "repeated",
                      'emphasis',
                      'censored'},
            all_caps_tag="wrap",
            fix_text=True,
            segmenter="twitter_2018",
            corrector="twitter_2018",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        ).pre_process_doc

        desc = "PreProcessing dataset {}...".format(name)

        data = [preprocessor(x) for x in tqdm(dataset, desc=desc)]
        return data

    def __len__(self):
        return len(self.data)

    def truncate(self, n):
        self.data = self.data[:n]

    def __getitem__(self, index):

        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training sample
                * label (string): the class label
                * length (int): the length (tokens) of the sentence
                * index (int): the index of the dataitem in the dataset.
                  It is useful for getting the raw input for visualizations.
        """
        sample, label = self.data[index], self.labels[index]

        if self.features_dict is not None:
            features_list = map_from_features_dict(sample,
                                                   self.features_dict,
                                                   self.feat_length)

        # transform the sample and the label,
        # in order to feed them to the model
        sample = vectorize(sample, self.word2idx)

        if self.label_transformer is not None:
            label = self.label_transformer.transform(label)

        if isinstance(label, (list, tuple)):
            label = numpy.array(label)

        if self.features_dict is not None:
            return sample, label, numpy.array(features_list), len(
                self.data[index])
        else:
            return sample, label, len(self.data[index])
