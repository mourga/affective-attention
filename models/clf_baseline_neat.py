import json
import os

import numpy
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.data_manager import load_dataset
from models.feature_manager import load_features
from modules.models import AffectiveAttention, Classifier
from sys_config import EMB_DIR, YAML_PATH, BASE_DIR
from utils import config
from utils.datasets import ClfDataset, SortedSampler, ClfCollate_withFeatures, ClfCollate
from utils.load_embeddings import load_word_vectors_from_fasttext, load_word_vectors
from utils.opts import train_options
from utils.training import load_checkpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def process_batch_test(model, src, labels, lengths):
    logits, representations, attentions = model(src, lengths)
    return src, logits, labels, attentions

def test_clf(model, iterator, device):
    model.eval()

    posteriors = []
    attentions = []
    labels = []
    texts = []

    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(iterator, 1),
                                   desc="calculating posteriors..."):

            # move all tensors in batch to the selected device
            batch = list(map(lambda x: x.to(device), batch))
            src, logits, label, att = process_batch_test(model, *batch)

            # losses.append(loss.item())

            labels.append(label)
            posteriors.append(logits)
            attentions += [att.cpu().numpy()[i] for i in range(0, att.shape[0])]
            texts += [src.cpu().numpy()[i] for i in range(0, src.shape[0])]

    posteriors = torch.cat(posteriors, dim=0)
    predicted = numpy.argmax(posteriors, 1)
    labels = numpy.array(torch.cat(labels, dim=0))
    posteriors_one_list = [posteriors.cpu().numpy()[i] for i in range(0, posteriors.shape[0])]
    return labels, predicted, posteriors_one_list, attentions, texts


####################################################################
# Test
####################################################################
psych_yamls = os.path.join(YAML_PATH, 'psychexp')
yaml = os.path.join(psych_yamls,"baseline.yaml")

opts, config = train_options(yaml)
device = opts.device
X_train, y_train, X_test, y_test = load_dataset(config["data"]["dataset"], test=True)

# load word embeddings
if config["data"]["embeddings"] == "wiki.en.vec":
    word2idx, idx2word, weights = load_word_vectors_from_fasttext(
        os.path.join(EMB_DIR, config["data"]["embeddings"]),
        config["data"]["embeddings_dim"])
else:
    word2idx, idx2word, weights = load_word_vectors(
        os.path.join(EMB_DIR, config["data"]["embeddings"]),
        config["data"]["embeddings_dim"])

checkpoint_name = "Psych_exp_baseline"

state = load_checkpoint(checkpoint_name)

# features, feat_length = load_features(config["data"]["features"])

test_set = ClfDataset(X_test, y_test, word2idx, name="psych_test")
test_lengths = [len(x) for x in test_set.data]
test_sampler = SortedSampler(test_lengths)
test_loader = DataLoader(test_set, sampler=test_sampler,
                        batch_size=config["batch_size"],
                        num_workers=opts.cores, collate_fn=ClfCollate())



model = Classifier(ntokens=weights.shape[0],
                           nclasses=7,
                           **config["model"])
model.load_state_dict(state["model"])

posteriors_list = []
predicted_list = []
#####################################################################
# Load Trained Model
#####################################################################
model.to(device)
print(model)

#####################################################################
# Evaluate Trained Model on test set & Calculate predictions
#####################################################################
labels, predicted, posteriors, attentions, texts = test_clf(model=model, iterator=test_loader,
                                         device=device)

words = []
for sample in texts:
    sample_words = []
    if 0 in sample:
        sample = numpy.delete(sample, numpy.where(sample == 0))
    for id in sample:
        sample_words.append(idx2word[id])
    words.append(sample_words)

# json for neat vision

indexes = [23, 666, 943, 1974, 2018]
index = 666

t = []
l = []
pr = []
po = []
a = []
i = []
for ind in indexes:
    t.append(words[ind])
    l.append(int(labels[ind]))
    pr.append(int(predicted[ind]))
    po.append(list(numpy.array(posteriors[ind], dtype=numpy.float64)))
    a.append(list(numpy.array(attentions[ind], dtype=numpy.float64)))
    i.append("sample_{}".format(index))

json_list = []
for index in range(0, len(X_test)):
    data_json_dict = {"text": words[index],
                 "label": int(labels[index]),
                 "prediction": int(predicted[index]),
                 "posterior": list(numpy.array(posteriors[index], dtype=numpy.float64)),
                 "attention": list(numpy.array(attentions[index], dtype=numpy.float64)),
                 "id": "sample_{}".format(index)}
    json_list.append(data_json_dict)

path = os.path.join(BASE_DIR, "json_files")
with open(os.path.join(path, "psych_data_baseline.json"), 'w') as f:
    json.dump(json_list, f)

label_jason_dict = {"0": {"name": "joy", "desc":" "},
                    "1": {"name": "fear", "desc": " "},
                    "2": {"name": "anger", "desc": " "},
                    "3": {"name": "sadness", "desc": " "},
                    "4": {"name": "disgust", "desc": " "},
                    "5": {"name": "shame", "desc": " "},
                    "6": {"name": "guilt", "desc": " "},
                    }

with open(os.path.join(path, "psych_label.json".format(index)), 'w') as f:
    json.dump(label_jason_dict, f)