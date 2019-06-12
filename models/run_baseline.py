import datetime
import json
import os
from pprint import pprint

import numpy

from models.clf import clf_baseline_runner
from models.clf_features import clf_features_runner
from models.clf_features_base import clf_features_baseline_runner
from sys_config import YAML_PATH, BASE_DIR, EMB_DIR
from utils.load_embeddings import load_word_vectors_from_fasttext, load_word_vectors
from utils.opts import train_options

# This script is responsible for running experiments.
# flag one_exp = True runs only one experiment
# if False then the runner is called multiple times and the results (val loss, accuracy, f1)
# are saved in a json file

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sst_yamls = os.path.join(YAML_PATH, 'sst')
subj_yamls = os.path.join(YAML_PATH, 'subj')
mr_yamls = os.path.join(YAML_PATH, 'mr')
affective_yamls = os.path.join(YAML_PATH, 'affective')
olympic_yamls = os.path.join(YAML_PATH, 'olympic')
ss_youtube_yamls = os.path.join(YAML_PATH, 'ss_youtube')
mbti_1_yamls = os.path.join(YAML_PATH, 'mbti_1')
trec_yamls = os.path.join(YAML_PATH, 'trec')

irony_yamls = os.path.join(YAML_PATH, 'irony')
sentiment_yamls = os.path.join(YAML_PATH, 'sentiment')
sst_fine_grained_yamls = os.path.join(YAML_PATH, 'sst_fine_grained')
scv1_yamls = os.path.join(YAML_PATH, 'scv1')
scv2_yamls = os.path.join(YAML_PATH, 'scv2_gen')
psych_yamls = os.path.join(YAML_PATH, 'psychexp')

one_exp = False

yaml = "gating.yaml"
yamls_path = irony_yamls

#########################
# Load embeddings
#########################
if yamls_path is irony_yamls or sentiment_yamls:
    word2idx, idx2word, weights = load_word_vectors(
        os.path.join(EMB_DIR, "word2vec_300_6_20_neg.txt"),"300")
else:
    word2idx, idx2word, weights = load_word_vectors_from_fasttext(
        os.path.join(EMB_DIR, "wiki.en.vec"), "300")

#########################
# Run experiments
#########################
if one_exp:
    loss, acc, f1, precision, recall, f1_test, acc_test = clf_features_runner(os.path.join(sentiment_yamls,
                                                                        "{}".format(yaml)), word2idx, idx2word, weights, cluster=True)

    experiments = {'loss':loss, 'acc':acc, 'f1':f1, 'precision':precision, 'recall':recall,
                   'f1_test':f1_test, 'acc_test':acc_test}

    now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")
    name = "sst_tanh_affine_ablation"
    name += "_{}".format(now)
    name += ".json"
    with open(name, 'w') as f:
        json.dump(experiments, f)

else:
    yamls_base = ["baseline.yaml"]

    name_metrics = ["loss", "acc", "f1", "precision", "recall", "f1_test", "acc_test"]
    experiments = {}

    for yaml in yamls_base:
        experiments[yaml] = {}
        val_losses = []
        accs = []
        f1s = []
        precs = []
        rcls = []
        f1_tests = []
        acc_tests = []
        for _ in range(5):
            val_loss, acc, f1, precision, recall, f1_test, acc_test = clf_baseline_runner(os.path.join(yamls_path,
                                                                                    "{}".format(yaml)), word2idx, idx2word, weights, cluster=True)
            val_losses.append(val_loss)
            accs.append(acc)
            f1s.append(f1)
            precs.append(precision)
            rcls.append(recall)
            f1_tests.append(f1_test)
            acc_tests.append(acc_test)
        metrics = [val_losses, accs, f1s, precs, rcls, f1_tests, acc_tests]

        for i, metric in enumerate(metrics):
            mean = numpy.array(metric).mean(axis=0)
            std = numpy.array(metric).std(axis=0)

            experiments[yaml][name_metrics[i]] = (mean, std)

    print(str(yamls_path))
    pprint(experiments)

    path = os.path.join(BASE_DIR, "json_files")
    now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")
    name = "{}_new_experiments_baseline".format(yamls_path)
    name += "_{}".format(now)
    name += ".json"
    with open(os.path.join(path, name), 'w') as f:
        json.dump(experiments, f)

    with open(name, 'w') as f:
        json.dump(experiments, f)





