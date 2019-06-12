####################################################################
# SETTINGS
####################################################################
import datetime
import os

import numpy
import torch
from sklearn.metrics import f1_score
from collections import Counter
from torch import nn, optim
from torch.utils.data import DataLoader

from models.clf_trainer import ClfTrainer_withFeatures
from models.data_manager import load_dataset
from models.feature_manager import load_features, feature_selection
from modules.models import AffectiveAttention
from sys_config import EXP_DIR, EMB_DIR
from utils.adam import Adam
from utils.datasets import ClfDataset, BucketBatchSampler, SortedSampler, \
    ClfCollate_withFeatures
from utils.early_stopping import Early_stopping
from utils.generic import number_h
from utils.load_embeddings import load_word_vectors, load_word_vectors_from_fasttext
from utils.opts import train_options
from utils.training import acc, f1_macro, precision_macro, recall_macro, class_weigths


def clf_features_runner(yaml, word2idx, idx2word, weights, cluster=False):
    if cluster is False:
        from logger.experiment import Experiment

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    opts, config = train_options(yaml)
    ####################################################################
    # Data Loading and Preprocessing
    ####################################################################
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(config["data"]["dataset"])

    # load word embeddings
    # if config["data"]["embeddings"] == "wiki.en.vec":
    #     word2idx, idx2word, weights = load_word_vectors_from_fasttext(
    #         os.path.join(EMB_DIR, config["data"]["embeddings"]),
    #         config["data"]["embeddings_dim"])
    # else:
    #     word2idx, idx2word, weights = load_word_vectors(
    #         os.path.join(EMB_DIR, config["data"]["embeddings"]),
    #         config["data"]["embeddings_dim"])

    ####################################################################
    # Linguistic Features Loading and Selection
    ####################################################################
    # Any features/lexicon should be in the form of a dictionary
    # For example: lex = {'word':[0., 1., ..., 0.]}

    # load affect features
    print("Loading linguistic features...")
    # todo: streamline feature loading pipeline
    features, feat_length = load_features(config["data"]["features"])
    # assert ... same len

    # zeros = []
    #
    # for word in features_all:
    #     zeros.append(features_all[word].count(0.0))
    # final_features, feat_length = feature_selection(list(features_all.values()), .9)
    # features = {key: value for key, value in zip(features_all.keys(), final_features)}



    # build dataset
    print("Building training dataset...")
    train_set = ClfDataset(X_train, y_train, word2idx,
                           feat_length=feat_length,
                           features_dict=features)

    print("Building validation dataset...")
    val_set = ClfDataset(X_val, y_val, word2idx,
                         features_dict=features,
                         feat_length=feat_length)

    test_set = ClfDataset(X_test, y_test, word2idx,
                         features_dict=features,
                         feat_length=feat_length)

    # train_set.truncate(1000)
    # val_set.truncate(100)

    src_lengths = [len(x) for x in train_set.data]
    val_lengths = [len(x) for x in val_set.data]
    test_lengths = [len(x) for x in test_set.data]

    # select sampler & dataloader
    train_sampler = BucketBatchSampler(src_lengths, config["batch_size"], True)
    val_sampler = SortedSampler(val_lengths)
    val_sampler_train = SortedSampler(src_lengths)
    test_sampler = SortedSampler(test_lengths)

    train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                              num_workers=opts.cores, collate_fn=ClfCollate_withFeatures())
    val_loader = DataLoader(val_set, sampler=val_sampler,
                            batch_size=config["batch_size"],
                            num_workers=opts.cores, collate_fn=ClfCollate_withFeatures())
    val_loader_train_dataset = DataLoader(train_set, sampler=val_sampler_train,
                            batch_size=config["batch_size"],
                            num_workers=opts.cores, collate_fn=ClfCollate_withFeatures())
    test_loader = DataLoader(test_set, sampler=test_sampler,
                            batch_size=config["batch_size"],
                            num_workers=opts.cores, collate_fn=ClfCollate_withFeatures())

    ####################################################################
    # Model
    ####################################################################
    # feature_size = numpy.array(affect_features).shape[1]
    model = AffectiveAttention(ntokens=weights.shape[0],
                               nclasses=len(set(train_set.labels)),
                               feature_size=feat_length,
                               **config["model"])
    model.word_embedding.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                                         requires_grad=False)
    model.to(opts.device)
    print(model)

    ####################################################################
    # Count total parameters of model
    ####################################################################
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters()
                                 if p.requires_grad)

    print("Total Params:", number_h(total_params))
    print("Total Trainable Params:", number_h(total_trainable_params))

    if config["class_weights"]:
        class_weights = class_weigths(train_set.labels,to_pytorch=True)
        class_weights = class_weights.to(opts.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    clf_parameters = filter(lambda p: p.requires_grad, model.parameters())
    clf_optimizer = Adam(clf_parameters,  weight_decay=1e-5)

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    # nof_params = count_parameters(model)
    # print(nof_params)

    ####################################################################
    # Training Pipeline
    ####################################################################
    _outputs = []


    def batch_callback(i_batch, losses, batch_outputs):
        _outputs.append(batch_outputs)

        if trainer.step % config["log_interval"] == 0:
            outputs = list(zip(*_outputs))
            _losses = numpy.array(losses[-config["log_interval"]:]).mean(0)
            exp.update_metric("clf-loss", _losses)
            _y_hat = torch.cat(outputs[0]).max(-1)[1].cpu().data.numpy()
            _y = torch.cat(outputs[1]).cpu().data.numpy()
            f1 = f1_score(_y, _y_hat, average='macro')
            exp.update_metric("f1-train", f1)

            losses_log = exp.log_metrics(["clf-loss", 'f1-train'])
            exp.update_value("progress", trainer.progress_log + "\n" + losses_log)

            # clean lines and move cursor back up N lines
            print("\n\033[K" + losses_log)
            print("\033[F" * (len(losses_log.split("\n")) + 2))

            _outputs.clear()


    # Trainer: responsible for managing the training process
    trainer = ClfTrainer_withFeatures(model=model, train_loader=train_loader,
                                      valid_loader=val_loader,
                                      valid_loader_train_set=val_loader_train_dataset,
                                      test_loader=test_loader,
                                      criterion=criterion,
                                      optimizers=clf_optimizer,
                                      config=config, device=opts.device,
                                      batch_end_callbacks=None)

    ####################################################################
    # Experiment: logging and visualizing the training process
    ####################################################################
    if cluster is False:
        exp = Experiment(opts.name, config, src_dirs=opts.source, output_dir=EXP_DIR)

        exp.add_metric("ep_loss", "line", "epoch loss", ["TRAIN", "VAL"])
        exp.add_metric("ep_f1", "line", "epoch f1", ["TRAIN", "VAL"])
        exp.add_metric("ep_acc", "line", "epoch accuracy", ["TRAIN", "VAL"])
        exp.add_metric("ep_pre", "line", "epoch precision", ["TRAIN", "VAL"])
        exp.add_metric("ep_rec", "line", "epoch recall", ["TRAIN", "VAL"])

        exp.add_value("epoch", title="epoch summary", vis_type="text")
        exp.add_value("progress", title="training progress", vis_type="text")

    ####################################################################
    # Training Loop
    ####################################################################
    best_loss = None
    early_stopping = Early_stopping("min", config["patience"])

    for epoch in range(config["epochs"]):
        train_loss = trainer.train_epoch()
        val_loss, y, y_pred = trainer.eval_epoch(val_set=True)
        _, y_train, y_pred_train = trainer.eval_epoch(train_set=True)

        # Calculate accuracy and f1-macro on the evaluation set
        if cluster is False:
            exp.update_metric("ep_loss", train_loss.item(), "TRAIN")
            exp.update_metric("ep_loss", val_loss.item(), "VAL")

            exp.update_metric("ep_f1", f1_macro(y_train, y_pred_train), "TRAIN")
            exp.update_metric("ep_f1", f1_macro(y, y_pred), "VAL")

            exp.update_metric("ep_acc", acc(y_train, y_pred_train), "TRAIN")
            exp.update_metric("ep_acc", acc(y, y_pred), "VAL")

            exp.update_metric("ep_pre", precision_macro(y_train, y_pred_train), "TRAIN")
            exp.update_metric("ep_pre", precision_macro(y, y_pred), "VAL")

            exp.update_metric("ep_rec", recall_macro(y_train, y_pred_train), "TRAIN")
            exp.update_metric("ep_rec", recall_macro(y, y_pred), "VAL")

            print()
            epoch_log = exp.log_metrics(["ep_loss", "ep_f1", "ep_acc", "ep_pre", "ep_rec"])
            print(epoch_log)
            exp.update_value("epoch", epoch_log)

            exp.save()
        else:
            print("epoch: {}, train loss: {}, val loss: {}, f1: {}".format(epoch,
                                                                           train_loss.item(),
                                                                           val_loss.item(),
                                                                           f1_macro(y, y_pred)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_loss or val_loss < best_loss:
            best_loss = val_loss
            trainer.best_val_loss = best_loss
            trainer.acc = acc(y, y_pred)
            trainer.f1 = f1_macro(y, y_pred)
            trainer.precision = precision_macro(y, y_pred)
            trainer.recall = recall_macro(y, y_pred)

            trainer.checkpoint(name=opts.name, verbose=False)

        if early_stopping.stop(val_loss):
            print("Early Stopping...")
            break

        print("\n")

    #################
    # Test
    #################
    _, y_test_, y_test_predicted = trainer.eval_epoch(test_set=True)
    f1_test = f1_macro(y_test_, y_test_predicted)
    acc_test = acc(y_test_, y_test_predicted)
    print("#"*33)
    print("F1 for test set: {}".format(f1_test))
    print("Accuracy for test set: {}".format(acc_test))
    print("#"*33)

    return trainer.best_val_loss, trainer.acc, trainer.f1, trainer.precision, trainer.recall, f1_test, acc_test

