import time

import numpy
import torch
from torch.nn.utils import clip_grad_norm_

from modules.trainer import Trainer
from utils.log import epoch_progress
from utils.training import save_checkpoint


class ClfTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f1 = None
        self.acc = None
        self.precision = None
        self.recall = None
        self.best_val_loss = None

    def process_batch(self, src, labels, lengths):
        logits, representations, attentions = self.model(src, lengths)

        loss = self.criterion(logits, labels)

        return loss, (logits, labels, representations, attentions)

    def get_state(self):
        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "optimizers": [x.state_dict() for x in self.optimizers],
            "f1": self.f1,
            "accuracy": self.acc,
            "precision": self.precision,
            "recall": self.recall,
            "val_loss": self.best_val_loss
        }

        return state

class ClfTrainer_withFeatures(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f1 = None
        self.acc = None
        self.precision = None
        self.recall = None
        self.best_val_loss = None

    def process_batch(self, src, labels, features, lengths):
        logits, representations, attentions = self.model(src, features, lengths)

        loss = self.criterion(logits, labels)

        return loss, (logits, labels, representations, attentions)

    def get_state(self):
        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "optimizers": [x.state_dict() for x in self.optimizers],
            "f1": self.f1,
            "accuracy": self.acc,
            "precision": self.precision,
            "recall": self.recall,
            "val_loss": self.best_val_loss
        }

        return state

