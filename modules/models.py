import torch
from torch import nn

from modules.layers import Embed, SelfAttention
from modules.modules import RecurrentHelper, RNNModule


class Classifier(nn.Module, RecurrentHelper):
    def __init__(self, ntokens, nclasses, **kwargs):
        super(Classifier, self).__init__()

        ############################################
        # Params
        ############################################
        self.ntokens = ntokens
        self.embed_finetune = kwargs.get("embed_finetune", False)
        self.emb_size = kwargs.get("emb_size", 100)
        self.embed_noise = kwargs.get("embed_noise", .0)
        self.embed_dropout = kwargs.get("embed_dropout", .0)
        self.rnn_size = kwargs.get("rnn_size", 100)
        self.rnn_layers = kwargs.get("rnn_layers", 1)
        self.rnn_dropout = kwargs.get("rnn_dropout", .0)
        self.pack = kwargs.get("pack", True)
        self.no_rnn = kwargs.get("no_rnn", False)
        self.bidir = kwargs.get("bidir", False)

        ############################################
        # Layers
        ############################################
        self.word_embedding = Embed(ntokens, self.emb_size,
                                    noise=self.embed_noise,
                                    dropout=self.embed_dropout)

        self.rnn = RNNModule(input_size=self.emb_size,
                             rnn_size=self.rnn_size,
                             num_layers=self.rnn_layers,
                             bidirectional=self.bidir,
                             dropout=self.rnn_dropout,
                             pack=self.pack)

        if self.no_rnn == False:
            self.attention_size = self.rnn_size
        else:
            self.attention_size = self.emb_size

        self.attention = SelfAttention(self.attention_size, baseline=True)

        self.classes = nn.Linear(self.attention_size, nclasses)

    def forward(self, src, lengths=None, features=None):
        # step 1: embed the sentences
        embeds = self.word_embedding(src)

        if self.no_rnn == False:
            # step 2: encode the sentences
            outputs, _ = self.rnn(embeds, lengths=lengths)
        else:
            outputs = embeds

        representations, attentions = self.attention(outputs, lengths)

        # step 3: output layers
        logits = self.classes(representations)

        return logits, representations, attentions


class BaselineConcClassifier(nn.Module, RecurrentHelper):
    def __init__(self, ntokens, nclasses, feat_size, **kwargs):
        super(BaselineConcClassifier, self).__init__()

        ############################################
        # Params
        ############################################
        self.ntokens = ntokens
        self.feat_size = feat_size
        self.embed_finetune = kwargs.get("embed_finetune", False)
        self.emb_size = kwargs.get("emb_size", 100)
        self.embed_noise = kwargs.get("embed_noise", .0)
        self.embed_dropout = kwargs.get("embed_dropout", .0)
        self.rnn_size = kwargs.get("rnn_size", 100)
        self.rnn_layers = kwargs.get("rnn_layers", 1)
        self.rnn_dropout = kwargs.get("rnn_dropout", .0)
        self.pack = kwargs.get("pack", True)
        self.no_rnn = kwargs.get("no_rnn", False)
        self.conc_emb = kwargs.get("conc_emb", False)
        self.conc_rnn = kwargs.get("conc_rnn", False)
        self.conc_out = kwargs.get("conc_out", False)
        self.bidir = kwargs.get("bidir", False)

        ############################################
        # Layers
        ############################################

        self.word_embedding = Embed(num_embeddings=ntokens,
                                    embedding_dim=self.emb_size,
                                    noise=self.embed_noise,
                                    dropout=self.embed_dropout)

        if self.conc_emb:
            rnn_input_size = self.emb_size + self.feat_size
        else:
            rnn_input_size = self.emb_size

        self.rnn = RNNModule(input_size=rnn_input_size,
                             rnn_size=self.rnn_size,
                             num_layers=self.rnn_layers,
                             bidirectional=self.bidir,
                             dropout=self.rnn_dropout,
                             pack=self.pack)

        if self.no_rnn:
            self.attention_size = rnn_input_size
        else:
            if self.conc_rnn:
                self.attention_size = self.rnn.feature_size + self.feat_size
            else:
                self.attention_size = self.rnn.feature_size

        self.attention = SelfAttention(self.attention_size, baseline=True)

        output_input_size = self.attention_size

        self.classes = nn.Linear(output_input_size, nclasses)

    def forward(self, src, features, lengths=None):
        """

        Args:
            src: token indices. 2D shape: (batch, max_len)
            features: additional features. 3D shape: (batch, max_len, feat_dim)
            lengths: actual length of each sample in batch. 1D shape: (batch)
        Returns:

        """
        # step 1: embed the sentences
        words = self.word_embedding(src)

        if self.conc_emb:
            final_words = torch.cat((words, features), dim=2)
        else:
            final_words = words

        # step 2: encode the sentences
        if self.no_rnn:
            outputs = final_words
        else:
            outputs, _ = self.rnn(final_words, lengths=lengths)

        # step 3: attend over the features
        if self.conc_rnn:
            attention_input = torch.cat((outputs, features), dim=2)
        else:
            attention_input = outputs

        representations, attentions = self.attention(attention_input, lengths)

        # step 4: output layers
        logits = self.classes(representations)

        return logits, representations, attentions


class AffectiveAttention(nn.Module, RecurrentHelper):
    def __init__(self, ntokens, nclasses, feature_size,
                 **kwargs):
        super(AffectiveAttention, self).__init__()

        ############################################
        # Params
        ############################################
        self.ntokens = ntokens
        self.feat_size = feature_size
        self.attention_type = kwargs["attention_type"]
        self.embed_finetune = kwargs.get("embed_finetune", False)
        self.emb_size = kwargs.get("emb_size", 100)
        self.embed_noise = kwargs.get("embed_noise", .0)
        self.embed_dropout = kwargs.get("embed_dropout", .0)
        self.rnn_size = kwargs.get("rnn_size", 100)
        self.rnn_layers = kwargs.get("rnn_layers", 1)
        self.rnn_dropout = kwargs.get("rnn_dropout", .0)
        self.att_dropout = kwargs.get("attention_dropout", .0)
        self.pack = kwargs.get("pack", True)
        self.no_rnn = kwargs.get("no_rnn", False)
        self.conc_emb = kwargs.get("conc_emb", False)
        self.conc_rnn = kwargs.get("conc_rnn", False)
        self.conc_out = kwargs.get("conc_out", False)
        self.bidir = kwargs.get("bidir", False)

        ############################################
        # Layers
        ############################################
        self.word_embedding = Embed(ntokens, self.emb_size,
                                    noise=self.embed_noise,
                                    dropout=self.embed_dropout)

        # todo: use features Embedding layer :)
        # self.feat_embedding = nn.Embedding(num_embeddings=num_features,
        #                              embedding_dim=feature_size)

        if self.conc_emb:

            rnn_input_size = self.emb_size + self.feat_size
        else:
            rnn_input_size = self.emb_size

        self.rnn = RNNModule(input_size=rnn_input_size,
                             rnn_size=self.rnn_size,
                             num_layers=self.rnn_layers,
                             bidirectional=False,
                             dropout=self.rnn_dropout,
                             pack=self.pack)

        if self.no_rnn:
            self.attention_size = rnn_input_size
        else:
            if self.conc_rnn:
                self.attention_size = self.rnn.feature_size + self.feat_size
            else:
                self.attention_size = self.rnn.feature_size

        if self.attention_type == "affine":
            self.scale = nn.Linear(feature_size, self.attention_size)
            self.shift = nn.Linear(feature_size, self.attention_size)
            self.attention = SelfAttention(attention_size=self.attention_size,
                                           dropout=self.att_dropout)

        elif self.attention_type == "non_linear_affine":
            self.scale = nn.Linear(feature_size, self.attention_size)
            self.shift = nn.Linear(feature_size, self.attention_size)
            self.tanh = nn.Tanh()
            self.attention = SelfAttention(attention_size=self.attention_size,
                                           dropout=self.att_dropout)

        elif self.attention_type == "concat":
            self.attention = SelfAttention(attention_size=self.attention_size + feature_size,
                                           dropout=self.att_dropout)

        elif self.attention_type == "gate":
            self.gate = nn.Linear(feature_size, self.attention_size)
            self.sigmoid = nn.Sigmoid()
            self.attention = SelfAttention(attention_size=self.attention_size,
                                           dropout=self.att_dropout)

        else:
            raise ValueError("Unknown attention_type")

        self.classes = nn.Linear(self.attention_size, nclasses)

    def forward(self, src, features, lengths=None):
        """

        Args:
            src: token indices. 2D shape: (batch, max_len)
            features: additional features. 3D shape: (batch, max_len, feat_dim)
            lengths: actual length of each sample in batch. 1D shape: (batch)
        Returns:

        """
        # step 1: embed the sentences
        words = self.word_embedding(src)

        if self.conc_emb:
            final_words = torch.cat((words, features), dim=2)
        else:
            final_words = words

        # step 2: encode the sentences
        if self.no_rnn:
            outputs = final_words
        else:
            outputs, _ = self.rnn(final_words, lengths=lengths)

        # step 3: add the features
        if self.conc_rnn:
            attention_input = torch.cat((outputs, features), dim=2)
        else:
            attention_input = outputs

        if self.attention_type == "affine":
            # b (scale transform) shape: (batch, max_len, rnn_size)
            b = self.scale(features)

            # b (shift transform) shape: (batch, max_len, rnn_size)
            g = self.shift(features)

            # c (transformed outputs) shape: (batch, max_len, rnn_size)
            c = attention_input * b + g

        elif self.attention_type == "non_linear_affine":
            # b (scale transform) shape: (batch, max_len, rnn_size)
            b = self.scale(features)

            # b (shift transform) shape: (batch, max_len, rnn_size)
            g = self.shift(features)

            # c (transformed outputs) shape: (batch, max_len, rnn_size)
            c = self.tanh(attention_input * b + g)

        elif self.attention_type == "concat":
            # c shape: (batch, max_len, rnn_size + feat_dim)
            c = torch.cat((attention_input, features), dim=2)

        elif self.attention_type == "gate":
            c = self.sigmoid(self.gate(features)) * attention_input

        else:
            raise ValueError("Unknown attention_type")

        # step 4: attend over the features
        representations, attentions = self.attention(c, lengths)

        if self.attention_type == "concat":
            # remove the features dimension
            representations = representations.narrow(-1, 0, attention_input.size(-1))
            # remove the features dimension
        else:
            representations = (attention_input * attentions.unsqueeze(-1)).sum(1)

        # step 5: output layers
        logits = self.classes(representations)

        return logits, representations, attentions
