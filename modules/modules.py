import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules.layers import Embed


class RecurrentHelper:
    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

    def last_timestep(self, outputs, lengths, bi=False):
        if bi:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    def pad_outputs(self, out_packed, max_length):

        out_unpacked, _lengths = pad_packed_sequence(out_packed,
                                                     batch_first=True)

        # pad to initial max length
        pad_length = max_length - out_unpacked.size(1)
        out_unpacked = F.pad(out_unpacked, (0, 0, 0, pad_length))
        return out_unpacked

    @staticmethod
    def project2vocab(output, projection):
        # output_unpacked.size() = batch_size, max_length, hidden_units
        # flat_outputs = (batch_size*max_length, hidden_units),
        # which means that it is a sequence of *all* the outputs (flattened)
        flat_output = output.contiguous().view(output.size(0) * output.size(1),
                                               output.size(2))

        # the sequence of all the output projections
        decoded_flat = projection(flat_output)

        # reshaped the flat sequence of decoded words,
        # in the original (reshaped) form (3D tensor)
        decoded = decoded_flat.view(output.size(0), output.size(1),
                                    decoded_flat.size(1))
        return decoded

    @staticmethod
    def sort_by(lengths):
        """
        Sort batch data and labels by length.
        Useful for variable length inputs, for utilizing PackedSequences
        Args:
            lengths (nn.Tensor): tensor containing the lengths for the data

        Returns:
            - sorted lengths Tensor
            - sort (callable) which will sort a given iterable
                according to lengths
            - unsort (callable) which will revert a given iterable to its
                original order

        """
        batch_size = lengths.size(0)

        sorted_lengths, sorted_idx = lengths.sort()
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

        if lengths.data.is_cuda:
            reverse_idx = reverse_idx.cuda()

        sorted_lengths = sorted_lengths[reverse_idx]

        def sort(iterable):

            if iterable is None:
                return None

            if len(iterable.shape) > 1:
                return iterable[sorted_idx][reverse_idx]
            else:
                return iterable

        def unsort(iterable):

            if iterable is None:
                return None

            if len(iterable.shape) > 1:
                return iterable[reverse_idx][original_idx][reverse_idx]
            else:
                return iterable

        return sorted_lengths, sort, unsort


class RNNModule(nn.Module, RecurrentHelper):
    def __init__(self, input_size,
                 rnn_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.,
                 pack=True, last=False):
        """
        A simple RNN Encoder, which produces a fixed vector representation
        for a variable length sequence of feature vectors, using the output
        at the last timestep of the RNN.
        Args:
            input_size (int): the size of the input features
            rnn_size (int):
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        """
        super(RNNModule, self).__init__()

        self.pack = pack
        self.last = last

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=rnn_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        # the dropout "layer" for the output of the RNN
        # self.dropout = nn.Dropout(dropout)

        # define output feature size
        self.feature_size = rnn_size

        # double if bidirectional
        if bidirectional:
            self.feature_size *= 2

    def reorder_hidden(self, hidden, order):
        if isinstance(hidden, tuple):
            hidden = hidden[0][:, order, :], hidden[1][:, order, :]
        else:
            hidden = hidden[:, order, :]

        return hidden

    def forward(self, x, hidden=None, lengths=None):

        batch, max_length, feat_size = x.size()

        if lengths is not None and self.pack:

            ###############################################
            # sorting
            ###############################################
            lenghts_sorted, sorted_i = lengths.sort(descending=True)
            _, reverse_i = sorted_i.sort()

            x = x[sorted_i]

            if hidden is not None:
                hidden = self.reorder_hidden(hidden, sorted_i)

            ###############################################
            # forward
            ###############################################
            packed = pack_padded_sequence(x, lenghts_sorted, batch_first=True)

            self.rnn.flatten_parameters()
            out_packed, hidden = self.rnn(packed, hidden)

            out_unpacked, _lengths = pad_packed_sequence(out_packed,
                                                         batch_first=True,
                                                         total_length=max_length)

            # out_unpacked = self.dropout(out_unpacked)

            ###############################################
            # un-sorting
            ###############################################
            outputs = out_unpacked[reverse_i]
            hidden = self.reorder_hidden(hidden, reverse_i)

        else:
            # raise NotImplementedError
            # todo: make hidden return the true last states
            # self.rnn.flatten_parameters()
            outputs, hidden = self.rnn(x, hidden)
            # self.rnn.flatten_parameters()
            # outputs = self.dropout(outputs)

        if self.last:
            return outputs, hidden, self.last_timestep(outputs, lengths,
                                                       self.rnn.bidirectional)

        return outputs, hidden
