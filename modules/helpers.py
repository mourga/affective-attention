import torch
from torch.nn import functional as F


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1)))


def masked_normalization(logits, mask):
    scores = F.softmax(logits, dim=-1)

    # apply the mask - zero out masked timesteps
    masked_scores = scores * mask.float()

    # re-normalize the masked scores
    normed_scores = masked_scores.div(masked_scores.sum(-1, keepdim=True))

    return normed_scores


def masked_normalization_inf(logits, mask):
    logits.masked_fill_(1 - mask, float('-inf'))
    # energies.masked_fill_(1 - mask, -1e18)

    scores = F.softmax(logits, dim=-1)

    return scores


def softmax_discrete(logits, tau=1):
    y_soft = F.softmax(logits.squeeze() / tau, dim=1)
    shape = logits.size()
    _, k = y_soft.max(-1)
    y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
    y = y_hard - y_soft.detach() + y_soft
    return y

