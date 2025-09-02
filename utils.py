import torch


def create_padding_mask(seq, pad_idx=0):
    # seq: [batch, seq_len]
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,L]


def create_subsequent_mask(size):
    mask = torch.tril(torch.ones((size, size))).unsqueeze(0).unsqueeze(0)
    return mask  # [1,1,L,L]


def get_lr(step, d_model, warmup=4000):
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))
