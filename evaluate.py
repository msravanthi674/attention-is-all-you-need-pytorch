import torch
import torch.nn.functional as F
from model.transformer import Transformer
from model.utils import create_padding_mask, create_subsequent_mask


def greedy_decode(model, src, start_symbol, end_symbol, max_len=20, pad_idx=0):
    model.eval()
    src_mask = create_padding_mask(src, pad_idx)
    memory = model.encoder(model.pos_encoder(model.src_embed(src)), src_mask)

    ys = torch.ones(src.size(0), 1, dtype=torch.long, device=src.device) * start_symbol
    for i in range(max_len-1):
        tgt_mask = create_subsequent_mask(ys.size(1)).to(src.device)
        out = model(src, ys, src_mask, tgt_mask)
        next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == end_symbol:
            break
    return ys
