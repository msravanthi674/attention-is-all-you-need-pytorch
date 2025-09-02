import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.transformer import Transformer
from model.utils import create_padding_mask, create_subsequent_mask, get_lr

# -------------------------------
# Toy dataset: copy task
# -------------------------------
vocab_size = 50
seq_len = 10
num_samples = 1000
pad_idx = 0
bos_idx = 1

src_data = torch.randint(2, vocab_size, (num_samples, seq_len))
tgt_in = torch.zeros_like(src_data)
tgt_in[:, 0] = bos_idx
tgt_in[:, 1:] = src_data[:, :-1]
tgt_out = src_data.clone()

dataset = TensorDataset(src_data, tgt_in, tgt_out)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------------
# Model, optimizer, loss
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(src_vocab=vocab_size, tgt_vocab=vocab_size, d_model=64,
                    num_heads=8, num_layers=2, d_ff=256).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)


# -------------------------------
# Training loop
# -------------------------------
for epoch in range(3):
    model.train()
    total_loss = 0
    for step, (src, tgt_in, tgt_out) in enumerate(loader, 1):
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        src_mask = create_padding_mask(src, pad_idx)
        tgt_mask = create_subsequent_mask(tgt_in.size(1)).to(device)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(logits.view(-1, vocab_size), tgt_out.view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # learning rate schedule
        lr = get_lr(step, d_model=64, warmup=400)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
