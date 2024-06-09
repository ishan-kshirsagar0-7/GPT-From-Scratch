import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.onnx

# Hyperparameters
batch_size = 32   # how many independent sequences will we process in parallel?
block_size = 128    # what is the maximum context length for predictions?
max_iters = 15000
eval_interval = 750
learning_rate = 1.5e-4
device = 'cuda'
eval_iters = 100
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.1

# For reproducibility
# torch.manual_seed(1337)

# Read the data
with open("scripts.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Unique characters that occur in the script
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create string-integer mappings
string_to_integers = {ch: i for i, ch in enumerate(chars)}
integers_to_string = {i: ch for i, ch in enumerate(chars)}

# Encoder: take a string, output a list of integers
def encode(s): return [string_to_integers[c] for c in s]

# Decoder: take a list of integers, output a string
def decode(l): return ''.join([integers_to_string[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))   # first 90% will be train, rest validation
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ A single Head of Self-Attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x)   # (B,T,C)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))   # (B,T,T)
        wei = F.softmax(wei, dim=-1)   # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x)   # (B,T,C)
        out = wei @ v   # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple Heads of Self-Attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: Communication followed by Computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)   # Final Layer Norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))   # (T,C)
        x = tok_emb + pos_emb   # (B,T,C)
        x = self.blocks(x)   # (B,T,C)
        x = self.ln_f(x)   # (B,T,C)
        logits = self.lm_head(x)   # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]   # (B,C)
            probs = F.softmax(logits, dim=-1)   # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1)   # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)   # (B,T+1)
        return idx

model = LanguageModel()
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Early stopping parameters
patience = 8  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change to qualify as an improvement
best_loss = float('inf')
patience_counter = 0

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Training Loss is {losses['train']:.4f} and Validation Loss is {losses['val']:.4f}")

        # Early stopping check
        if losses['val'] < best_loss - min_delta:
            best_loss = losses['val']
            patience_counter = 0  # Reset the counter
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Sample a batch of data
    xb, yb = get_batch("train")

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), 'customgpt.pth')