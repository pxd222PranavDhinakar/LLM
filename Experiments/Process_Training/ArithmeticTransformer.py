# ArithmeticDecoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AbacusEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length):
        super().__init__()
        # Create an embedding layer for the input tokens
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Create a separate embedding layer for positional encodings
        self.pos_embed = nn.Embedding(max_length, embed_size)
        self.max_length = max_length

    def forward(self, x):
        # Get the sequence length of the input
        seq_length = x.size(1)
        # Generate position indices
        pos = torch.arange(seq_length, device=x.device).unsqueeze(0)
        # Truncate positions to max_length
        pos = torch.clamp(pos, max=self.max_length - 1)
        # Get the token embeddings
        embedded = self.embed(x)
        # Get the positional embeddings
        positional = self.pos_embed(pos)
        # Combine token embeddings and positional embeddings
        return embedded + positional[:, :seq_length]

class Head(nn.Module):
    def __init__(self, head_size, embed_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = head_size ** -0.5

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
            
        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Weighted aggregation of values
        out = att @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embed_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ArithmeticTransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, head_size, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, head_size, embed_size, dropout)
        self.feed_forward = FeedForward(embed_size, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.feed_forward(self.ln2(x))
        return x

class ArithmeticTransformer(nn.Module):
    """
    A transformer model specialized for arithmetic operations.
    
    Args:
        vocab_size (int): Size of the vocabulary (typically 14 for digits 0-9 plus special tokens)
        embed_size (int): Dimension of the embeddings
        num_heads (int): Number of attention heads
        head_size (int): Size of each attention head
        ff_dim (int): Dimension of the feed-forward network
        num_layers (int): Number of transformer blocks
        max_length (int): Maximum sequence length
        dropout (float): Dropout rate (default: 0.1)
    """
    def __init__(self, vocab_size, embed_size, num_heads, head_size, ff_dim,
                 num_layers, max_length, dropout=0.1):
        super().__init__()
        self.embedding = AbacusEmbedding(vocab_size, embed_size, max_length)
        self.blocks = nn.ModuleList([
            ArithmeticTransformerBlock(embed_size, num_heads, head_size, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.fc_out(x)
        return logits

def create_arithmetic_transformer(
    vocab_size,
    embed_size,
    num_heads,
    head_size,
    ff_dim,
    num_layers,
    max_length,
    dropout=0.1
):
    """
    A transformer model specialized for arithmetic operations.

    Args:
        vocab_size (int): Size of the vocabulary (typically 14 for digits 0-9 plus special tokens)
        embed_size (int): Dimension of the embeddings
        num_heads (int): Number of attention heads
        head_size (int): Size of each attention head
        ff_dim (int): Dimension of the feed-forward network
        num_layers (int): Number of transformer blocks
        max_length (int): Maximum sequence length
        dropout (float): Dropout rate (default: 0.1)
    """
    return ArithmeticTransformer(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        head_size=head_size,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_length=max_length,
        dropout=dropout
    )
    
    
    