import torch
import torch.nn as nn
import torch.nn.functional as F

class AbacusEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_length, embed_size)
        self.max_length = max_length

    def forward(self, x):
        seq_length = x.size(1)
        pos = torch.arange(seq_length, device=x.device).unsqueeze(0)
        pos = torch.clamp(pos, max=self.max_length - 1)
        embedded = self.embed(x)
        positional = self.pos_embed(pos)
        return embedded + positional[:, :seq_length]

class DecoderHead(nn.Module):
    def __init__(self, head_size, embed_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = head_size ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        
        # Compute attention scores with causal mask
        att = (q @ k.transpose(-2, -1)) * self.scale # (B, T, T)
        
        # Create causal mask (lower triangular)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att.masked_fill_(causal_mask, float('-inf'))
            
        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Weighted aggregation of values
        out = att @ v
        return out

class MultiHeadDecoder(nn.Module):
    def __init__(self, num_heads, head_size, embed_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([DecoderHead(head_size, embed_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
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

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, head_size, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadDecoder(num_heads, head_size, embed_size, dropout)
        self.feed_forward = FeedForward(embed_size, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Pre-norm architecture as specified in the paper
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class ArithmeticDecoder(nn.Module):
    """
    A decoder-only transformer model specialized for arithmetic operations.
    Implementation follows the architecture described in "Transformers Can Do Arithmetic 
    with the Right Embeddings" paper.
    
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
            DecoderBlock(embed_size, num_heads, head_size, ff_dim, dropout)
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

def create_arithmetic_decoder(vocab_size=14, embed_size=64, num_heads=2, ff_dim=256, 
                            num_layers=2, max_length=42, dropout=0.1):
    """
    Helper function to create an ArithmeticDecoder with default parameters.
    
    Args:
        vocab_size (int): Vocabulary size (default: 14)
        embed_size (int): Embedding dimension (default: 64)
        num_heads (int): Number of attention heads (default: 2)
        ff_dim (int): Feed-forward dimension (default: 256)
        num_layers (int): Number of transformer layers (default: 2)
        max_length (int): Maximum sequence length (default: 42)
        dropout (float): Dropout rate (default: 0.1)
        
    Returns:
        ArithmeticDecoder: The configured model
    """
    head_size = embed_size // num_heads
    return ArithmeticDecoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        head_size=head_size,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_length=max_length,
        dropout=dropout
    )