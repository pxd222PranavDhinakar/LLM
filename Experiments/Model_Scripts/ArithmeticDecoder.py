import torch
import torch.nn as nn
import torch.nn.functional as F

class AbacusEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length):
        super().__init__()
        # Token embeddings
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Positional embeddings for Abacus encoding
        self.pos_embed = nn.Embedding(max_length, embed_size)
        self.max_length = max_length

    def forward(self, x, pos_ids=None):
        seq_length = x.size(1)
        
        # Generate or use provided position indices
        if pos_ids is None:
            pos = torch.arange(seq_length, device=x.device).unsqueeze(0)
            pos = torch.clamp(pos, max=self.max_length - 1)
        else:
            pos = pos_ids
            
        # Get embeddings
        embedded = self.embed(x)
        positional = self.pos_embed(pos)
        
        # Combine embeddings
        return embedded + positional[:, :seq_length]

class DecoderAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        assert embed_size % num_heads == 0
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        
        # Linear projections
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = self.head_size ** -0.5

    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Project and reshape for multi-head attention
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Create causal mask if none provided
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
            
        # Apply mask and softmax
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Compute output
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.out_dropout(y)
        
        return y

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.GELU(),  # Using GELU as in modern transformers
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Pre-norm architecture
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        
        # Attention and feedforward
        self.attention = DecoderAttention(embed_size, num_heads, dropout)
        self.feed_forward = FeedForward(embed_size, ff_dim, dropout)

    def forward(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class ArithmeticDecoder(nn.Module):
    """
    A decoder-only transformer for arithmetic operations, following the paper's architecture.
    Uses pre-norm, causal attention, and Abacus embeddings.
    """
    def __init__(self, vocab_size, embed_size, num_heads, ff_dim, num_layers, max_length, dropout=0.1):
        super().__init__()
        self.max_length = max_length
        self.embed_size = embed_size
        
        # Embeddings
        self.embedding = AbacusEmbedding(vocab_size, embed_size, max_length)
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x, pos_ids=None):
        # Get embeddings
        x = self.embedding(x, pos_ids)
        
        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.fc_out(x)
        
        return logits

def create_arithmetic_decoder(vocab_size=14, embed_size=64, num_heads=2, ff_dim=256, 
                            num_layers=2, max_length=42, dropout=0.1):
    """
    Helper function to create an ArithmeticDecoder with default parameters.
    """
    return ArithmeticDecoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_length=max_length,
        dropout=dropout
    )