# NewArchitecture.py: Contains the new architecture for the Transformer model
import torch
import torch.nn as nn
import torch.nn.functional as F

class AbacusEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_length, embed_size)
        self.max_length = max_length
        self.embed_size = embed_size

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        positions = positions.expand(B, T)
        positions = torch.clamp(positions, max=self.max_length - 1)

        tok_embedding = self.embed(x)
        pos_embedding = self.pos_embed(positions)
        return tok_embedding + pos_embedding

class Head(nn.Module):
    def __init__(self, head_size, embed_size, max_length, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size
        
        # Create and register causal mask buffer
        causal_mask = torch.triu(torch.ones(max_length, max_length), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask)  # Changed 'mask' to 'causal_mask'

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Compute attention scores with scaled dot product
        attn = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        
        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_size, max_length, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, embed_size, max_length, dropout) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run heads in parallel and concat
        out_heads = [head(x) for head in self.heads]
        out = torch.cat(out_heads, dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ArithmeticTransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, head_size, ff_dim, max_length, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(
            num_heads, head_size, embed_size, max_length, dropout
        )
        self.ff = FeedForward(embed_size, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class AutoregressiveArithmeticTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, head_size, ff_dim,
                 num_layers, max_length, dropout=0.1, temperature=1.0):
        super().__init__()
        self.embedding = AbacusEmbedding(vocab_size, embed_size, max_length)
        self.blocks = nn.ModuleList([
            ArithmeticTransformerBlock(
                embed_size, num_heads, head_size, ff_dim, max_length, dropout
            )
            for _ in range(num_layers + 2)  # Increased from `num_layers` by 2
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.ff_out = nn.Linear(embed_size, vocab_size)
        self.temperature = temperature * 0.8  # Lowered temperature for coherence
        
        # Add max_length attribute
        self.max_length = max_length  # Initialize max_length
        
        # Add vocab_size attribute
        self.vocab_size = vocab_size
                
        # Special tokens
        self.carry_token = 14  # 'A' token for carry operations
        self.pad_token = 0     # padding token

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= self.max_length, f"Sequence length {T} exceeds maximum length {self.max_length}"

        # Get embeddings
        x = self.embedding(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection with temperature scaling
        x = self.ln_f(x)
        logits = self.ff_out(x) / self.temperature
        
        # Clamp logits to valid vocabulary range
        logits = torch.clamp(logits, min=float('-inf'), max=float('inf'))
        logits[..., self.vocab_size:] = float('-inf')  # Mask invalid tokens
        
        if targets is None:
            return logits
            
        # Match sequence lengths if targets provided
        min_len = min(logits.size(1), targets.size(1))
        logits = logits[:, :min_len, :]
        targets = targets[:, :min_len]
        
        # Create padding mask
        padding_mask = (targets != self.pad_token).float()
        
        # Calculate weighted cross-entropy loss with padding handling
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)
        
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = (loss * padding_mask.view(-1)).sum() / (padding_mask.sum() + 1e-8)
        
        # **Add sequence mismatch penalty**
        predictions = logits.argmax(dim=-1)
        sequence_mismatch_penalty = ((predictions != targets) & (padding_mask.bool())).sum(dim=1).float().mean()
        loss += 0.1 * sequence_mismatch_penalty  # Weight penalty by 0.1
        
        return logits, loss

def create_arithmetic_transformer(config):
    """Helper function to create model from config"""
    return AutoregressiveArithmeticTransformer(
        vocab_size=config.vocab_size,
        embed_size=config.embed_size,
        num_heads=config.num_heads,
        head_size=config.embed_size // config.num_heads,
        ff_dim=config.ff_dim,
        num_layers=config.num_layers,
        max_length=config.max_length,
        dropout=config.dropout,
        temperature=1.0  # default temperature
    )