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
        
        # Register causal mask buffer
        mask = torch.triu(torch.ones(max_length, max_length), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Compute attention scores with causal mask
        attn = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        attn = attn.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Store attention for visualization (detach here)
        self.last_attention = attn.detach()

        out = attn @ v
        return out, self.last_attention  # Return detached attention
    
    '''
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Compute attention scores with causal mask
        attn = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        attn = attn.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Store attention for visualization
        self.last_attention = attn.detach()

        out = attn @ v
        return out, attn
    '''

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
        out_heads = []
        attn_heads = []
        for head in self.heads:
            out, attn = head(x)
            out_heads.append(out)
            attn_heads.append(attn)
        
        # Combine heads
        out = torch.cat(out_heads, dim=-1)
        out = self.dropout(self.proj(out))
        return out, attn_heads

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
        attn_out, attention = self.attention(self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attention

class AutoregressiveArithmeticTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, head_size, ff_dim,
                 num_layers, max_length, dropout=0.1):
        super().__init__()
        self.embedding = AbacusEmbedding(vocab_size, embed_size, max_length)
        self.blocks = nn.ModuleList([
            ArithmeticTransformerBlock(
                embed_size, num_heads, head_size, ff_dim, max_length, dropout
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.ff_out = nn.Linear(embed_size, vocab_size)
        
        # Save dimensions
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # For debugging/visualization
        self.attention_maps = []

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= self.max_length, f"Sequence length {T} exceeds maximum length {self.max_length}"

        # Reset attention maps
        self.attention_maps = []
        
        # Get embeddings
        x = self.embedding(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x, attention = block(x)
            self.attention_maps.append(attention)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.ff_out(x)
        
        if targets is None:
            return logits
        else:
            # For training, compute loss
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss

    def get_attention_maps(self):
        """Return attention maps for visualization"""
        return self.attention_maps

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0):
        """Generate tokens autoregressively"""
        # Print generation process for debugging
        print("\nAutoregressive Generation Steps:")
        print("Step | Context -> Next Token (Probability)")
        print("-" * 50)
        
        for i in range(max_new_tokens):
            # Crop context if needed
            x_crop = x if x.size(1) <= self.max_length else x[:, -self.max_length:]
            
            # Get logits for next token only
            logits = self(x_crop)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Debug print
            context = ''.join([self.vocab[t.item()] for t in x[0]])
            next_token_str = self.vocab[next_token.item()]
            prob = probs[0, next_token.item()].item()
            print(f"{i:3d} | {context:20s} -> {next_token_str} ({prob:.4f})")
            
            # Stop if we reach a special end token (if you have one)
            # if next_token.item() == self.vocab['END']:
            #     break
                
            # Append to sequence
            x = torch.cat([x, next_token], dim=1)
        
        return x

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
        dropout=config.dropout
    )