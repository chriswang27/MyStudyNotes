# LLM CodeBase

## A Complete Enc-Dec Transformer

### Key Components

1. **PositionalEncoding**: Adds positional information to token embeddings to preserve sequence order
   1. Here the implementation uses: $e^{a * \log b}=b^a$

2. **MultiHeadAttention**: The core attention mechanism that allows the model to focus on different parts of the sequence
3. **PointwiseFeedForward**: A two-layer neural network applied to each position independently
4. **EncoderLayer**: Combines self-attention and feed-forward networks for the encoder
5. **DecoderLayer**: Uses masked self-attention, encoder-decoder attention, and feed-forward networks
6. **Encoder**: Full encoder with embedding, positional encoding, and multiple encoder layers
7. **Decoder**: Full decoder with embedding, positional encoding, and multiple decoder layers
8. **Transformer**: Complete model that combines encoder and decoder

### Important Features

#### Masking

The implementation includes two crucial masks:

- **Padding mask**: Prevents the model from attending to padding tokens
- **Look-ahead mask**: Prevents the decoder from "seeing the future" during training

#### Attention Mechanism

The attention mechanism calculates scores between queries and keys, scales them, applies masking, and then uses these weights to compute a weighted sum of values:

```
scores = (Q × K^T) / sqrt(d_k)
attention_weights = softmax(scores)
output = attention_weights × V
```

#### Multi-Head Attention

Instead of performing attention once, the model splits dimensions into multiple heads:

1. Project inputs to queries, keys, and values
2. Split into multiple heads
3. Apply attention per head
4. Concatenate results
5. Project to output dimension

### How Inference Works

During inference:

1. The source sequence is encoded once
2. The decoder generates tokens one by one
3. For each new token:
   - Previous generated tokens are fed into the decoder
   - A look-ahead mask ensures proper autoregressive generation
   - The model predicts the next token
   - The process repeats until an end-of-sequence token or maximum length

### Tensor Shapes

The code includes detailed comments about tensor shapes throughout the pipeline:

- Input embeddings: [batch_size, seq_length, d_model]
- Attention scores: [batch_size, num_heads, seq_length_q, seq_length_k]
- Attention weights: [batch_size, num_heads, seq_length_q, seq_length_k]
- Output logits: [batch_size, seq_length, vocab_size]

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the token embeddings to introduce a notion of word order.
    
    Args:
        d_model: Hidden dimension size
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    """
    def __init__(self, d_model, max_seq_length=4096, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create a vector of shape (max_seq_length)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Fill in the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        # Add a batch dimension [1, max_seq_length, d_model]
        pe = pe.unsqueeze(0)
        
        # Register buffer makes it a persistent state but not a parameter
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
        
        Returns:
            Output tensor with positional encoding added of shape [batch_size, seq_length, d_model]
        """
        # Add positional encoding to input embeddings
        # x shape: [batch_size, seq_length, d_model]
        # pe shape: [1, max_seq_length, d_model] -> using only up to seq_length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module as described in "Attention is All You Need".
    
    Args:
        d_model: Hidden dimension size
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model          # Model dimension (e.g., 512)
        self.num_heads = num_heads      # Number of attention heads (e.g., 8)
        self.d_k = d_model // num_heads # Dimension per head (e.g., 64)
        
        # Linear projections for Query, Key, Value, and Output
        self.W_q = nn.Linear(d_model, d_model)  # Query projection
        self.W_k = nn.Linear(d_model, d_model)  # Key projection
        self.W_v = nn.Linear(d_model, d_model)  # Value projection
        self.W_o = nn.Linear(d_model, d_model)  # Output projection
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Query tensor [batch_size, seq_length_q, d_model]
            key: Key tensor [batch_size, seq_length_k, d_model]
            value: Value tensor [batch_size, seq_length_v, d_model]
            mask: Optional mask tensor [batch_size, 1, seq_length_q, seq_length_k]
                  (seq_length_k = seq_length_v)
        
        Returns:
            Output tensor of shape [batch_size, seq_length_q, d_model]
            Attention weights of shape [batch_size, num_heads, seq_length_q, seq_length_k]
        """
        batch_size = query.size(0)
        
        # Linear projections
        # [batch_size, seq_length, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for multi-head attention
        # [batch_size, seq_length, num_heads, d_k] -> [batch_size, num_heads, seq_length, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate scaled dot-product attention
        # [batch_size, num_heads, seq_length_q, d_k] @ [batch_size, num_heads, d_k, seq_length_k]
        # = [batch_size, num_heads, seq_length_q, seq_length_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (mask should be 1 for positions we want to attend to and 0 for positions to ignore)
        if mask is not None:
            # Expand mask for multi-head: [batch_size, 1, seq_length_q, seq_length_k] 
            # -> [batch_size, num_heads, seq_length_q, seq_length_k]
            # Replace 0s with -1e9 (large negative number) for softmax
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Calculate attention weights
        # [batch_size, num_heads, seq_length_q, seq_length_k]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # [batch_size, num_heads, seq_length_q, seq_length_k] @ [batch_size, num_heads, seq_length_v, d_k]
        # = [batch_size, num_heads, seq_length_q, d_k]
        out = torch.matmul(attention_weights, V)
        
        # Reshape back: [batch_size, num_heads, seq_length_q, d_k] -> [batch_size, seq_length_q, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        # [batch_size, seq_length_q, d_model]
        out = self.W_o(out)
        
        return out, attention_weights


class PointwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.
    
    Args:
        d_model: Hidden dimension size
        d_ff: Feed forward dimension size (usually 4*d_model)
        dropout: Dropout rate
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PointwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
        
        Returns:
            Output tensor of shape [batch_size, seq_length, d_model]
        """
        # First linear layer with ReLU activation
        # [batch_size, seq_length, d_model] -> [batch_size, seq_length, d_ff]
        x = F.relu(self.linear1(x))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Second linear layer
        # [batch_size, seq_length, d_ff] -> [batch_size, seq_length, d_model]
        x = self.linear2(x)
        
        return x


class EncoderLayer(nn.Module):
    """
    Encoder layer consisting of multi-head self-attention and position-wise feed forward network.
    
    Args:
        d_model: Hidden dimension size
        num_heads: Number of attention heads
        d_ff: Feed forward dimension size
        dropout: Dropout rate
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention mechanism
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Position-wise feed forward network
        self.feed_forward = PointwiseFeedForward(d_model, d_ff, dropout)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            mask: Padding mask of shape [batch_size, 1, 1, seq_length]
        
        Returns:
            Output tensor of shape [batch_size, seq_length, d_model]
            Attention weights of shape [batch_size, num_heads, seq_length, seq_length]
        """
        # Multi-head self-attention with residual connection and layer normalization
        # x shape: [batch_size, seq_length, d_model]
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        
        # Residual connection and layer normalization
        # [batch_size, seq_length, d_model]
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Position-wise feed forward network with residual connection and layer normalization
        # [batch_size, seq_length, d_model]
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attention_weights


class DecoderLayer(nn.Module):
    """
    Decoder layer consisting of masked multi-head self-attention, 
    multi-head encoder-decoder attention, and position-wise feed forward network.
    
    Args:
        d_model: Hidden dimension size
        num_heads: Number of attention heads
        d_ff: Feed forward dimension size
        dropout: Dropout rate
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Multi-head self-attention mechanism
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Multi-head encoder-decoder attention
        self.encoder_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Position-wise feed forward network
        self.feed_forward = PointwiseFeedForward(d_model, d_ff, dropout)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            x: Decoder input tensor of shape [batch_size, seq_length_q, d_model]
            enc_output: Encoder output tensor of shape [batch_size, seq_length_k, d_model]
            look_ahead_mask: Mask for self-attention of shape [batch_size, 1, seq_length_q, seq_length_q]
            padding_mask: Mask for encoder-decoder attention of shape [batch_size, 1, seq_length_q, seq_length_k]
        
        Returns:
            Output tensor of shape [batch_size, seq_length_q, d_model]
            Self-attention weights of shape [batch_size, num_heads, seq_length_q, seq_length_q]
            Encoder-decoder attention weights of shape [batch_size, num_heads, seq_length_q, seq_length_k]
        """
        # Masked multi-head self-attention
        # x shape: [batch_size, seq_length_q, d_model]
        attn1_output, self_attention_weights = self.self_attention(x, x, x, look_ahead_mask)
        
        # Residual connection and layer normalization
        # [batch_size, seq_length_q, d_model]
        x = self.norm1(x + self.dropout1(attn1_output))
        
        # Multi-head encoder-decoder attention
        # query=x shape: [batch_size, seq_length_q, d_model]
        # key=value=enc_output shape: [batch_size, seq_length_k, d_model]
        attn2_output, enc_dec_attention_weights = self.encoder_attention(
            x, enc_output, enc_output, padding_mask)
        
        # Residual connection and layer normalization
        # [batch_size, seq_length_q, d_model]
        x = self.norm2(x + self.dropout2(attn2_output))
        
        # Position-wise feed forward network with residual connection and layer normalization
        # [batch_size, seq_length_q, d_model]
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x, self_attention_weights, enc_dec_attention_weights


class Encoder(nn.Module):
    """
    Transformer Encoder consisting of multiple encoder layers.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Hidden dimension size
        num_layers: Number of encoder layers
        num_heads: Number of attention heads
        d_ff: Feed forward dimension size
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Word embedding layer
        # [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        # [batch_size, seq_length, d_model] -> [batch_size, seq_length, d_model]
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length]
            mask: Padding mask of shape [batch_size, 1, 1, seq_length]
        
        Returns:
            Output tensor of shape [batch_size, seq_length, d_model]
            List of attention weights from each layer
        """
        # Convert input token indices to embeddings
        # [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        # [batch_size, seq_length, d_model]
        x = self.pos_encoding(x)
        
        attention_weights = []
        
        # Pass through each encoder layer
        for layer in self.layers:
            # [batch_size, seq_length, d_model], [batch_size, num_heads, seq_length, seq_length]
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
            
        return x, attention_weights


class Decoder(nn.Module):
    """
    Transformer Decoder consisting of multiple decoder layers.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Hidden dimension size
        num_layers: Number of decoder layers
        num_heads: Number of attention heads
        d_ff: Feed forward dimension size
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Word embedding layer
        # [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        # [batch_size, seq_length, d_model] -> [batch_size, seq_length, d_model]
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            x: Decoder input tensor of shape [batch_size, seq_length]
            enc_output: Encoder output tensor of shape [batch_size, enc_seq_length, d_model]
            look_ahead_mask: Mask for self-attention of shape [batch_size, 1, seq_length, seq_length]
            padding_mask: Mask for encoder-decoder attention of shape [batch_size, 1, seq_length, enc_seq_length]
        
        Returns:
            Output tensor of shape [batch_size, seq_length, d_model]
            Lists of self-attention and encoder-decoder attention weights from each layer
        """
        # Convert input token indices to embeddings
        # [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        # [batch_size, seq_length, d_model]
        x = self.pos_encoding(x)
        
        self_attention_weights = []
        enc_dec_attention_weights = []
        
        # Pass through each decoder layer
        for layer in self.layers:
            # [batch_size, seq_length, d_model], 
            # [batch_size, num_heads, seq_length, seq_length],
            # [batch_size, num_heads, seq_length, enc_seq_length]
            x, self_attn_weights, enc_dec_attn_weights = layer(
                x, enc_output, look_ahead_mask, padding_mask)
            
            self_attention_weights.append(self_attn_weights)
            enc_dec_attention_weights.append(enc_dec_attn_weights)
            
        return x, self_attention_weights, enc_dec_attention_weights


class Transformer(nn.Module):
    """
    Complete Transformer model with encoder and decoder.
    
    Args:
        src_vocab_size: Size of source vocabulary
        tgt_vocab_size: Size of target vocabulary
        d_model: Hidden dimension size
        num_layers: Number of encoder/decoder layers
        num_heads: Number of attention heads
        d_ff: Feed forward dimension size
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, 
                 num_heads=8, d_ff=2048, max_seq_length=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Encoder
        self.encoder = Encoder(
            src_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        
        # Decoder
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        
        # Final linear layer to project to vocabulary size
        # [batch_size, seq_length, d_model] -> [batch_size, seq_length, tgt_vocab_size]
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def encode(self, src, src_mask=None):
        """
        Args:
            src: Source input tensor of shape [batch_size, src_seq_length]
            src_mask: Padding mask of shape [batch_size, 1, 1, src_seq_length]
        
        Returns:
            Encoder output of shape [batch_size, src_seq_length, d_model]
        """
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            tgt: Target input tensor of shape [batch_size, tgt_seq_length]
            enc_output: Encoder output tensor of shape [batch_size, src_seq_length, d_model]
            look_ahead_mask: Mask for self-attention of shape [batch_size, 1, tgt_seq_length, tgt_seq_length]
            padding_mask: Mask for encoder-decoder attention of shape [batch_size, 1, tgt_seq_length, src_seq_length]
        
        Returns:
            Decoder output of shape [batch_size, tgt_seq_length, d_model]
        """
        return self.decoder(tgt, enc_output, look_ahead_mask, padding_mask)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, tgt_padding_mask=None):
        """
        Args:
            src: Source input tensor of shape [batch_size, src_seq_length]
            tgt: Target input tensor of shape [batch_size, tgt_seq_length]
            src_mask: Source padding mask of shape [batch_size, 1, 1, src_seq_length]
            tgt_mask: Look-ahead mask of shape [batch_size, 1, tgt_seq_length, tgt_seq_length]
            tgt_padding_mask: Target padding mask for encoder-decoder attention
                              of shape [batch_size, 1, tgt_seq_length, src_seq_length]
                              
        Returns:
            Output logits of shape [batch_size, tgt_seq_length, tgt_vocab_size]
            Encoder attention weights
            Decoder self-attention weights
            Encoder-decoder attention weights
        """
        # Get encoder output and attention weights
        # enc_output: [batch_size, src_seq_length, d_model]
        # enc_attention: list of [batch_size, num_heads, src_seq_length, src_seq_length]
        enc_output, enc_attention = self.encode(src, src_mask)
        
        # Get decoder output and attention weights
        # dec_output: [batch_size, tgt_seq_length, d_model]
        # dec_self_attention: list of [batch_size, num_heads, tgt_seq_length, tgt_seq_length]
        # dec_enc_attention: list of [batch_size, num_heads, tgt_seq_length, src_seq_length]
        dec_output, dec_self_attention, dec_enc_attention = self.decode(
            tgt, enc_output, tgt_mask, tgt_padding_mask)
        
        # Project to vocabulary size
        # [batch_size, tgt_seq_length, tgt_vocab_size]
        output = self.final_layer(dec_output)
        
        return output, enc_attention, dec_self_attention, dec_enc_attention


def create_padding_mask(seq):
    """
    Creates a padding mask for attention mechanism.
    
    Args:
        seq: Input sequence tensor of shape [batch_size, seq_length]
    
    Returns:
        Padding mask of shape [batch_size, 1, 1, seq_length]
    """
    # Create mask for padding (0 tokens)
    # seq shape: [batch_size, seq_length]
    # (seq == 0) shape: [batch_size, seq_length] with True where tokens are padding (0)
    seq_mask = (seq == 0).float()
    
    # Add dimensions for broadcasting with attention scores
    # [batch_size, 1, 1, seq_length]
    return seq_mask.unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(seq_length):
    """
    Creates a look-ahead mask for decoder self-attention.
    
    Args:
        seq_length: Length of the sequence
    
    Returns:
        Look-ahead mask of shape [seq_length, seq_length]
    """
    # Create upper triangular matrix with 1s
    # [seq_length, seq_length]
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).float()
    
    # Flip values: 0 -> 1, 1 -> 0 (1 means positions to mask)
    # [seq_length, seq_length]
    return mask


def create_masks(src, tgt):
    """
    Creates all masks needed for transformer training.
    
    Args:
        src: Source sequence tensor of shape [batch_size, src_seq_length]
        tgt: Target sequence tensor of shape [batch_size, tgt_seq_length]
    
    Returns:
        Source padding mask of shape [batch_size, 1, 1, src_seq_length]
        Target combined mask of shape [batch_size, 1, tgt_seq_length, tgt_seq_length]
        Target padding mask of shape [batch_size, 1, 1, src_seq_length]
    """
    # Source padding mask
    # [batch_size, 1, 1, src_seq_length]
    src_mask = create_padding_mask(src)
    
    # Target padding mask
    # [batch_size, 1, 1, tgt_seq_length]
    tgt_padding_mask = create_padding_mask(tgt)
    
    # Look-ahead mask for decoder self-attention
    # [tgt_seq_length, tgt_seq_length]
    look_ahead_mask = create_look_ahead_mask(tgt.size(1))
    
    # Combined decoder self-attention mask (both padding and look-ahead)
    # [batch_size, 1, tgt_seq_length, tgt_seq_length]
    combined_mask = torch.max(tgt_padding_mask, look_ahead_mask.unsqueeze(0))
    
    return src_mask, combined_mask, src_mask


def demonstration():
    """
    Demonstrates the transformer model with a simple example.
    """
    # Example parameters
    batch_size = 2
    src_seq_length = 5
    tgt_seq_length = 4
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    
    # Create a simple example input
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_length))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_length))
    
    # Create padding in source (set last position to 0 for first example)
    src[0, -1] = 0
    
    # Add padding in target (set last position to 0 for first example)
    tgt[0, -1] = 0
    
    print("Source shape:", src.shape)
    print("Source tensor:")
    print(src)
    print("\nTarget shape:", tgt.shape)
    print("Target tensor:")
    print(tgt)
    
    # Create masks
    src_mask, tgt_mask, tgt_padding_mask = create_masks(src, tgt)
    
    print("\nSource padding mask shape:", src_mask.shape)
    print("Target look-ahead + padding mask shape:", tgt_mask.shape)
    print("Target padding mask for encoder-decoder attention shape:", tgt_padding_mask.shape)
    
    # Initialize model
    print("\nInitializing transformer model...")
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_layers=2,  # Using 2 layers for demonstration
        num_heads=8,
        d_ff=2048
    )
    
    # Forward pass
    print("\nPerforming forward pass...")
    output, enc_attention, dec_self_attention, dec_enc_attention = transformer(
        src, tgt, src_mask, tgt_mask, tgt_padding_mask)
    
    print("\nOutput logits shape:", output.shape)
    print("Number of encoder attention layers:", len(enc_attention))
    print("Encoder self-attention shape (layer 0):", enc_attention[0].shape)
    print("Number of decoder self-attention layers:", len(dec_self_attention))
    print("Decoder self-attention shape (layer 0):", dec_self_attention[0].shape)
    print("Decoder encoder-attention shape (layer 0):", dec_enc_attention[0].shape)
    
    # Visualize target mask (look-ahead mask)
    print("\nTarget mask visualization (1 = masked position):")
    print(tgt_mask[0, 0].numpy())
    
    # Show prediction for the first batch item, first position
    probabilities = F.softmax(output[0, 0], dim=-1)
    top5_probs, top5_indices = torch.topk(probabilities, 5)
    
    print("\nTop 5 predictions for first batch item, first position:")
    for i, (prob, idx) in enumerate(zip(top5_probs.detach().numpy(), top5_indices.detach().numpy())):
        print(f"  {i+1}. Token {idx}: {prob:.4f}")
    
    # Visualize attention patterns
    print("\nVisualizing attention patterns for first example...")
    
    # Show encoder self-attention (first layer, first head)
    print("\nEncoder self-attention (layer 0, head 0):")
    enc_attn = enc_attention[0][0, 0].detach().numpy()
    print(enc_attn)
    
    # Show decoder self-attention (first layer, first head)
    print("\nDecoder self-attention (layer 0, head 0):")
    dec_attn = dec_self_attention[0][0, 0].detach().numpy()
    print(dec_attn)
    
    # Show encoder-decoder attention (first layer, first head)
    print("\nEncoder-decoder attention (layer 0, head 0):")
    enc_dec_attn = dec_enc_attention[0][0, 0].detach().numpy()
    print(enc_dec_attn)


def create_target_mask_example():
    """
    Creates and visualizes a target mask for a specific sequence length.
    """
    seq_length = 5
    
    # Create look-ahead mask
    mask = create_look_ahead_mask(seq_length)
    
    print("Look-ahead mask for sequence length", seq_length)
    print("1 means position is masked (cannot attend to)")
    print(mask.numpy())
    
    # Explain the mask
    print("\nExplanation:")
    print("- Each position can attend to itself and previous positions.")
    print("- Position 0 can only attend to position 0.")
    print("- Position 1 can attend to positions 0 and 1.")
    print("- Position 2 can attend to positions 0, 1, and 2.")
    print("- And so on...")
    
    # Create a simple target sequence with padding
    batch_size = 1
    tgt = torch.tensor([[1, 2, 3, 0, 0]])  # Last two tokens are padding
    
    # Create the combined padding and look-ahead mask
    tgt_padding_mask = create_padding_mask(tgt)
    combined_mask = torch.max(tgt_padding_mask, mask.unsqueeze(0))
    
    print("\nTarget sequence with padding:")
    print(tgt.numpy())
    
    print("\nCombined look-ahead and padding mask:")
    print(combined_mask[0, 0].numpy())
    print("\nExplanation:")
    print("- Now, positions 3 and 4 are fully masked (padding).")
    print("- Other positions follow the look-ahead pattern.")
    
    return combined_mask


def inference_example():
    """
    Demonstrates inference with the transformer model.
    """
    # Example parameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    max_seq_length = 10
    
    # Initialize model
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_layers=2,  # Using 2 layers for demonstration
        num_heads=8,
        d_ff=2048
    )
    
    # Set model to evaluation mode
    transformer.eval()
    
    # Create a source sentence (e.g., [1, 5, 9, 2, 0, 0] where 0s are padding)
    src = torch.tensor([[1, 5, 9, 2, 0, 0]])
    src_mask = create_padding_mask(src)
    
    # Encode the source sentence
    with torch.no_grad():
        enc_output, _ = transformer.encoder(src, src_mask)
    
    # Start with a beginning-of-sentence token (e.g., token 1)
    tgt = torch.tensor([[1]])
    
    # Generate tokens autoregressively
    for i in range(max_seq_length - 1):
        # Create look-ahead mask for the current target sequence
        tgt_mask = create_look_ahead_mask(tgt.size(1)).unsqueeze(0)
        
        # Decode current output
        with torch.no_grad():
            dec_output, _, _ = transformer.decoder(tgt, enc_output, tgt_mask, src_mask)
            output = transformer.final_layer(dec_output)
        
        # Get the next token (greedy decoding)
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
        
        # Append the next token to the target sequence
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if end-of-sentence token is generated (e.g., token 2)
        if next_token.item() == 2:
            break
    
    print("Source sequence:", src[0].numpy())
    print("Generated sequence:", tgt[0].numpy())
    
    return src, tgt


if __name__ == "__main__":
    print("=== Transformer Model Demonstration ===\n")
    
    # Basic demonstration of the transformer model
    demonstration()
    
    print("\n=== Look-Ahead Mask Example ===\n")
    # Show target mask example
    create_target_mask_example()
    
    print("\n=== Inference Example ===\n")
    # Show inference example
    inference_example()
```

## Decoder-only LLM

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask (decoder self-attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DecoderLLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
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
    
    def create_causal_mask(self, seq_len):
        """Create causal mask to prevent attention to future positions"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        
        return logits, loss
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits, _ = self.forward(input_ids)
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

# Example usage and training setup
def train_step(model, batch, optimizer):
    """Single training step"""
    input_ids, targets = batch
    
    # Forward pass
    logits, loss = model(input_ids, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Example initialization
if __name__ == "__main__":
    # Model parameters
    vocab_size = 50257  # GPT-2 vocab size
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    
    # Initialize model
    model = DecoderLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, loss = model(input_ids, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item()}")
    
    # Example generation
    prompt = torch.randint(0, vocab_size, (1, 5))  # Random prompt
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"Generated sequence length: {generated.shape[1]}")
```

### Adding KV-Cache

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None, kv_cache=None, use_cache=False):
        """
        Forward pass with optional KV-caching
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, n_heads, seq_len, cached_len + seq_len]
            kv_cache: Dictionary containing cached keys and values
            use_cache: Whether to use and update the cache
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            updated_cache: Updated KV cache (if use_cache=True)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections for current input
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Handle KV-cache for efficient generation
        if use_cache and kv_cache is not None:
            # Concatenate current k,v with cached k,v from previous steps
            # This avoids recomputing attention for all previous tokens
            cached_k = kv_cache.get('k')  # [batch_size, n_heads, cached_len, d_k]
            cached_v = kv_cache.get('v')  # [batch_size, n_heads, cached_len, d_k]
            
            if cached_k is not None and cached_v is not None:
                k = torch.cat([cached_k, k], dim=2)  # Concat along sequence dimension
                v = torch.cat([cached_v, v], dim=2)
        
        # Update cache with current keys and values for next iteration
        updated_cache = None
        if use_cache:
            updated_cache = {'k': k, 'v': v}
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask to prevent attending to future positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape: concatenate all attention heads
        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(attn_output)
        
        return output, updated_cache

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None, kv_cache=None, use_cache=False):
        """
        Forward pass through decoder block with optional KV-caching
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
            kv_cache: KV cache for this specific layer
            use_cache: Whether to use and return cache
            
        Returns:
            output: Block output [batch_size, seq_len, d_model]  
            updated_cache: Updated cache for this layer
        """
        # Self-attention with residual connection and layer norm
        # Pre-norm architecture: normalize before attention
        normalized_x = self.norm1(x)
        attn_output, updated_cache = self.self_attn(
            normalized_x, mask, kv_cache, use_cache
        )
        x = x + self.dropout(attn_output)  # Residual connection
        
        # Feed-forward with residual connection and layer norm  
        # Pre-norm: normalize before feed-forward
        normalized_x = self.norm2(x)
        ff_output = self.feed_forward(normalized_x)
        x = x + self.dropout(ff_output)  # Residual connection
        
        return x, updated_cache

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DecoderLLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
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
    
    def create_causal_mask(self, seq_len, cached_len=0):
        """
        Create causal mask to prevent attention to future positions
        
        Args:
            seq_len: Length of current sequence
            cached_len: Length of cached sequence (for KV-cache)
            
        Returns:
            mask: Causal attention mask [1, 1, seq_len, cached_len + seq_len]
        """
        total_len = cached_len + seq_len
        
        # Create causal mask: can attend to all cached positions + current and previous positions
        # For cached tokens: can attend to all (they're from the past)
        # For current tokens: can only attend to cached + previous current tokens
        mask = torch.ones(seq_len, total_len)
        
        # Mask future positions in the current sequence
        # Current tokens can attend to: [all cached tokens] + [current and previous current tokens]
        if seq_len > 1:
            current_mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular for current tokens
            mask[:, cached_len:] = current_mask  # Apply causal mask to current portion
            
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids, targets=None, kv_cache=None, use_cache=False):
        """
        Forward pass with optional KV-caching support
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            targets: Target token IDs for loss calculation [batch_size, seq_len] 
            kv_cache: List of KV caches for each layer (for generation)
            use_cache: Whether to use and return updated cache
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Cross-entropy loss (if targets provided)
            updated_cache: Updated KV cache for all layers (if use_cache=True)
        """
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings scaled by sqrt(d_model) - common practice in transformers
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        # During caching, we need to add position encoding for the correct positions
        if use_cache and kv_cache is not None and len(kv_cache) > 0:
            # Get cached length from first layer's cache
            cached_len = kv_cache[0]['k'].size(2) if kv_cache[0] and 'k' in kv_cache[0] else 0
            # Add positional encoding starting from the cached position
            pos_start = cached_len
            x = x + self.pos_encoding.pe[:, pos_start:pos_start + seq_len]
        else:
            cached_len = 0
            x = self.pos_encoding(x)
        
        # Create causal mask considering cached sequence length
        mask = self.create_causal_mask(seq_len, cached_len).to(input_ids.device)
        
        # Initialize cache list if not provided
        if use_cache and kv_cache is None:
            kv_cache = [None] * len(self.decoder_layers)
        
        # Pass through decoder layers with KV-caching
        updated_cache = []
        for i, layer in enumerate(self.decoder_layers):
            layer_cache = kv_cache[i] if (use_cache and kv_cache) else None
            x, layer_updated_cache = layer(x, mask, layer_cache, use_cache)
            
            if use_cache:
                updated_cache.append(layer_updated_cache)
        
        # Final layer normalization
        x = self.ln_f(x)
        
        # Language modeling head - project to vocabulary size
        logits = self.lm_head(x)
        
        # Calculate cross-entropy loss if targets are provided (training)
        loss = None
        if targets is not None:
            # Flatten logits and targets for loss calculation
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        
        # Return cache only if use_cache is True
        if use_cache:
            return logits, loss, updated_cache
        else:
            return logits, loss
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Generate text autoregressively using KV-cache for efficiency
        
        This method demonstrates the key benefit of KV-caching: instead of recomputing
        attention for all previous tokens at each step, we cache the key-value pairs
        and reuse them, significantly speeding up generation.
        
        Args:
            input_ids: Starting tokens [batch_size, initial_seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (None = no filtering)
            
        Returns:
            generated_ids: Complete sequence including input and generated tokens
        """
        self.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            # Initialize KV cache - will be populated and grown during generation
            kv_cache = [None] * len(self.decoder_layers)
            
            # First forward pass: process initial input and populate cache
            current_ids = input_ids
            logits, _, kv_cache = self.forward(current_ids, use_cache=True, kv_cache=kv_cache)
            
            # Generate tokens one by one
            for step in range(max_new_tokens):
                # Get logits for the last token (most recent)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    # Keep only top-k logits, set others to -inf
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                # Convert logits to probabilities and sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add the new token to our sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # CRITICAL: For next iteration, we only pass the NEW token through the model
                # The KV-cache contains all the previous computations, so we don't need to
                # recompute attention for tokens we've already processed
                if step < max_new_tokens - 1:  # Don't compute on last iteration
                    logits, _, kv_cache = self.forward(
                        next_token,  # Only the new token!
                        use_cache=True, 
                        kv_cache=kv_cache  # Reuse cached key-value pairs
                    )
        
        return current_ids

    def generate_without_cache(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Generate text without KV-cache (for comparison/debugging)
        
        This is the naive approach where we recompute attention for the entire sequence
        at each generation step. Much slower than the cached version above.
        """
        self.eval()
        
        with torch.no_grad():
            current_ids = input_ids
            
            for _ in range(max_new_tokens):
                # Forward pass through entire sequence every time (inefficient!)
                logits, _ = self.forward(current_ids, use_cache=False)
                
                # Sample next token (same logic as cached version)
                next_token_logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        return current_ids

# Example usage and training setup
def train_step(model, batch, optimizer):
    """
    Single training step - KV-cache is not used during training
    Training processes entire sequences at once, so caching doesn't help
    """
    input_ids, targets = batch
    
    # Forward pass without cache (training mode)
    logits, loss = model(input_ids, targets, use_cache=False)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def compare_generation_speed():
    """
    Utility function to compare generation speed with and without KV-cache
    This demonstrates the performance benefit of caching
    """
    import time
    
    # Initialize model
    model = DecoderLLM(vocab_size=1000, d_model=256, n_heads=4, n_layers=4)
    model.eval()
    
    # Create a sample input
    input_ids = torch.randint(0, 1000, (1, 10))
    
    # Time generation with cache
    start_time = time.time()
    with_cache = model.generate(input_ids, max_new_tokens=20)
    cache_time = time.time() - start_time
    
    # Time generation without cache  
    start_time = time.time()
    without_cache = model.generate_without_cache(input_ids, max_new_tokens=20)
    no_cache_time = time.time() - start_time
    
    print(f"Generation with KV-cache: {cache_time:.3f}s")
    print(f"Generation without cache: {no_cache_time:.3f}s") 
    print(f"Speedup: {no_cache_time/cache_time:.2f}x")
    
    # Verify both methods produce same length output
    print(f"Output lengths match: {with_cache.shape == without_cache.shape}")

# Example initialization and usage
if __name__ == "__main__":
    # Model parameters
    vocab_size = 50257  # GPT-2 vocab size
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    
    # Initialize model
    model = DecoderLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example forward pass (training mode - no cache)
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, loss = model(input_ids, targets, use_cache=False)
    print(f"Training - Logits shape: {logits.shape}")
    print(f"Training - Loss: {loss.item()}")
    
    # Example forward pass with cache (generation mode)
    single_input = torch.randint(0, vocab_size, (1, 5))
    logits_cached, _, cache = model(single_input, use_cache=True)
    print(f"Cached - Logits shape: {logits_cached.shape}")
    print(f"Cache contains {len(cache)} layer caches")
    
    # Example generation with KV-cache
    prompt = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"Generated sequence length: {generated.shape[1]} (input: {prompt.shape[1]}, new: {generated.shape[1] - prompt.shape[1]})")
    
    # Uncomment to compare generation speeds
    # compare_generation_speed()
```

#### The Problem: Positional Encoding with Caching

When using KV-cache, we need to ensure tokens get the **correct positional encodings** based on their **absolute position** in the full sequence, not just their position in the current input chunk.

**1. Check if we're using cache with existing cached data**

python

```python
if use_cache and kv_cache is not None and len(kv_cache) > 0:
```

This checks:

- `use_cache`: Are we in caching mode?
- `kv_cache is not None`: Do we have a cache object?
- `len(kv_cache) > 0`: Does the cache actually contain data from previous steps?

**2. Determine how many tokens are already cached**

python

```python
cached_len = kv_cache[0]['k'].size(2) if kv_cache[0] and 'k' in kv_cache[0] else 0
```

Let's break this down:

- `kv_cache[0]`: Get the cache from the first layer
- `kv_cache[0]['k']`: Get the cached Key tensor
- `.size(2)`: Get dimension 2 (sequence length dimension)

**Key tensor shape**: `[batch_size, n_heads, seq_len, d_k]`

- Dimension 0: batch_size
- Dimension 1: n_heads
- **Dimension 2: seq_len** ← This tells us how many tokens are cached
- Dimension 3: d_k

**3. Apply positional encoding starting from the correct position**

python

```python
pos_start = cached_len
x = x + self.pos_encoding.pe[:, pos_start:pos_start + seq_len]
```

**Key insight**: We slice the positional encoding to start from `cached_len`, not from 0!

**Visual Example:**

Let's say we're generating text step by step:

**Step 1: Initial prompt "Hello world"**

python

```python
# Input: ["Hello", "world"] (tokens 0, 1)
# No cache yet
cached_len = 0
pos_start = 0
# Use positional encodings [0, 1]
x = x + pos_encoding.pe[:, 0:2]  # positions 0, 1
```

**Step 2: Generate next token "how"**

python

```python
# Input: ["how"] (this is token 2 in the full sequence)
# Cache contains: ["Hello", "world"] 
cached_len = 2  # Two tokens already processed
pos_start = 2
# Use positional encoding [2] (not [0]!)
x = x + pos_encoding.pe[:, 2:3]  # position 2
```

**Step 3: Generate next token "are"**

python

```python
# Input: ["are"] (this is token 3 in the full sequence)
# Cache contains: ["Hello", "world", "how"]
cached_len = 3
pos_start = 3
# Use positional encoding [3]
x = x + pos_encoding.pe[:, 3:4]  # position 3
```

**Why This Matters:**

**Without proper positional encoding:**

python

```python
# WRONG: This would give every new token position 0
x = x + self.pos_encoding(x)  # Always starts from position 0
```

**With correct positional encoding:**

python

```python
# CORRECT: Each token gets its true absolute position
x = x + self.pos_encoding.pe[:, pos_start:pos_start + seq_len]
```

**The Fallback Case:**

python

```python
else:
    cached_len = 0
    x = self.pos_encoding(x)
```

This handles:

- **Training mode** (no cache)
- **First generation step** (cache is empty)
- **Regular forward pass** (no caching)

In these cases, we use normal positional encoding starting from position 0.

**Complete Example:**

python

```python
# Generation sequence: "The cat sat on the mat"
# Tokens: [1, 2, 3, 4, 5, 6]

# Step 1: Process initial prompt "The cat"
input_ids = [1, 2]  # "The cat"
cached_len = 0
# Positions used: [0, 1]

# Step 2: Generate "sat" 
input_ids = [3]  # just "sat"
cached_len = 2  # "The cat" in cache
# Position used: [2] ← Critical: position 2, not 0!

# Step 3: Generate "on"
input_ids = [4]  # just "on"  
cached_len = 3  # "The cat sat" in cache
# Position used: [3]

# And so on...
```

**Key Takeaway:**

This code ensures that **each token receives the positional encoding corresponding to its absolute position in the full sequence**, even when we're only processing one new token at a time during cached generation. This maintains the model's understanding of token positions throughout the generation process.

#### **The Challenge: Masking with Cache**

When using KV-cache, our attention needs to work on a **mixed sequence**:

- **Cached tokens**: From previous generation steps (all in the "past")
- **Current tokens**: New tokens being processed (need causal masking among themselves)

**1. Calculate total sequence length**

python

```python
total_len = cached_len + seq_len
```

This is the **total attention context**: cached tokens + current tokens.

**2. Initialize mask with all 1s**

python

```python
mask = torch.ones(seq_len, total_len)
```

**Key insight**: The mask shape is `[current_seq_len, total_context_len]`

- **Rows**: Current tokens (what we're computing attention for)
- **Columns**: All tokens (cached + current) that we can potentially attend to

Starting with all 1s means "can attend to everything" - we'll selectively mask out future positions.

**3. Apply causal masking to current tokens**

python

```python
if seq_len > 1:
    current_mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular
    mask[:, cached_len:] = current_mask  # Apply to current portion only
```

This creates a lower triangular mask for the current tokens only.

**Visual Examples:**

- **Example 1: First generation step (no cache)**

python

```python
# Processing initial prompt: "The cat sat"
seq_len = 3, cached_len = 0
total_len = 3

mask = torch.ones(3, 3)
# Initial: [[1, 1, 1],
#          [1, 1, 1], 
#          [1, 1, 1]]

current_mask = torch.tril(torch.ones(3, 3))
# current_mask: [[1, 0, 0],
#               [1, 1, 0],
#               [1, 1, 1]]

mask[:, 0:] = current_mask  # cached_len=0, so start from 0
# Final: [[1, 0, 0],  ← "The" can only see "The"
#        [1, 1, 0],  ← "cat" can see "The", "cat" 
#        [1, 1, 1]]  ← "sat" can see "The", "cat", "sat"
```

- **Example 2: Second generation step (with cache)**

python

```python
# Cache contains: "The cat sat" (3 tokens)
# Processing: "on" (1 new token)
seq_len = 1, cached_len = 3
total_len = 4

mask = torch.ones(1, 4)
# Initial: [[1, 1, 1, 1]]  ← "on" can see everything initially

# seq_len = 1, so the if condition is false
# No additional masking needed!

# Final: [[1, 1, 1, 1]]  ← "on" can see "The", "cat", "sat", "on"
```

- **Example 3: Processing multiple new tokens with cache**

python

```python
# Cache contains: "The cat" (2 tokens)  
# Processing: "sat on" (2 new tokens)
seq_len = 2, cached_len = 2
total_len = 4

mask = torch.ones(2, 4)
# Initial: [[1, 1, 1, 1],
#          [1, 1, 1, 1]]

current_mask = torch.tril(torch.ones(2, 2))
# current_mask: [[1, 0],
#               [1, 1]]

mask[:, 2:] = current_mask  # Apply to positions 2 and 3 (current tokens)
# Final: [[1, 1, 1, 0],  ← "sat" sees "The", "cat", "sat" (not "on")
#        [1, 1, 1, 1]]  ← "on" sees "The", "cat", "sat", "on"
```

**Key Insights:**

**1. Cached tokens are always visible**

python

```python
mask[:, 0:cached_len] = 1  # (implicitly, since we start with all 1s)
```

All cached tokens are from the past, so current tokens can always attend to them.

**2. Current tokens need causal masking among themselves**

python

```python
mask[:, cached_len:] = current_mask
```

Current tokens can only see previous current tokens, not future ones.

**3. The mask structure**

```
[cached_tokens] [current_tokens]
     ↑              ↑
  Always 1s    Causal mask applied
```

**Complete Example with Attention:**

python

```python
# Let's trace through attention computation
# Cache: "Hello world" (tokens 0,1)
# Current: "how are" (tokens 2,3)

# Queries: from "how are" [2x64] 
# Keys: from "Hello world how are" [4x64] (cached + current)
# Values: from "Hello world how are" [4x64]

# Attention scores before masking: [2x4]
# scores = [[s00, s01, s02, s03],  ← "how" attending to all
#          [s10, s11, s12, s13]]  ← "are" attending to all

# Mask: [[1, 1, 1, 0],  ← "how" can't see "are" 
#       [1, 1, 1, 1]]  ← "are" can see everything

# After masking:
# scores = [[s00, s01, s02, -inf],  ← "how" can't attend to "are"
#          [s10, s11, s12, s13]]   ← "are" can attend to all
```

**Why This Design Works:**

1. **Cached tokens**: Always visible because they're from the past
2. **Current tokens**: Follow causal masking rules among themselves
3. **Efficiency**: We only compute attention for current tokens, not recalculate for cached ones
4. **Correctness**: Maintains the autoregressive property that tokens can't see future tokens

The mask ensures that the attention mechanism respects causality even when we're mixing cached computations with new token processing!

## MOE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) implementation.
    
    The MoE layer contains multiple "expert" neural networks and a router network
    that decides which experts to use for each input token.
    
    Args:
        hidden_dim (int): Hidden dimension size of input tensor
        num_experts (int): Number of expert networks
        expert_dim (int): Hidden dimension size in each expert network
        k (int): Number of experts to route each token to (top-k routing)
        capacity_factor (float): Multiplicative factor to increase expert capacity
    """
    def __init__(self, hidden_dim, num_experts, expert_dim, k=2, capacity_factor=1.0):
        super(MixtureOfExperts, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.k = k  # Number of experts to route each token to
        
        # Router network: maps input features to expert routing probabilities
        # Shape transformation: [hidden_dim] -> [num_experts]
        self.router = nn.Linear(hidden_dim, num_experts)
        
        # Define the expert networks (each expert is an MLP)
        # Each expert: [hidden_dim] -> [expert_dim] -> [hidden_dim]
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])
        
        # Calculate expert capacity - how many tokens can be routed to each expert
        # Shape: The maximum number of tokens each expert can process
        # Capacity factor increases the buffer to handle load imbalance
        self.capacity_factor = capacity_factor
        
    def forward(self, x):
        """
        Forward pass through the MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_dim]
            
        Returns:
            output: Tensor of shape [batch_size, seq_length, hidden_dim]
        """
        # Input shape: [batch_size, seq_length, hidden_dim]
        batch_size, seq_length, hidden_dim = x.shape
        
        # Get expert capacity
        total_tokens = batch_size * seq_length
        expected_tokens_per_expert = (total_tokens * self.k) / self.num_experts
        expert_capacity = int(self.capacity_factor * expected_tokens_per_expert)
        
        # Reshape input for routing: [batch_size * seq_length, hidden_dim]
        # This flattens the batch and sequence dimensions to treat each token individually
        x_reshaped = x.reshape(-1, hidden_dim)  # Shape: [batch_size * seq_length, hidden_dim]
        
        # Router logits - scores for each expert for each token
        router_logits = self.router(x_reshaped)  # Shape: [batch_size * seq_length, num_experts]
        
        # Calculate routing probabilities with softmax
        router_probs = F.softmax(router_logits, dim=-1)  # Shape: [batch_size * seq_length, num_experts]
        
        # Select top-k experts for each token
        # Values: probability values of top-k experts, shape: [batch_size * seq_length, k]
        # Indices: indices of top-k experts, shape: [batch_size * seq_length, k]
        router_probs_topk, router_indices_topk = torch.topk(router_probs, k=self.k, dim=-1)
        
        # Normalize the top-k probabilities to sum to 1
        router_probs_topk = router_probs_topk / router_probs_topk.sum(dim=-1, keepdim=True)
        
        # Create a mask tensor of shape [batch_size * seq_length, num_experts, k]
        expert_mask = torch.zeros(
            batch_size * seq_length,
            self.num_experts,
            self.k,
            device=x.device
        )
        
        # Fill the mask with routing probabilities
        # This is a bit tricky: we need to place each top-k probability in the right position
        # For each token and each of its top-k expert choice, we set the corresponding position in the mask
        token_indices = torch.arange(batch_size * seq_length).unsqueeze(-1).expand(-1, self.k)
        k_indices = torch.arange(self.k).unsqueeze(0).expand(batch_size * seq_length, -1)
        
        # Place the top-k probabilities in the right positions in the mask
        expert_mask[token_indices, router_indices_topk, k_indices] = router_probs_topk
        
        # Reshape the mask to [batch_size * seq_length, num_experts * k]
        expert_mask = expert_mask.reshape(batch_size * seq_length, self.num_experts * self.k)
        
        # Create a list to store the output from each expert
        expert_outputs = []
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Extract the tokens assigned to this expert by combining all k choices
            # This is a mask of shape [batch_size * seq_length, k] indicating whether
            # each token is routed to this expert in any of its k choices
            expert_selector = torch.zeros(
                batch_size * seq_length,
                self.k,
                device=x.device
            )
            
            # For each k value, find tokens where this expert was chosen
            for k_idx in range(self.k):
                expert_selector[:, k_idx] = (router_indices_topk[:, k_idx] == expert_idx).float()
            
            # Flatten expert_selector to get a single indicator per token
            # Shape: [batch_size * seq_length]
            token_is_routed = (expert_selector.sum(dim=-1) > 0).float()
            
            # Find the indices of tokens assigned to this expert
            indices_to_this_expert = token_is_routed.nonzero().squeeze(-1)
            
            # Limit the number of tokens per expert based on capacity
            if indices_to_this_expert.shape[0] > 0:  # Only if there are tokens assigned to this expert
                if indices_to_this_expert.shape[0] > self.expert_capacity:
                    # If more tokens than capacity, randomly select up to capacity
                    perm = torch.randperm(indices_to_this_expert.shape[0])
                    indices_to_this_expert = indices_to_this_expert[perm[:self.expert_capacity]]
                
                # Select tokens for this expert
                # Shape: [num_tokens_for_expert, hidden_dim]
                expert_inputs = x_reshaped[indices_to_this_expert]
                
                # Process tokens through the expert
                # Shape: [num_tokens_for_expert, hidden_dim]
                expert_output = expert(expert_inputs)
                
                # Create an output tensor for this expert
                # Initialize with zeros for all tokens
                # Shape: [batch_size * seq_length, hidden_dim]
                expanded_expert_output = torch.zeros(
                    batch_size * seq_length,
                    hidden_dim,
                    device=x.device
                )
                
                # Place the expert outputs back to their original positions
                expanded_expert_output[indices_to_this_expert] = expert_output
                
                # Extract the routing probabilities for this expert
                # Shape: [batch_size * seq_length, k]
                expert_probs = expert_selector * router_probs_topk
                
                # Sum probabilities across k
                # Shape: [batch_size * seq_length]
                expert_probs_sum = expert_probs.sum(dim=-1)
                
                # Weight the expert output by routing probabilities
                # Shape: [batch_size * seq_length, hidden_dim]
                weighted_expert_output = expanded_expert_output * expert_probs_sum.unsqueeze(-1)
                
                # Add to the list of expert outputs
                expert_outputs.append(weighted_expert_output)
            else:
                # If no tokens routed to this expert, add a zero tensor
                expert_outputs.append(torch.zeros(
                    batch_size * seq_length,
                    hidden_dim,
                    device=x.device
                ))
        
        # Combine outputs from all experts by summing them
        # Shape: [batch_size * seq_length, hidden_dim]
        combined_output = sum(expert_outputs)
        
        # Reshape back to original dimensions
        # Shape: [batch_size, seq_length, hidden_dim]
        output = combined_output.reshape(batch_size, seq_length, hidden_dim)
        
        return output

# Example usage and tensor shape demonstration
def test_moe():
    # Set up example parameters
    batch_size = 32
    seq_length = 128
    hidden_dim = 512
    num_experts = 8
    expert_dim = 1024
    k = 2
    
    # Create random input
    x = torch.randn(batch_size, seq_length, hidden_dim)
    print(f"Input shape: {x.shape}")  # [32, 128, 512]
    
    # Initialize MoE model
    moe = MixtureOfExperts(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        expert_dim=expert_dim,
        k=k
    )
    
    # Forward pass
    output = moe(x)
    print(f"Output shape: {output.shape}")  # [32, 128, 512]
    
    # Print model architecture summary
    print("\nMixtureOfExperts Model Architecture:")
    print(f"- Input dimension: {hidden_dim}")
    print(f"- Number of experts: {num_experts}")
    print(f"- Expert hidden dimension: {expert_dim}")
    print(f"- Top-k routing: {k}")
    print(f"- Expert capacity: {moe.expert_capacity}")
    
if __name__ == "__main__":
    test_moe()
```

## BPE

```python
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set

class SimpleBPETokenizer:
    """
    A simple implementation of Byte Pair Encoding (BPE) tokenizer.
    
    BPE is a subword tokenization algorithm that iteratively merges the most
    frequent pair of consecutive symbols in a corpus to create a vocabulary
    of subword units.
    
    The algorithm works in two phases:
    1. Training: Learn merge rules from a training corpus
    2. Encoding: Apply learned rules to tokenize new text
    """
    
    def __init__(self, vocab_size: int = 1000):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (number of tokens to learn)
        """
        self.vocab_size = vocab_size
        self.word_freqs = {}  # Frequency of each word in training corpus
        self.vocab = set()    # Final vocabulary (all possible tokens)
        self.merges = []      # List of merge rules learned during training
        self.token_to_id = {} # Mapping from token to integer ID
        self.id_to_token = {} # Mapping from integer ID to token
        
        # Special tokens
        self.unk_token = "<UNK>"  # Unknown token for out-of-vocabulary words
        self.end_token = "</w>"   # End-of-word token
    
    def _get_word_frequencies(self, text: str) -> Dict[str, int]:
        """
        Extract word frequencies from text and add end-of-word markers.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping words (with </w> suffix) to their frequencies
            
        Example:
            Input: "hello world hello"
            Output: {"h e l l o </w>": 2, "w o r l d </w>": 1}
        """
        # Split text into words and count frequencies
        words = text.lower().split()
        word_freqs = Counter(words)
        
        # Convert each word to character sequence with end-of-word marker
        char_word_freqs = {}
        for word, freq in word_freqs.items():
            # Split word into characters and add end-of-word token
            char_word = ' '.join(list(word)) + ' ' + self.end_token
            char_word_freqs[char_word] = freq
            
        return char_word_freqs
    
    def _get_pairs(self, word_freqs: Dict[str, int]) -> Counter:
        """
        Get all pairs of consecutive symbols and their frequencies.
        
        Args:
            word_freqs: Dictionary of word frequencies
            
        Returns:
            Counter object with pairs and their total frequencies
            
        Example:
            Input: {"h e l l o </w>": 2, "w o r l d </w>": 1}
            Output: Counter({('l', 'l'): 2, ('h', 'e'): 2, ('e', 'l'): 2, ...})
        """
        pairs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            symbols = word.split()
            
            # Count all consecutive pairs in this word
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
                
        return Counter(pairs)
    
    def _merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        """
        Merge the most frequent pair in the vocabulary.
        
        Args:
            pair: The pair of symbols to merge (e.g., ('l', 'l'))
            word_freqs: Current word frequencies
            
        Returns:
            Updated word frequencies after merging the pair
            
        Example:
            pair = ('l', 'l')
            Before: {"h e l l o </w>": 2}
            After:  {"h e ll o </w>": 2}
        """
        new_word_freqs = {}
        for word in word_freqs:
            # Replace the pair with merged version
            new_word = word.replace(" ".join(pair), "".join(pair))
            new_word_freqs[new_word] = word_freqs[word]
            
        return new_word_freqs
    
    def train(self, text: str) -> None:
        """
        Train the BPE tokenizer on the given text.
        
        This is the core training algorithm:
        1. Initialize vocabulary with all characters
        2. Repeatedly find the most frequent pair of symbols
        3. Merge this pair to create a new symbol
        4. Continue until we reach the desired vocabulary size
        
        Args:
            text: Training text corpus
        """
        print("Starting BPE training...")
        print(f"Target vocabulary size: {self.vocab_size}")
        
        # Step 1: Get word frequencies with character-level splitting
        self.word_freqs = self._get_word_frequencies(text)
        print(f"\nInitial word frequencies (first 3):")
        for i, (word, freq) in enumerate(list(self.word_freqs.items())[:3]):
            print(f"  '{word}' -> {freq}")
        
        # Step 2: Initialize vocabulary with all unique characters
        self.vocab = set()
        for word in self.word_freqs.keys():
            self.vocab.update(word.split())
        
        print(f"\nInitial vocabulary size: {len(self.vocab)}")
        print(f"Initial vocab (first 10): {sorted(list(self.vocab))[:10]}")
        
        # Step 3: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.vocab)
        print(f"\nWill perform {num_merges} merges...")
        
        for i in range(num_merges):
            # Get all pairs and their frequencies
            pairs = self._get_pairs(self.word_freqs)
            
            if not pairs:
                print(f"No more pairs to merge. Stopping at iteration {i}")
                break
                
            # Find the most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            best_freq = pairs[best_pair]
            
            if i < 5 or i % 100 == 0:  # Show progress for first few and every 100
                print(f"  Merge {i+1}: {best_pair} (frequency: {best_freq})")
            
            # Merge this pair in all words
            self.word_freqs = self._merge_vocab(best_pair, self.word_freqs)
            
            # Add the new merged token to vocabulary and merges list
            new_token = ''.join(best_pair)
            self.vocab.add(new_token)
            self.merges.append(best_pair)
        
        # Step 4: Create token-to-ID mappings
        self.vocab.add(self.unk_token)  # Add unknown token
        vocab_list = sorted(list(self.vocab))
        
        self.token_to_id = {token: i for i, token in enumerate(vocab_list)}
        self.id_to_token = {i: token for i, token in enumerate(vocab_list)}
        
        print(f"\nTraining complete!")
        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Number of merge rules learned: {len(self.merges)}")
        
        # Show some example merge rules
        print(f"\nFirst 5 merge rules:")
        for i, (left, right) in enumerate(self.merges[:5]):
            print(f"  {i+1}. '{left}' + '{right}' -> '{left}{right}'")
    
    def _apply_merges(self, word: str) -> List[str]:
        """
        Apply learned merge rules to a single word.
        
        Args:
            word: Word to tokenize (space-separated characters)
            
        Returns:
            List of tokens after applying all merge rules
            
        Example:
            Input: "h e l l o </w>"
            After merges: ["he", "ll", "o", "</w>"]
        """
        if len(word.split()) == 1:
            return [word]
        
        # Apply each merge rule in order
        for pair in self.merges:
            bigram = re.escape(' '.join(pair))
            pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            word = pattern.sub(''.join(pair), word)
        
        return word.split()
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs using learned BPE rules.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        if not self.merges:
            raise ValueError("Tokenizer not trained! Call train() first.")
        
        token_ids = []
        words = text.lower().split()
        
        for word in words:
            # Convert word to character sequence with end-of-word marker
            char_word = ' '.join(list(word)) + ' ' + self.end_token
            
            # Apply learned merges
            tokens = self._apply_merges(char_word)
            
            # Convert tokens to IDs
            for token in tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    token_ids.append(self.token_to_id[self.unk_token])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        if not self.id_to_token:
            raise ValueError("Tokenizer not trained! Call train() first.")
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(self.unk_token)
        
        # Join tokens and clean up
        text = ''.join(tokens)
        
        # Replace end-of-word markers with spaces
        text = text.replace(self.end_token, ' ')
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subword tokens (without converting to IDs).
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of subword tokens
        """
        token_ids = self.encode(text)
        return [self.id_to_token[tid] for tid in token_ids]
    
    def get_vocab_info(self) -> Dict:
        """Get information about the learned vocabulary."""
        return {
            'vocab_size': len(self.vocab),
            'num_merges': len(self.merges),
            'sample_tokens': sorted(list(self.vocab))[:20],
            'sample_merges': self.merges[:10]
        }


# Example usage and demonstration
if __name__ == "__main__":
    print("=== BPE Tokenizer Demo ===\n")
    
    # Sample training text
    training_text = """
    hello world hello there world wonderful
    hello wonderful world this is a test
    testing testing one two three
    the quick brown fox jumps over the lazy dog
    hello again world wonderful day today
    programming is fun and wonderful
    machine learning with transformers
    """
    
    print("Training Text:")
    print(training_text.strip())
    print("\n" + "="*50)
    
    # Initialize and train tokenizer
    tokenizer = SimpleBPETokenizer(vocab_size=150)
    tokenizer.train(training_text)
    
    print("\n" + "="*50)
    print("VOCABULARY INFORMATION")
    print("="*50)
    
    vocab_info = tokenizer.get_vocab_info()
    print(f"Final vocabulary size: {vocab_info['vocab_size']}")
    print(f"Number of merge rules: {vocab_info['num_merges']}")
    
    print(f"\nSample tokens from vocabulary:")
    for token in vocab_info['sample_tokens']:
        print(f"  '{token}'")
    
    print(f"\nSample merge rules:")
    for i, (left, right) in enumerate(vocab_info['sample_merges']):
        print(f"  {i+1}. '{left}' + '{right}' -> '{left}{right}'")
    
    print("\n" + "="*50)
    print("TOKENIZATION EXAMPLES")
    print("="*50)
    
    # Test tokenization on various examples
    test_sentences = [
        "hello world",
        "wonderful day",
        "testing machine learning",
        "unknown words here",
        "programming transformers"
    ]
    
    for sentence in test_sentences:
        print(f"\nInput: '{sentence}'")
        
        # Tokenize
        tokens = tokenizer.tokenize(sentence)
        print(f"Tokens: {tokens}")
        
        # Encode to IDs
        token_ids = tokenizer.encode(sentence)
        print(f"Token IDs: {token_ids}")
        
        # Decode back
        decoded = tokenizer.decode(token_ids)
        print(f"Decoded: '{decoded}'")
        
        # Verify round-trip
        if decoded.lower() == sentence.lower():
            print("✓ Round-trip successful!")
        else:
            print("✗ Round-trip mismatch!")
    
    print("\n" + "="*50)
    print("ALGORITHM INSIGHTS")
    print("="*50)
    
    print("""
BPE Algorithm Summary:
1. Start with character-level vocabulary
2. Find most frequent adjacent symbol pairs
3. Merge the most frequent pair into a new symbol
4. Repeat until reaching target vocabulary size
5. Use learned merge rules to tokenize new text

Benefits:
- Handles out-of-vocabulary words gracefully
- Balances vocabulary size with representation quality
- Works well for multiple languages
- Efficient subword representation

Applications:
- Neural machine translation (original use case)
- Language models (GPT, BERT variants)
- Text classification and generation
- Multilingual NLP systems
    """)
```

