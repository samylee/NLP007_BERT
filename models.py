import torch
import torch.nn as nn
import numpy as np


# The positional encoding vector
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=512):
        super(PositionalEncoder, self).__init__()
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float, requires_grad=False).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:, :x.size(1)]


# Self-attention layer
class SelfAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        key_dim = key.size(-1)
        attn = torch.matmul(query / np.sqrt(key_dim), key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, value)

        return output


# Multi-head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout)
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        # Apply the linear projections
        batch_size = query.size(0)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        # Calculate the attention
        scores = self.self_attention(query, key, value, mask)
        # Reshape the output
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        # Apply the linear projection
        output = self.out(output)
        return output


# Transformer encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        # Add and Muti-head attention
        x = x + self.dropout1(self.self_attention(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x2))
        return x


# Encoder transformer
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim)
        self.segment_embedding = nn.Embedding(3, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [EncoderLayer(embedding_dim, num_heads, embedding_dim * 4, dropout) for _ in range(num_layers)])
    
    def forward(self, source, segment_label, source_mask):
        # Embed the source
        x = self.token_embedding(source) + self.position_embedding(source) + self.segment_embedding(segment_label)
        x = self.dropout(x)
        # Propagate through the layers
        for layer in self.layers:
            x = layer(x, source_mask)
        return x


# BERT
class BertModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(BertModel, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, num_heads, num_layers, dropout)
        self.next_sentence_linear = nn.Linear(embedding_dim, 2)
        self.mask_language_linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, source, segment_label):
        # source mask
        source_mask = (source > 0).unsqueeze(1)

        # Encoder forward
        encoder = self.encoder(source, segment_label, source_mask)

        # Final linear layer
        next_sentence = self.next_sentence_linear(encoder[:, 0])
        mask_language = self.mask_language_linear(encoder)
        return next_sentence, mask_language