import torch
import torch.nn as nn


class SimpleGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=128,
        num_heads=4,
        num_layers=2,
        max_length=128,
        dropout=0.1
    ):
        super().__init__()

        self.embed_size = embed_size
        self.max_length = max_length

        # Word embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)

        # Position embeddings
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.dropout = nn.Dropout(dropout)

        # Decoder-only transformer block: self-attention + feedforward,
        # no cross-attention. This is what GPT-style models actually use
        # (the old version routed through nn.TransformerDecoder with a
        # zeroed-out "memory" input just to get a decoder-shaped API,
        # which wastes a full cross-attention pass every forward).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.ln_f = nn.LayerNorm(embed_size)

        # Output layer
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_length = x.shape

        positions = torch.arange(
            0,
            seq_length,
            device=x.device
        ).unsqueeze(0)

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings
        x = self.dropout(x)

        # Causal mask so each position can only see itself and earlier tokens
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device),
            diagonal=1
        ).bool()

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask
        )

        x = self.ln_f(x)

        logits = self.fc_out(x)

        return logits
