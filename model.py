import torch
import torch.nn as nn

class SimpleGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=128,
        num_heads=4,
        num_layers=2,
        max_length=128
    ):
        super().__init__()

        self.embed_size = embed_size
        self.max_length = max_length

        # Word embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)

        # Position embeddings
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output layer
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.shape

        positions = torch.arange(
            0,
            seq_length,
            device=x.device
        ).unsqueeze(0)

        token_embeddings = self.token_embedding(x)

        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        # Causal mask
        mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device),
            diagonal=1
        ).bool()

        # Decoder needs memory input
        memory = torch.zeros_like(x)

        x = self.transformer(
            tgt=x,
            memory=memory,
            tgt_mask=mask
        )

        logits = self.fc_out(x)

        return logits
