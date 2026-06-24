"""
Ablation: One-hot encoding replaces ESM2.
Architecture identical except ``seq_proj`` first layer is ``Linear(20, 64)``
instead of ``Linear(1280, 640)``. Everything else (Transformer, pooling,
head) unchanged.
"""

import torch
import torch.nn as nn


class MultiQueryAttentionPooling(nn.Module):
    """Multi-query cross-attention pooling layer."""

    def __init__(
        self,
        hidden: int = 512,
        n_queries: int = 4,
        nhead: int = 8,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden))
        nn.init.xavier_uniform_(self.queries)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        queries = self.queries.expand(B, -1, -1)
        key_padding_mask = ~mask
        attn_out, _ = self.cross_attn(
            queries, x, x,
            key_padding_mask=key_padding_mask,
        )
        pooled = attn_out.mean(dim=1)
        return self.norm(pooled)


class ResidueLLPSClassifier(nn.Module):
    """Residue-level LLPS classifier (one-hot + IUPred3).

    Inputs
    ------
    seq_embeds : torch.Tensor, shape ``(B, L_max, 20)``
        One-hot encoded residues.
    iupred_scores : torch.Tensor, shape ``(B, L_max)``
        IUPred3 per-residue disorder scores.
    lengths : torch.Tensor, shape ``(B,)``
        Actual sequence lengths.

    Output
    ------
    logits : torch.Tensor, shape ``(B,)``
        Raw logits (before sigmoid).
    """

    def __init__(
        self,
        d_alphabet: int = 20,
        hidden: int = 512,
        n_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.15,
        max_len: int = 5000,
        n_queries: int = 4,
        seq_proj_hidden: int = 64,
        iupred_proj_hidden: int = 256,
    ):
        super().__init__()

        # Sequence (one-hot) projection tower
        # Changed: Linear(20, 64) instead of Linear(1280, 640)
        self.seq_proj = nn.Sequential(
            nn.Linear(d_alphabet, seq_proj_hidden),
            nn.LayerNorm(seq_proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_proj_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # IUPred3 projection tower
        self.iupred_proj = nn.Sequential(
            nn.Linear(1, iupred_proj_hidden),
            nn.LayerNorm(iupred_proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(iupred_proj_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Multi-query attention pooling
        self.query_pool = MultiQueryAttentionPooling(
            hidden=hidden,
            n_queries=n_queries,
            nhead=nhead,
            dropout=dropout,
        )

        # Learned positional encoding
        self.pos_embed = nn.Embedding(max_len, hidden)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        seq_embeds: torch.Tensor,
        iupred_scores: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, L_max = seq_embeds.size(0), seq_embeds.size(1)
        device = seq_embeds.device

        # Attention mask: True for valid residue positions
        mask = torch.arange(L_max, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        # Project sequence one-hot embeddings
        e = self.seq_proj(seq_embeds)

        # Project IUPred3 scores
        iupred = iupred_scores.unsqueeze(-1)
        i = self.iupred_proj(iupred)

        # Element-wise fusion
        x = e + i

        # Add positional encoding
        positions = torch.arange(L_max, device=device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        # Transformer encoder
        key_padding_mask = ~mask
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Zero out padding positions
        x = x * mask.unsqueeze(-1).float()

        # Multi-query attention pooling
        pooled = self.query_pool(x, mask)

        return self.head(pooled).squeeze(-1)
