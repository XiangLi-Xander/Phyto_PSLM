"""
LLPS sequence-based prediction model.

Architecture
------------
Residue-level dual-tower network for predicting liquid-liquid phase separation
(LLPS) propensity from protein sequence:

1. **Dual-tower projection** with asymmetric intermediate widths:
   - ESM2 per-residue embeddings: ``1280 -> 640 -> 512``
   - IUPred3 per-residue disorder scores: ``1 -> 256 -> 512``
2. **Element-wise addition** fuses the two modalities.
3. **Learned positional encoding** is added to retain order information.
4. **N-layer TransformerEncoder** (Pre-LN) models cross-residue interactions.
5. **Multi-query cross-attention pooling** aggregates residues into a
   sequence-level vector.
6. A compact **MLP head** produces the final logit.

Reference
---------
This architecture is designed for the ``Standard_LLPS_plants`` pipeline.
"""

import torch
import torch.nn as nn


class MultiQueryAttentionPooling(nn.Module):
    """Multi-query cross-attention pooling layer.

    A fixed set of learnable query vectors attends over the residue-level
    representations via multi-head cross-attention.  Different queries can
    focus on distinct functional motifs (e.g., multiple LLPS stickers).
    """

    def __init__(
        self,
        hidden: int = 512,
        n_queries: int = 4,
        nhead: int = 8,
        dropout: float = 0.15,
    ):
        """
        Parameters
        ----------
        hidden : int, optional
            Hidden dimension (default: 512).
        n_queries : int, optional
            Number of learnable query vectors (default: 4).
        nhead : int, optional
            Number of attention heads (default: 8).
        dropout : float, optional
            Dropout probability (default: 0.15).
        """
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
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, L, hidden)`` — residue-level representations.
        mask : torch.Tensor
            Shape ``(B, L)``, boolean — ``True`` for valid positions.

        Returns
        -------
        torch.Tensor
            Shape ``(B, hidden)`` — pooled sequence-level vector.
        """
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
    """Residue-level LLPS classifier.

    Inputs
    ------
    esm_embeds : torch.Tensor, shape ``(B, L_max, 1280)``
        ESM2 per-residue embeddings.
    iupred_scores : torch.Tensor, shape ``(B, L_max)``
        IUPred3 per-residue disorder scores.
    lengths : torch.Tensor, shape ``(B,)``
        Actual sequence lengths (used to build the attention mask).

    Output
    ------
    logits : torch.Tensor, shape ``(B,)``
        Raw logits (before sigmoid).
    """

    def __init__(
        self,
        d_model: int = 1280,
        hidden: int = 512,
        n_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.15,
        max_len: int = 5000,
        n_queries: int = 4,
        esm_proj_hidden: int = 640,
        iupred_proj_hidden: int = 256,
    ):
        """
        Parameters
        ----------
        d_model : int, optional
            ESM2 embedding dimension (default: 1280).
        hidden : int, optional
            Hidden dimension of the classifier (default: 512).
        n_layers : int, optional
            Number of Transformer encoder layers (default: 6).
        nhead : int, optional
            Number of attention heads (default: 8).
        dropout : float, optional
            Dropout probability (default: 0.15).
        max_len : int, optional
            Maximum sequence length for positional encoding (default: 5000).
        n_queries : int, optional
            Number of attention queries for pooling (default: 4).
        esm_proj_hidden : int, optional
            Intermediate dimension for ESM2 projection (default: 640).
        iupred_proj_hidden : int, optional
            Intermediate dimension for IUPred3 projection (default: 256).
        """
        super().__init__()

        # ESM2 projection tower
        self.esm_proj = nn.Sequential(
            nn.Linear(d_model, esm_proj_hidden),
            nn.LayerNorm(esm_proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(esm_proj_hidden, hidden),
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
        esm_embeds: torch.Tensor,
        iupred_scores: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, L_max = esm_embeds.size(0), esm_embeds.size(1)
        device = esm_embeds.device

        # Attention mask: True for valid residue positions
        mask = torch.arange(L_max, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        # Project ESM2 embeddings
        e = self.esm_proj(esm_embeds)

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
