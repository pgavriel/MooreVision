import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter



class PatchProjection(nn.Module):
    """
    Projects a 1D patch representation vector into the transformer's d_model space.
    Your novel patch encoder produces a 768-dim vector; this maps it to d_model.
    """
    def __init__(self, patch_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(patch_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, patch_dim] → [B, N, d_model]
        return self.dropout(self.norm(self.projection(x)))


class BBoxEmbedding(nn.Module):
    """
    Learned linear positional embedding for inter-patch spatial relationships.
    Encodes (x_center, y_center, w, h) normalized to [0, 1] relative to the
    original image dimensions.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.embedding = nn.Linear(4, d_model)

    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        # bboxes: [B, N, 4] → [B, N, d_model]
        return self.embedding(bboxes)


class MooreTransformer(nn.Module):
    """
    Transformer classifier that operates over a sequence of image patch tokens.

    Each image is represented as N patches at variable positions and scales.
    A novel patch encoder (external to this class) maps each patch to a fixed
    768-dim vector. This model:
      1. Projects those vectors into d_model space
      2. Adds learned (x, y, w, h) inter-patch positional embeddings
      3. Prepends a learnable CLS token
      4. Passes the sequence through a standard Transformer encoder
      5. Classifies from the CLS token output

    Args:
        num_classes:  Number of output classes (e.g. 10 for STL-10)
        patch_dim:    Dimensionality of your patch encoder's output (default 768)
        d_model:      Internal transformer dimension (default 256)
        num_heads:    Number of attention heads (default 8, must divide d_model)
        num_layers:   Number of transformer encoder layers (default 6)
        ffn_dim:      Feedforward network hidden dimension (default 1024)
        dropout:      Dropout rate applied throughout (default 0.1)
        max_patches:  Maximum number of patches N per image (default 64)

    Inputs:
        patches:  [B, N, patch_dim]  — output of your patch encoder
        bboxes:   [B, N, 4]          — (x_center, y_center, w, h) in [0, 1]

    Output:
        logits:   [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        patch_dim: int = 768,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_patches: int = 64,
    ):
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.max_patches = max_patches

        # --- Input processing ---
        # Projects 768-dim patch vectors into d_model space
        self.patch_projection = PatchProjection(patch_dim, d_model, dropout)

        # Learned inter-patch positional embedding from (x, y, w, h)
        self.bbox_embedding = BBoxEmbedding(d_model)

        # Learnable CLS token — aggregates sequence info for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",      # GELU is standard in vision transformers
            batch_first=True,       # Expects [B, seq_len, d_model]
            norm_first=True,        # Pre-norm (more stable than post-norm)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # --- Classification head ---
        # Applied to CLS token output only
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Standard weight initialization for transformer models."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        patches: torch.Tensor,
        bboxes: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            patches:          [B, N, patch_dim]  patch encoder outputs
            bboxes:           [B, N, 4]          (x_center, y_center, w, h) in [0,1]
            key_padding_mask: [B, N] bool tensor, True for positions to ignore.
                              Useful if batches contain variable numbers of patches
                              padded to max_patches.

        Returns:
            logits: [B, num_classes]
        """
        B, N, _ = patches.shape

        # 1. Project patch vectors into d_model space
        tokens = self.patch_projection(patches)         # [B, N, d_model]

        # 2. Add inter-patch positional embeddings (x, y, w, h)
        pos_embed = self.bbox_embedding(bboxes)         # [B, N, d_model]
        tokens = tokens + pos_embed                     # [B, N, d_model]

        # 3. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)          # [B, 1, d_model]
        tokens = torch.cat([cls, tokens], dim=1)        # [B, N+1, d_model]

        # 4. Extend padding mask to account for the prepended CLS token
        if key_padding_mask is not None:
            cls_mask = torch.zeros(
                B, 1, dtype=torch.bool, device=patches.device
            )
            key_padding_mask = torch.cat(
                [cls_mask, key_padding_mask], dim=1
            )                                           # [B, N+1]

        # 5. Pass through transformer encoder
        encoded = self.transformer(
            tokens,
            src_key_padding_mask=key_padding_mask,
        )                                               # [B, N+1, d_model]

        # 6. Classify from CLS token (index 0)
        cls_output = encoded[:, 0, :]                   # [B, d_model]
        logits = self.classifier(cls_output)            # [B, num_classes]

        return logits

    def get_patch_embeddings(
        self,
        patches: torch.Tensor,
        bboxes: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns the full sequence of transformer output embeddings,
        excluding the CLS token. Useful for visualization, probing,
        or future self-supervised extensions.

        Returns:
            patch_outputs: [B, N, d_model]
        """
        B, N, _ = patches.shape
        tokens = self.patch_projection(patches)
        pos_embed = self.bbox_embedding(bboxes)
        tokens = tokens + pos_embed

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        if key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=patches.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

        return encoded[:, 1:, :]                        # [B, N, d_model]


# -----------------------------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B = 4       # batch size
    N = 16      # patches per image
    patch_dim = 768

    model = MooreTransformer(
        num_classes=10,
        patch_dim=patch_dim,
        d_model=256,
        num_heads=8,
        num_layers=6,
        ffn_dim=1024,
        dropout=0.1,
    )

    # Simulated patch encoder output
    patches = torch.randn(B, N, patch_dim)

    # Simulated bounding boxes (x_center, y_center, w, h) normalized to [0,1]
    bboxes = torch.rand(B, N, 4)

    logits = model(patches, bboxes)
    print(f"Input patches:  {patches.shape}")
    print(f"Input bboxes:   {bboxes.shape}")
    print(f"Output logits:  {logits.shape}")   # expect [4, 10]

    writer = SummaryWriter('logs')
    writer.add_graph(model, [patches,bboxes])
    writer.close()

    # Test patch embedding extraction
    embeddings = model.get_patch_embeddings(patches, bboxes)
    print(f"Patch embeddings: {embeddings.shape}")  # expect [4, 16, 256]

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")