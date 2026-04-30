"""
train_ViT_baseline.py — Vanilla ViT baseline for comparison with MooreTransformer

Designed to be a fair comparison:
  - Same dataset, split, and augmentation as train_MooreTransformer.py
  - Same transformer backbone hyperparameters (d_model, num_layers, num_heads, ffn_dim)
  - Same optimizer, LR schedule, and training loop
  - Same logging, checkpointing, and plotting infrastructure

The only difference is tokenization:
  - MooreTransformer: N random Moore-curve patches at variable positions/scales
  - ViT baseline:     fixed grid of non-overlapping square patches (standard ViT)

ViT-specific CONFIG additions (see VIT_CONFIG below):
  - patch_size:   side length of each square patch in pixels (e.g. 16 → 16×16 patches)
                  For STL-10 (96×96): patch_size=8  → 144 tokens
                                      patch_size=12 → 64 tokens
                                      patch_size=16 → 36 tokens
                                      patch_size=24 → 16 tokens  ← closest to Moore N=16
"""

import os
import sys
import csv
import time
import math
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir,".."))

sys.path.insert(0, os.path.join(script_dir,"../moore_curve"))
from config import CONFIG
from utils import *

# =============================================================================
# ViT-SPECIFIC CONFIG — extends / overrides shared CONFIG where needed
# =============================================================================
VIT_CONFIG = {
    **CONFIG,   # inherit everything from shared config

    # --- Output (separate from MooreTransformer runs) ---
    "output_dir":       "./runs_vit",
    "val_cache_path":   "./val_cache/vit_baseline.pt",

    # --- ViT tokenization ---
    # patch_size controls the grid resolution. Choose to match MooreTransformer's
    # num_patches as closely as possible for a fair sequence-length comparison.
    # STL-10 is 96×96. patch_size=16 → (96/16)^2 = 36 tokens (test moore N=36 for compare).
    "patch_size":       16,


    # patch_dim for ViT is derived automatically: patch_size^2 * 3 channels
    # For patch_size=16: 16*16*3 = 768. This is the raw token dim before projection.
    # d_model, num_layers etc. are inherited from CONFIG and stay identical.
}
# =============================================================================
# Create a unique output path for each run
CONFIG["output_dir"] = create_incremental_dir(VIT_CONFIG["output_dir"],prefix=CONFIG["test_name"])
CONFIG["test_name"] = Path(CONFIG["output_dir"]).name
CONFIG["num_patches"] = int((CONFIG["image_size"]/VIT_CONFIG["patch_size"])**2)

# =============================================================================
# ViT PATCH TOKENIZER
# =============================================================================

class ViTPatchTokenizer(nn.Module):
    """
    Splits an image into a grid of non-overlapping square patches and
    projects each flattened patch to d_model via a linear layer.

    This is exactly the tokenization used in the original ViT paper.
    Produces (image_size / patch_size)^2 tokens per image.

    Args:
        image_size:  spatial dimension of the input image (assumes square)
        patch_size:  side length of each patch in pixels
        in_channels: number of input channels (3 for RGB)
        d_model:     output embedding dimension
    """
    def __init__(
        self,
        image_size:  int,
        patch_size:  int,
        in_channels: int = 3,
        d_model:     int = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0, \
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"

        self.patch_size  = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim   = patch_size * patch_size * in_channels

        # Linear projection of flattened patch pixels → d_model
        self.projection  = nn.Linear(self.patch_dim, d_model)
        self.norm        = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)

        # Learned 1D positional embedding over the fixed patch grid
        # Shape: [1, num_patches, d_model] — added to every item in the batch
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W]
        Returns:
            tokens: [B, num_patches, d_model]
        """
        B, C, H, W = images.shape
        P = self.patch_size

        # Rearrange into patches: [B, num_patches, patch_dim]
        # unfold extracts sliding windows; with stride=P and size=P this gives
        # non-overlapping patches across both spatial dimensions.
        x = images.unfold(2, P, P).unfold(3, P, P)
        # x: [B, C, H/P, W/P, P, P]
        x = x.contiguous().view(B, C, -1, P, P)
        # x: [B, C, num_patches, P, P]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # x: [B, num_patches, C, P, P]
        x = x.view(B, self.num_patches, -1)
        # x: [B, num_patches, patch_dim]

        # Project to d_model and add positional embedding
        tokens = self.dropout(self.norm(self.projection(x)))
        tokens = tokens + self.pos_embedding

        return tokens   # [B, num_patches, d_model]


# =============================================================================
# ViT MODEL
# =============================================================================

class ViTBaseline(nn.Module):
    """
    Vanilla Vision Transformer classifier.

    Architecture is intentionally identical to MooreTransformer except for
    tokenization:
      - Same Transformer encoder (d_model, num_layers, num_heads, ffn_dim)
      - Same CLS token + linear classification head
      - Same pre-norm, GELU, dropout settings

    The tokenizer uses a fixed patch grid with learned 1D positional embeddings,
    which is the standard ViT approach.

    Args:
        image_size:  spatial dimension of input image
        patch_size:  side length of each square patch
        num_classes: number of output classes
        d_model:     transformer hidden dimension
        num_heads:   number of attention heads
        num_layers:  number of transformer encoder layers
        ffn_dim:     feedforward network hidden dimension
        dropout:     dropout rate
        in_channels: number of input channels
    """
    def __init__(
        self,
        image_size:  int   = 96,
        patch_size:  int   = 24,
        num_classes: int   = 10,
        d_model:     int   = 256,
        num_heads:   int   = 8,
        num_layers:  int   = 6,
        ffn_dim:     int   = 1024,
        dropout:     float = 0.1,
        in_channels: int   = 3,
    ):
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        # Tokenizer — the only part that differs from MooreTransformer
        self.tokenizer = ViTPatchTokenizer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder — identical config to MooreTransformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # pre-norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Classification head — identical to MooreTransformer
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W]
        Returns:
            logits: [B, num_classes]
        """
        B = images.shape[0]

        # Tokenize image into patch sequence
        tokens = self.tokenizer(images)             # [B, num_patches, d_model]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)     # [B, 1, d_model]
        tokens = torch.cat([cls, tokens], dim=1)   # [B, num_patches+1, d_model]

        # Transformer encoder
        encoded = self.transformer(tokens)          # [B, num_patches+1, d_model]

        # Classify from CLS token
        logits = self.classifier(encoded[:, 0, :]) # [B, num_classes]
        return logits


# =============================================================================
# DATASET — identical to MooreTransformer
# =============================================================================

class PatchDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, root: str, split: str, augment: bool = True):
        assert split in ("train", "test")
        if augment and split == "train":
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(self.MEAN, self.STD),
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(self.MEAN, self.STD),
            ])
        self.dataset = torchvision.datasets.STL10(
            root=root, split=split, download=True, transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# =============================================================================
# LR SCHEDULE — identical to MooreTransformer
# =============================================================================

def build_lr_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps  = cfg["epochs"]        * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_frac = cfg["min_lr"] / cfg["lr"]
        return min_frac + (1.0 - min_frac) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# LOGGING & PLOTTING — identical to MooreTransformer
# =============================================================================

class TrainingLogger:
    FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s"]

    def __init__(self, output_dir: str):
        self.log_dir  = Path(output_dir) / "logs"
        self.plot_dir = Path(output_dir) / "plots"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "train_log.csv"
        self.history  = {f: [] for f in self.FIELDS}
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, metrics: dict):
        for k in self.FIELDS:
            self.history[k].append(metrics.get(k, float("nan")))
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(metrics)

    def plot(self):
        epochs = self.history["epoch"]
        if len(epochs) < 2:
            return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("ViT Baseline Training", fontsize=13, fontweight="bold")

        axes[0].plot(epochs, self.history["train_loss"], label="Train", linewidth=2)
        axes[0].plot(epochs, self.history["val_loss"],   label="Val",   linewidth=2)
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history["train_acc"], label="Train", linewidth=2)
        axes[1].plot(epochs, self.history["val_acc"],   label="Val",   linewidth=2)
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)"); axes[1].set_ylim(0, 100)
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.history["lr"], color="green", linewidth=2)
        axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR"); axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_dir / "training_curves.png", dpi=120)
        plt.close(fig)


def print_header(cfg: dict, model: nn.Module, device: torch.device):
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_patches      = model.tokenizer.num_patches
    patch_dim        = model.tokenizer.patch_dim

    print("=" * 65)
    print("  ViT Baseline — Training Run")
    print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device:     {device}")
    print("-" * 65)
    print("  Model")
    print(f"    d_model   {cfg['d_model']}    num_layers  {cfg['num_layers']}")
    print(f"    num_heads {cfg['num_heads']}     ffn_dim     {cfg['ffn_dim']}")
    print(f"    patch_size {cfg['patch_size']}   num_patches {num_patches}")
    print(f"    patch_dim  {patch_dim}  (patch_size^2 * 3)")
    print(f"    dropout    {cfg['dropout']}")
    print(f"    Total params:     {total_params:>12,}")
    print(f"    Trainable params: {trainable_params:>12,}")
    print("-" * 65)
    print("  Training")
    print(f"    epochs      {cfg['epochs']}     batch_size  {cfg['batch_size']}")
    print(f"    lr          {cfg['lr']}   weight_decay {cfg['weight_decay']}")
    print(f"    warmup      {cfg['warmup_epochs']} epochs   grad_clip   {cfg['grad_clip']}")
    print("-" * 65)
    print("  Output")
    print(f"    {cfg['output_dir']}")
    print("=" * 65)


def print_epoch_summary(epoch, epochs, train_loss, train_acc, val_loss, val_acc, lr, elapsed, is_best):
    flag = " ◀ best" if is_best else ""
    print(
        f"  Epoch [{epoch:>3}/{epochs}]  "
        f"train loss {train_loss:.4f}  acc {train_acc:5.1f}%  │  "
        f"val loss {val_loss:.4f}  acc {val_acc:5.1f}%  │  "
        f"lr {lr:.2e}  ({elapsed:.1f}s){flag}"
    )


# =============================================================================
# TRAIN / EVAL LOOPS
# =============================================================================

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device:    torch.device,
    cfg:       dict,
    is_train:  bool,
) -> tuple[float, float]:
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    step_times = []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for step, (images, labels) in enumerate(loader):
            t0 = time.perf_counter()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if cfg["grad_clip"] is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                scheduler.step()

            preds          = logits.argmax(dim=-1)
            correct        = (preds == labels).sum().item()
            total_correct += correct
            total_loss    += loss.item() * images.size(0)
            total_samples += images.size(0)
            step_times.append(time.perf_counter() - t0)

            if is_train and (step + 1) % cfg["print_every_n_steps"] == 0:
                batch_acc  = 100.0 * correct / images.size(0)
                avg_ms     = 1000 * sum(step_times[-cfg["print_every_n_steps"]:]) / cfg["print_every_n_steps"]
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"    step [{step+1:>4}/{len(loader)}]  "
                    f"loss {loss.item():.4f}  "
                    f"batch_acc {batch_acc:5.1f}%  "
                    f"lr {current_lr:.2e}  "
                    f"({avg_ms:.1f} ms/step)"
                )

    return total_loss / total_samples, 100.0 * total_correct / total_samples


# =============================================================================
# CHECKPOINTING — identical to MooreTransformer
# =============================================================================

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_acc, cfg):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "sched_state":  scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "config":       cfg,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    print(f"  Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    scheduler.load_state_dict(ckpt["sched_state"])
    start_epoch  = ckpt["epoch"] + 1
    best_val_acc = ckpt["best_val_acc"]
    print(f"  Resuming from epoch {start_epoch}, best val acc {best_val_acc:.2f}%")
    return start_epoch, best_val_acc


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = VIT_CONFIG

    # --- Reproducibility ---
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    # --- Device ---
    device = torch.device(
        "cuda" if torch.cuda.is_available()       else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # --- Output directories ---
    ckpt_dir = Path(cfg["output_dir"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Datasets ---
    print("\nLoading STL-10...")
    train_dataset = PatchDataset(cfg["data_dir"], split="train", augment=True)
    val_dataset   = PatchDataset(cfg["data_dir"], split="test",  augment=False)

    # Same 10k/3k re-split as MooreTransformer for a fair comparison
    combined = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    n_train  = 10000
    n_val    = len(combined) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)   # same seed → same split
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    print(f"  Train: {len(train_dataset):,} images | Val: {len(val_dataset):,} images")
    print(f"  Steps per epoch: {len(train_loader)}")

    # --- Model ---
    model = ViTBaseline(
        image_size  = cfg["image_size"],
        patch_size  = cfg["patch_size"],
        num_classes = cfg["num_classes"],
        d_model     = cfg["d_model"],
        num_heads   = cfg["num_heads"],
        num_layers  = cfg["num_layers"],
        ffn_dim     = cfg["ffn_dim"],
        dropout     = cfg["dropout"],
    ).to(device)

    # --- Optimizer & scheduler ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=cfg["betas"],
    )
    scheduler = build_lr_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Resume ---
    start_epoch, best_val_acc = 1, 0.0
    best_epoch = start_epoch
    if cfg["resume_from"] is not None:
        start_epoch, best_val_acc = load_checkpoint(
            cfg["resume_from"], model, optimizer, scheduler, device
        )
        for _ in range((start_epoch - 1) * len(train_loader)):
            scheduler.step()

    # --- Logger ---
    logger = TrainingLogger(cfg["output_dir"])
    print_header(cfg, model, device)

    # --- Training loop ---
    print("\nTraining...\n")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t_start = time.perf_counter()

        print(f"\n{'─'*65}")
        print(f"  Epoch {epoch}/{cfg['epochs']}")
        print(f"{'─'*65}")

        print("  [Train]")
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, cfg, is_train=True,
        )

        print("  [Val]")
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, scheduler,
            criterion, device, cfg, is_train=False,
        )

        elapsed  = time.perf_counter() - t_start
        curr_lr  = scheduler.get_last_lr()[0]
        is_best  = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                str(ckpt_dir / "best_model.pt"),
                epoch, model, optimizer, scheduler, best_val_acc, cfg,
            )

        if epoch % cfg["save_every_n_epochs"] == 0:
            save_checkpoint(
                str(ckpt_dir / f"epoch_{epoch:04d}.pt"),
                epoch, model, optimizer, scheduler, best_val_acc, cfg,
            )

        metrics = {
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "train_acc":    round(train_acc,  4),
            "val_loss":     round(val_loss,   6),
            "val_acc":      round(val_acc,    4),
            "lr":           round(curr_lr,    8),
            "epoch_time_s": round(elapsed,    2),
        }
        logger.log(metrics)
        logger.plot()

        print_epoch_summary(
            epoch, cfg["epochs"],
            train_loss, train_acc,
            val_loss,   val_acc,
            curr_lr, elapsed, is_best,
        )

    print("\n" + "=" * 65)
    print(f"  Training complete.")
    print(f"  Best val accuracy: {best_val_acc:.2f}%")
    print(f"  Checkpoints:       {ckpt_dir}")
    print(f"  Log:               {logger.csv_path}")
    print(f"  Plots:             {logger.plot_dir}")
    print("=" * 65 + "\n")

    # Append Master Log
    log_experiment(
            filepath = CONFIG["master_log"],
            data = {
                "run_name":    CONFIG["test_name"],
                "model":        "Vanilla ViT",
                "patch_sampling_method": "Non-overlapping fixed grid",
                "num_patches":  CONFIG["num_patches"],
                "patch_representation": "RxR (Standard Row by Row)",
                "best_val_acc": best_val_acc,
                "final_val_acc": metrics["val_acc"],
                "best_epoch": best_epoch,
                "train_split": len(train_dataset),
                "val_split": len(val_dataset),
                "total_params": sum(p.numel() for p in model.parameters()),
                "notes":        CONFIG["test_note"],
                **{k: CONFIG[k] for k in [
                    "patch_dim","epochs","d_model",
                    "num_heads","num_layers","ffn_dim","dropout",
                    "batch_size","lr","weight_decay","warmup_epochs",
                ]},
            },
        )


if __name__ == "__main__":
    main()
