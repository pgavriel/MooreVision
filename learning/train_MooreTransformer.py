"""
train.py — Training script for MooreTransformer on STL-10

Usage:
    python train.py

All hyperparameters are defined in the CONFIG block below.
Outputs (all written to CONFIG["output_dir"]):
    - checkpoints/best_model.pt       best validation accuracy checkpoint
    - checkpoints/epoch_{N}.pt        periodic checkpoints
    - logs/train_log.csv              per-epoch metrics
    - plots/loss_curve.png            live-updated loss & accuracy plots
    - plots/lr_schedule.png           learning rate schedule visualization
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
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import torchvision
import torchvision.transforms as T

import matplotlib
# matplotlib.use("Agg")   # non-interactive backend, safe for servers
import matplotlib.pyplot as plt
import numpy as np

from model_MooreTransformer import MooreTransformer
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir,".."))

sys.path.insert(0, os.path.join(script_dir,"../moore_curve"))
from moore_curve.FocusCurve import Focus
from utils import *


# =============================================================================
# CONFIG — all hyperparameters and settings in one place
# =============================================================================
from config import CONFIG

# --- Focus Curve Global Declaration --- 
f = Focus(iter=CONFIG["curve_iter"],pos=[0,0],mode=CONFIG["curve_mode"],mem=CONFIG["batch_size"])
f.set_size(CONFIG["image_size"])
coords_tensor = torch.tensor(f.coords, dtype=torch.float32)  # [256, 2] # Compute this only once

FIRST_BATCH = True

# =============================================================================
# PATCH EXTRACTION — fill in your implementation here
# =============================================================================

def extract_patches_and_bboxes(
    images: torch.Tensor,
    f: Focus,
    num_patches: int,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract N patches from each image and return their representations
    alongside normalized [cx, cx, size] information.

    Args:
        images:      [B, 3, H, W]  normalized image batch
        num_patches: N patches to sample per image
        min_scale:   minimum patch size as fraction of image_size
        max_scale:   maximum patch size as fraction of image_size
        image_size:  spatial dimension of the image (assumes square)

    Returns:
        patches:  [B, N, patch_dim]  your 1D patch representations
        bboxes:   [B, N, 4]          (x_center, y_center, w, h) in [0, 1]

    Notes on bboxes:
        - All values should be in [0, 1] relative to image dimensions
        - x_center, y_center are the center of the patch
        - w, h are the width and height of the patch
        - Example: a patch covering the full image → (0.5, 0.5, 1.0, 1.0)
    """
    global coords_tensor, FIRST_BATCH
    B, C, H, W = images.shape
    BN = B * num_patches

    # 1. Sample and validate all states at once — fully vectorized, on CPU or GPU
    valid_states = generate_validate_states_batched(BN,CONFIG["image_size"],f.walker.width,f.walker.step_size,f.min_size)
    if FIRST_BATCH:
        print(f"FIRST BATCH:\nUnique K_Sizes: {valid_states['k_size'].unique()}")
        FIRST_BATCH = False

    # 2. Transform curve coords for all BN patches at once
    #    coords [256, 2] × scale [BN] + pos [BN, 2] → [BN, 256, 2]
    pos   = valid_states["pos"].float()      # [BN, 2]
    scale = valid_states["scale"]            # [BN]
    coords_scaled = (
        coords_tensor[None, :, :] * scale[:, None, None]  # [BN, 256, 2]
        + pos[:, None, :]                                 # broadcast pos
    )
    # 3. Normalize to [-1, 1] for grid_sample
    # coords_scaled is in pixel space [0, image_size]
    grid = (coords_scaled / (image_size - 1)) * 2 - 1   # [BN, 256, 2]
    grid = grid.unsqueeze(2)                            # [BN, 256, 1, 2]

    # 4. Sample the image at all curve points
    images_exp = images.repeat_interleave(num_patches, dim=0)  # [BN, C, H, W]

    sampled = F.grid_sample(
        images_exp, grid,
        mode="bilinear",
        padding_mode="border",              # edge pixels clamp rather than zero-pad
        align_corners=True,
    )                                       # [BN, C, 256, 1]
    sampled = sampled.squeeze(-1)           # [BN, C, 256]

    # 5. Apply variable kernel size sampling, reshape to final patch tensor
    sampled = apply_variable_kernel(sampled, valid_states["k_size"])    # [BN, C, 256]
    sampled = sampled.permute(0, 2, 1)                                  # [BN, 256, C]
    patches = sampled.flatten(start_dim=1)                              # [BN, 768]
    # Final patches tensor
    patches = patches.reshape(B, num_patches, -1)                       # [B, N, 768]

    # 6. Build states tensor
    # All values normalized to [0, 1] except 1/(ks+1) which is already in (0, 1]
    pos_norm  = valid_states["pos"].float() / image_size        # [BN, 2]  x, y
    size_norm = valid_states["size"].float() / image_size       # [BN]
    ks_enc    = 1.0 / (valid_states["k_size"].float() + 1.0)   # [BN]  1/(ks+1)

    states = torch.stack([
        pos_norm[:, 0],   # norm x
        pos_norm[:, 1],   # norm y
        size_norm,        # norm size
        ks_enc,           # 1/(ks+1)
    ], dim=-1)                                                   # [BN, 4]

    # Final states tensor
    states = states.reshape(B, num_patches, 4)                  # [B, N, 4]

    # OLD IMPLEMENTATION: (KEEP FOR NOW)
    # all_patches = []
    # all_states  = []

    # for b in range(B):
    #     img = images[b]
    #     img_hwc = img.permute(1, 2, 0)
    #     # print(f"img shape: {img_hwc.shape} , {type(img_hwc)}")
    #     img_patches, img_states = f.sample_random_views(img_hwc,CONFIG["num_patches"])
        
    #     # Convert states: list of [4] → tensor [N, 4]
    #     img_states_tensor = torch.tensor(img_states, dtype=torch.float32)

    #     # Convert patches: list of (256, 3) numpy arrays → flatten → tensor [N, 768]
    #     img_patches_tensor = torch.tensor(
    #         np.stack(img_patches),          # [N, 256, 3]
    #         dtype=torch.float32
    #     ).flatten(start_dim=1)              # [N, 768]
    #     # print("States min-max:",img_states_tensor.min(), img_states_tensor.max())
    #     all_patches.append(img_patches_tensor)   # [N, patch_dim]
    #     all_states.append(img_states_tensor)     # [N, 4]

    # patches = torch.stack(all_patches)     # [B, N, patch_dim]
    # states  = torch.stack(all_states)      # [B, N, 4]

    return patches, states

def precompute_val_views(
    val_loader: DataLoader,
    device: torch.device,
    cache_path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns precomputed (patches, states, labels) for the full validation set.
    Saves to cache_path so it's identical across training runs.
    """
    global f, coords_tensor
    if Path(cache_path).exists():
        print(f"  Loading cached val views from {cache_path}")
        cached = torch.load(cache_path)
        return cached["patches"], cached["states"], cached["labels"]

    print("  Precomputing validation views (once)...")
    torch.manual_seed(CONFIG["seed"])    # deterministic across runs

    all_patches, all_states, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            B = images.shape[0]
            BN = B * CONFIG["num_patches"]

            # valid      = generate_validate_states_batched(BN,CONFIG["image_size"],f.walker.width,f.walker.step_size,f.min_size)
            patches, states = extract_patches_and_bboxes(
                images,
                f,
                num_patches=CONFIG["num_patches"],
                image_size=CONFIG["image_size"],
            )
            all_patches.append(patches.cpu())
            all_states.append(states.cpu())
            all_labels.append(labels)

    patches = torch.cat(all_patches)   # [N_val, N, patch_dim]
    states  = torch.cat(all_states)    # [N_val, N, 4]
    labels  = torch.cat(all_labels)    # [N_val]

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"patches": patches, "states": states, "labels": labels}, cache_path)
    print(f"  Saved to {cache_path}")

    return patches, states, labels

# =============================================================================
# DATASET WRAPPER
# =============================================================================

class PatchDataset(Dataset):
    """
    Wraps a torchvision STL-10 dataset and returns image tensors alongside
    their labels. Patch extraction happens in the training loop on the GPU
    (or CPU) rather than here, so DataLoader workers stay lightweight.
    """

    # ImageNet statistics — appropriate for natural image datasets
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
            root=root,
            split=split,
            download=True,
            transform=transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def build_lr_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """
    Cosine annealing with linear warmup.
    Returns a per-step lambda scheduler.
    """
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps  = cfg["epochs"]        * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / max(warmup_steps, 1)
        # Cosine decay from lr → min_lr
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_frac = cfg["min_lr"] / cfg["lr"]
        return min_frac + (1.0 - min_frac) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# LOGGING & PLOTTING
# =============================================================================

class TrainingLogger:
    """Handles CSV logging and matplotlib plot generation."""

    FIELDS = [
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_acc", "lr", "epoch_time_s"
    ]

    def __init__(self, output_dir: str):
        self.log_dir   = Path(output_dir) / "logs"
        self.plot_dir  = Path(output_dir) / "plots"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "train_log.csv"
        self.history  = {f: [] for f in self.FIELDS}

        # Write CSV header
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
        fig.suptitle("MooreTransformer Training", fontsize=13, fontweight="bold")

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], label="Train", linewidth=2)
        axes[0].plot(epochs, self.history["val_loss"],   label="Val",   linewidth=2)
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, self.history["train_acc"], label="Train", linewidth=2)
        axes[1].plot(epochs, self.history["val_acc"],   label="Val",   linewidth=2)
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning rate
        axes[2].plot(epochs, self.history["lr"], color="green", linewidth=2)
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_dir / "training_curves.png", dpi=120)
        plt.close(fig)


def print_header(cfg: dict, model: nn.Module, device: torch.device):
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 65)
    print("  MooreTransformer — Training Run")
    print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device:     {device}")
    print("-" * 65)
    print("  Model")
    print(f"    d_model   {cfg['d_model']}    num_layers  {cfg['num_layers']}")
    print(f"    num_heads {cfg['num_heads']}     ffn_dim     {cfg['ffn_dim']}")
    print(f"    dropout   {cfg['dropout']}   patch_dim   {cfg['patch_dim']}")
    print(f"    Total params:     {total_params:>12,}")
    print(f"    Trainable params: {trainable_params:>12,}")
    print("-" * 65)
    print("  Training")
    print(f"    epochs      {cfg['epochs']}     batch_size  {cfg['batch_size']}")
    print(f"    lr          {cfg['lr']}   weight_decay {cfg['weight_decay']}")
    print(f"    warmup      {cfg['warmup_epochs']} epochs   grad_clip   {cfg['grad_clip']}")
    print(f"    num_patches {cfg['num_patches']}    patch_scale [{cfg['min_patch_scale']}, {cfg['max_patch_scale']}]")
    print("-" * 65)
    print("  Output")
    print(f"    {cfg['output_dir']}")
    print("=" * 65)


def print_epoch_summary(
    epoch: int, epochs: int,
    train_loss: float, train_acc: float,
    val_loss: float,   val_acc: float,
    lr: float, elapsed: float, is_best: bool
):
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
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   optim.Optimizer,
    scheduler,
    criterion:   nn.Module,
    device:      torch.device,
    cfg:         dict,
    epoch:       int,
    is_train:    bool,
) -> tuple[float, float]:
    """
    Run one full pass over the dataset.
    Returns (mean_loss, accuracy_percent).
    """
    global f
    model.train() if is_train else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    step_times    = []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for step, (images, labels) in enumerate(loader):
            t0 = time.perf_counter()
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # --- Extract patches and bboxes ---
            patches, states = extract_patches_and_bboxes(
                images,
                f,
                num_patches=cfg["num_patches"],
                image_size=cfg["image_size"],
            )
            patches = patches.to(device, non_blocking=True)
            states  = states.to(device,  non_blocking=True)

            # --- Forward pass ---
            logits = model(patches, states)
            loss   = criterion(logits, labels)

            # --- Backward pass ---
            if is_train:
                optimizer.zero_grad()
                loss.backward()

                if cfg["grad_clip"] is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

                optimizer.step()
                scheduler.step()

            # --- Metrics ---
            preds         = logits.argmax(dim=-1)
            correct       = (preds == labels).sum().item()
            total_correct += correct
            total_loss    += loss.item() * images.size(0)
            total_samples += images.size(0)
            step_times.append(time.perf_counter() - t0)

            # --- Batch-level printout ---
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

    mean_loss = total_loss    / total_samples
    accuracy  = 100.0 * total_correct / total_samples
    return mean_loss, accuracy

def run_val_epoch(
    model:      nn.Module,
    loader:     DataLoader,    # yields (patches, states, labels)
    criterion:  nn.Module,
    device:     torch.device,
    cfg:        dict,
) -> tuple[float, float]:

    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for step, (patches, states, labels) in enumerate(loader):
            patches = patches.to(device, non_blocking=True)
            states  = states.to(device,  non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)

            logits  = model(patches, states)
            loss    = criterion(logits, labels)

            total_correct += (logits.argmax(-1) == labels).sum().item()
            total_loss    += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            if (step + 1) % cfg["print_every_n_steps"] == 0:
                print(
                    f"    step [{step+1:>4}/{len(loader)}]  "
                    f"loss {loss.item():.4f}  "
                    f"acc {100.*total_correct/total_samples:5.1f}%"
                )

    return total_loss / total_samples, 100. * total_correct / total_samples

# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    best_val_acc: float,
    cfg: dict,
):
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "sched_state":  scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "config":       cfg,
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, float]:
    """Returns (start_epoch, best_val_acc)."""
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
    global f, coords_tensor
    cfg = CONFIG

    # --- Reproducibility ---
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    # --- Device ---
    device = torch.device(
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )

    # --- Output directories ---
    run_dir   = Path(cfg["output_dir"])
    ckpt_dir  = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Focus Curve --- 
    coords_tensor = coords_tensor.to(device)
    # example_states = generate_validate_states_batched(2500,CONFIG["image_size"],f.walker.width,f.walker.step_size,f.min_size)
    # plot_state_debug(example_states,CONFIG["image_size"],None)
    # sys.exit()

    # --- Datasets & loaders ---
    print("\nLoading STL-10...")
    train_dataset = PatchDataset(cfg["data_dir"], split="train", augment=True)
    val_dataset   = PatchDataset(cfg["data_dir"], split="test",  augment=False)

    # --- VISUALIZE DATASET SAMPLES:
    visualize_patch_samples2(train_dataset,f,8,16)
    
    visualize_patch_samples2(train_dataset,f,8,16)
    sys.exit()

    # NOTE: THIS DEVIATES FROM THE 5k/8k split native to the dataset, after running experiments, maybe change it back for comparison
    # Re-split 10k/3k
    combined = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    n_train = 10000
    n_val   = len(combined) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,     # keeps batch size consistent for norm layers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    print(f"  Train: {len(train_dataset):,} images | Val: {len(val_dataset):,} images")
    print(f"  Steps per epoch: {len(train_loader)}")

    # --- Precompute/load Val views ONCE before the training loop ---
    print("\nPrecomputing validation views...")
    val_patches, val_states, val_labels = precompute_val_views(
        val_loader=val_loader,
        device=device,
        cache_path=cfg["val_cache_path"],
    )
    val_tensor_loader = DataLoader(
        TensorDataset(val_patches, val_states, val_labels),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,       # already tensors in memory, workers add overhead
        pin_memory=True,
    )
    # --- Model ---
    model = MooreTransformer(
        num_classes=cfg["num_classes"],
        patch_dim=cfg["patch_dim"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ffn_dim=cfg["ffn_dim"],
        dropout=cfg["dropout"],
        max_patches=cfg["num_patches"],
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

    # --- Resume from checkpoint if specified ---
    start_epoch  = 1
    best_val_acc = 0.0
    if cfg["resume_from"] is not None:
        start_epoch, best_val_acc = load_checkpoint(
            cfg["resume_from"], model, optimizer, scheduler, device
        )
        # Fast-forward scheduler to correct step
        for _ in range((start_epoch - 1) * len(train_loader)):
            scheduler.step()

    # --- Logger ---
    logger = TrainingLogger(cfg["output_dir"])

    # --- Print run summary ---
    print_header(cfg, model, device)

    # --- Training loop ---
    print("\nTraining...\n")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t_start = time.perf_counter()

        print(f"\n{'─'*65}")
        print(f"  Epoch {epoch}/{cfg['epochs']}")
        print(f"{'─'*65}")

        # Train
        print("  [Train]")
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, cfg, epoch, is_train=True
        )

        # Validate
        print("  [Val]")
        val_loss, val_acc = run_val_epoch(
            model, val_tensor_loader, criterion, device, cfg
        )

        elapsed  = time.perf_counter() - t_start
        curr_lr  = scheduler.get_last_lr()[0]
        is_best  = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc
            save_checkpoint(
                str(ckpt_dir / "best_model.pt"),
                epoch, model, optimizer, scheduler, best_val_acc, cfg
            )

        # Periodic checkpoint
        if epoch % cfg["save_every_n_epochs"] == 0:
            save_checkpoint(
                str(ckpt_dir / f"epoch_{epoch:04d}.pt"),
                epoch, model, optimizer, scheduler, best_val_acc, cfg
            )

        # Log
        metrics = {
            "epoch":         epoch,
            "train_loss":    round(train_loss, 6),
            "train_acc":     round(train_acc,  4),
            "val_loss":      round(val_loss,   6),
            "val_acc":       round(val_acc,    4),
            "lr":            round(curr_lr,    8),
            "epoch_time_s":  round(elapsed,    2),
        }
        logger.log(metrics)
        logger.plot()

        print_epoch_summary(
            epoch, cfg["epochs"],
            train_loss, train_acc,
            val_loss,   val_acc,
            curr_lr, elapsed, is_best
        )

    # --- Final summary ---
    print("\n" + "=" * 65)
    print(f"  Training complete.")
    print(f"  Best val accuracy: {best_val_acc:.2f}%")
    print(f"  Checkpoints:       {ckpt_dir}")
    print(f"  Log:               {logger.csv_path}")
    print(f"  Plots:             {logger.plot_dir}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()