import torch
import torch.nn.functional as F
import csv
import os
from datetime import datetime
from pathlib import Path

def create_incremental_dir(root, prefix="test", digits=3):
    os.makedirs(root, exist_ok=True)  # Ensure root exists
    i = 1
    while True:
        new_dir = os.path.join(root, f"{prefix}_{i:0{digits}}")
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        i += 1

def apply_variable_kernel(
    sampled: torch.Tensor,      # [BN, C, 256]
    k_sizes: torch.Tensor,      # [BN]  integer kernel radii
) -> torch.Tensor:
    
    output = sampled.clone()
    unique_ks = k_sizes.unique()

    for ks in unique_ks:
        ks = ks.item()
        if ks == 0:
            continue                        # no averaging needed

        mask = (k_sizes == ks)              # [BN] bool
        group = sampled[mask]               # [G, C, 256]

        kernel = ks * 2
        padded = F.pad(group, (kernel//2, kernel//2), mode="reflect")
        averaged = F.avg_pool1d(
            padded,
            kernel_size=kernel,
            stride=1,
            padding=0,
        )[:, :, :256]                       # [G, C, 256]

        output[mask] = averaged

    return output                           # [BN, C, 256]

def generate_validate_states_batched(
    num_states: int,   # How many states to generate
    image_size: int,
    walker_width: int,          # focus.walker.width
    walker_step_size: float,    # focus.walker.step_size
    min_size: int,              # focus.min_size
    device
) -> dict:
    """
    Vectorized equivalent of set_state_normed → set_size → move_to → enforce_bounds.
    Returns a dict of validated state tensors, all shape [BN].
    """
    raw_states    = torch.rand(num_states, 3,device=device)

    norm_pos  = raw_states[:, :2]   # [BN, 2]
    norm_size = raw_states[:, 2]    # [BN]

    # --- Validate size (set_size) ---
    size = (image_size * norm_size).int()
    size = size.clamp(min=min_size, max=image_size)
    size = size + (size % 2)        # round up to even

    # --- Derive scale and k_size from size ---
    scale  = size.float() / float(walker_width)
    k_size = ((walker_step_size * scale) / 2).int().clamp(min=0)

    # --- Validate position (enforce_bounds) ---
    # pos is CENTER of patch, not top-left
    half = (size // 2).unsqueeze(1)                          # [BN, 1]
    pos  = (image_size * norm_pos).int()                     # [BN, 2]
    pos  = pos.clamp(
        min=half,                                            # [BN, 1] broadcasts over x,y
        max=(image_size - half)                              # [BN, 1] broadcasts over x,y
    )                                                        # [BN, 2]

    return {
        "pos":    pos,      # [BN, 2]  pixel-space top-left
        "size":   size,     # [BN]     pixel-space patch size
        "scale":  scale,    # [BN]     scale factor
        "k_size": k_size,   # [BN]     kernel radius
    }

def plot_state_debug(valid_states: dict, image_size: int, save_path: str = None):
    """
    valid_states: dict returned by validate_states_batched, with keys:
        pos    [BN, 2]
        size   [BN]
        scale  [BN]
        k_size [BN]
    """
    import matplotlib.pyplot as plt

    pos    = valid_states["pos"].cpu().numpy()
    size   = valid_states["size"].cpu().numpy()
    scale  = valid_states["scale"].cpu().numpy()
    ksize  = valid_states["k_size"].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Patch state distributions", fontsize=12, fontweight="bold")

    axes[0].scatter(pos[:, 0], pos[:, 1], s=4, alpha=0.4, color="#378ADD")
    axes[0].set_xlim(0, image_size)
    axes[0].set_ylim(0, image_size)
    axes[0].set_aspect("equal")
    axes[0].set_title("position (x, y)")
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")
    axes[0].grid(True, alpha=0.2)

    for ax, vals, title, xlabel in zip(
        axes[1:],
        [size, scale, ksize],
        ["size", "scale", "k_size"],
        ["pixels", "scale factor", "kernel radius"]
    ):
        ax.hist(vals, bins=30, color="#1D9E75", alpha=0.85, edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.axvline(vals.mean(), color="#D85A30", linewidth=1.5, linestyle="--",
                   label=f"mean {vals.mean():.2f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

from moore_curve.FocusCurve import Focus
def visualize_patch_samples(
    dataset,
    f:              Focus, #Focus object
    n_images:       int = 4,
    m_patches:      int = 8,
    save_path:      str = None,
    show:           bool = True,
):
    """
    For N random images, draws the original image with patch bounding boxes
    overlaid, and a grid of the extracted patch views alongside it.

    Args:
        dataset:       a PatchDataset (or any dataset returning (image, label))
        f:              Focus Curve object
        n_images:      number of images to visualize
        m_patches:     number of patches to sample per image
        save_path:     if set, saves the figure here instead of showing
        show:          call plt.show() if True
    """
    import cv2
    import random
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from pathlib import Path
    from train_MooreTransformer import extract_patches_and_bboxes

    # ImageNet denormalization constants
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def denormalize(tensor):
        """[C, H, W] normalized tensor → [H, W, C] uint8 numpy array"""
        img = tensor.cpu().numpy().transpose(1, 2, 0)   # [H, W, C]
        img = (img * STD + MEAN)                         # undo normalization
        img = np.clip(img, 0.0, 1.0)
        return (img * 255).astype(np.uint8)

    # Pick N random image indices
    indices = random.sample(range(len(dataset)), n_images)

    # Generate a distinct color per patch — evenly spaced around the hue wheel
    def patch_colors(n):
        colors_float = [
            tuple(int(c*255) for c in plt.cm.hsv(i / n)[:3])
            for i in range(n)
        ]
        return colors_float   # list of (R, G, B) uint8 tuples

    colors = patch_colors(m_patches)

    # Each image gets one row: [original+boxes | patch_0 | patch_1 | ... | patch_M-1]
    # We'll build each row as a fixed-height numpy image strip and vstack them.
    PATCH_DISPLAY_SIZE = 96     # each patch thumbnail rendered at this resolution
    BORDER             = 1      # colored border thickness in pixels
    H_strip            = PATCH_DISPLAY_SIZE + BORDER * 2

    row_images = []

    for idx in indices:
        image_tensor, label = dataset[idx]
        print(f"Image Tensor Shape: {image_tensor.shape} ({type(image_tensor)})")
        img_hw = denormalize(image_tensor)              # [H, W, 3] uint8
        img_h, img_w = img_hw.shape[:2]
        img_size = min(img_h,img_w)
        image_tensor_b = image_tensor.unsqueeze(0)
        # --- Sample and validate m_patches states ---
        patches, states = extract_patches_and_bboxes(
            image_tensor_b,
            f,
            num_patches=m_patches,
            image_size=img_size,
        )
        print(f"Patches Tensor Shape: {patches.shape} ({type(patches)})")
        # TODO: This doesn't seem to be working properly
        patches_np = patches.squeeze(0).cpu().numpy()   # [M, 768]
        patch_views = []
        for p in range(m_patches):
            view = patches_np[p].reshape(256, 3)        # [256, 3] float32, normalized
            view = (view * STD + MEAN)                  # denormalize per-channel
            view = np.clip(view, 0.0, 1.0)
            view = (view * 255).astype(np.uint8)        # [256, 3] uint8
            patch_views.append(view)
 
 
        # states: [1, M, 4]  (norm_x, norm_y, norm_size, 1/(ks+1))
        states = states.squeeze(0).cpu().numpy()        # [M, 4]

        # --- Draw bounding boxes on the original image ---
        img_annotated = img_hw.copy()
        for p in range(m_patches):
            

            cx   = states[p, 0] * img_w
            cy   = states[p, 1] * img_h
            size = states[p, 2] * img_size
            half = size / 2
            x1, y1 = int(cx - half), int(cy - half)
            x2, y2 = int(cx + half), int(cy + half)

            r, g, b = colors[p]
            cv2.rectangle(
                img_annotated,
                (x1, y1), (x2, y2),
                (b, g, r),          # OpenCV uses BGR
                thickness=1
            )
            # small patch index label
            cv2.putText(
                img_annotated, str(p),
                (max(x1+2, 0), max(y1+12, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (b, g, r), 1, cv2.LINE_AA
            )

        # Resize annotated image to strip height for consistent row layout
        scale_factor = H_strip / img_h
        orig_resized = cv2.resize(
            img_annotated,
            (int(img_w * scale_factor), H_strip),
            interpolation=cv2.INTER_LINEAR
        )

        # --- Build patch thumbnails ---
        # If you have a render function that accepts (256, 3) numpy view data,
        # call it here. Otherwise we reconstruct a square thumbnail from the
        # raw patch pixels via your existing sample data.
        patch_thumbs = []
        for p in range(m_patches):
            cx   = states[p, 0] * img_w
            cy   = states[p, 1] * img_h
            size = states[p, 2] * min(img_w, img_h)
            half = size / 2
            x1 = int(max(cx - half, 0))
            y1 = int(max(cy - half, 0))
            x2 = int(min(cx + half, img_w))
            y2 = int(min(cy + half, img_h))

            view_img = f.reconstruct(custom_mem=patch_views[p])
           
            thumb = cv2.resize(
                view_img,
                (PATCH_DISPLAY_SIZE, PATCH_DISPLAY_SIZE),
                interpolation=None
            )

            # Add colored border
            r, g, b = colors[p]
            thumb = cv2.copyMakeBorder(
                thumb, BORDER, BORDER, BORDER, BORDER,
                cv2.BORDER_CONSTANT,
                value=(b, g, r)     # BGR
            )
            patch_thumbs.append(thumb)

        # Concatenate all patch thumbs horizontally
        patches_strip = np.concatenate(patch_thumbs, axis=1)   # [H_strip, M*tile_w, 3]

        # Pad original to same height if needed, then hstack
        orig_padded = cv2.copyMakeBorder(
            orig_resized,
            0, max(0, H_strip - orig_resized.shape[0]),
            0, 0,
            cv2.BORDER_CONSTANT, value=(30, 30, 30)
        )

        # Separator line between original and patches
        sep = np.full((H_strip, 3, 3), 60, dtype=np.uint8)
        row = np.concatenate([orig_padded, sep, patches_strip], axis=1)
        row_images.append(row)

    # Pad all rows to the same width before vstacking
    max_w = max(r.shape[1] for r in row_images)
    padded_rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1],
                           cv2.BORDER_CONSTANT, value=(30, 30, 30))
        for r in row_images
    ]

    # Add a 2px gap between rows
    gap = np.full((2, max_w, 3), 45, dtype=np.uint8)
    final = padded_rows[0]
    for row in padded_rows[1:]:
        final = np.concatenate([final, gap, row], axis=0)

    # --- Plot ---
    fig_w = max_w / 96      # roughly scale figure to pixel count
    fig_h = final.shape[0] / 96
    fig, ax = plt.subplots(figsize=(max(fig_w, 10), max(fig_h, 3)))
    ax.imshow(final)
    # ax.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(
        f"Patch sampling visualization  —  {n_images} images × {m_patches} patches",
        fontsize=11, pad=8
    )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

def visualize_patch_samples2(
    dataset,
    f:              Focus,
    n_images:       int = 4,
    m_patches:      int = 8,
    save_path:      str = None,
    show:           bool = True,
):
    import cv2
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    from pathlib import Path
    from train_MooreTransformer import extract_patches_and_bboxes

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    STL10_CLASSES = ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]

    def save_confusion_matrix(
        all_preds:   np.ndarray,
        all_labels:  np.ndarray,
        epoch:       int,
        val_acc:     float,
        class_names: list,
        save_dir:    str,
    ):
        import matplotlib.pyplot as plt
        from pathlib import Path

        num_classes = len(class_names)
        cm          = np.zeros((num_classes, num_classes), dtype=np.int64)
        for pred, label in zip(all_preds, all_labels):
            cm[label, pred] += 1

        # Per-class accuracy
        per_class_acc = {}
        print(f"\n  Per-class accuracy (epoch {epoch}, val acc {val_acc:.2f}%):")
        for i, name in enumerate(class_names):
            total = cm[i].sum()
            acc   = 100.0 * cm[i, i] / total if total > 0 else 0.0
            per_class_acc[name] = acc
            print(f"    {name:<12} {acc:5.1f}%  ({cm[i,i]}/{total})")

        # Normalised confusion matrix for plotting
        cm_norm = cm.astype(np.float32)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(cm_norm, row_sums, where=row_sums > 0)

        fig, ax = plt.subplots(figsize=(max(8, num_classes), max(6, num_classes - 1)))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.set_title(
            f"Confusion Matrix — Epoch {epoch}  (val acc {val_acc:.2f}%)",
            fontsize=12, fontweight="bold", pad=12,
        )

        # Annotate cells with raw counts
        thresh = 0.5
        for i in range(num_classes):
            for j in range(num_classes):
                color = "white" if cm_norm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=8, color=color)

        plt.tight_layout()
        save_path = Path(save_dir) / f"confusion_matrix_best_epoch{epoch:04d}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=130)
        plt.close(fig)
        print(f"  Saved confusion matrix → {save_path}")

        return cm, per_class_acc

    def denormalize(tensor):
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * STD + MEAN, 0.0, 1.0)
        return (img * 255).astype(np.uint8)

    indices = random.sample(range(len(dataset)), n_images)

    def patch_colors(n):
        return [tuple(int(c*255) for c in plt.cm.hsv(i / n)[:3]) for i in range(n)]

    colors = patch_colors(m_patches)

    PATCH_DISPLAY_SIZE = 96
    BORDER             = 2
    TILE               = PATCH_DISPLAY_SIZE + BORDER * 2    # single patch tile size
    LABEL_BAR          = 20                                 # px height for class label text

    # Original image is displayed at 2x tile size
    ORIG_DISPLAY_SIZE  = TILE * 2                           # e.g. 200px square
    ORIG_WITH_LABEL    = ORIG_DISPLAY_SIZE + LABEL_BAR      # original + label bar height

    # patches_per_row — split m_patches evenly across 2 rows, rounding up for row 0
    patches_per_row = math.ceil(m_patches / 2)
    strip_w = patches_per_row * TILE                        # width of patch strip

    row_images = []

    for idx in indices:
        image_tensor, label = dataset[idx]
        label_name = STL10_CLASSES[label] if label < len(STL10_CLASSES) else str(label)
        print(f"LABEL: {label_name}")
        img_hw   = denormalize(image_tensor)
        img_h, img_w = img_hw.shape[:2]
        img_size = min(img_h, img_w)

        # --- Extract patches ---
        patches, states = extract_patches_and_bboxes(
            image_tensor.unsqueeze(0), f,
            num_patches=m_patches, image_size=img_size,
        )
        patches_np = patches.squeeze(0).cpu().numpy()   # [M, 768]
        states     = states.squeeze(0).cpu().numpy()    # [M, 4]

        patch_views = []
        for p in range(m_patches):
            view = patches_np[p].reshape(256, 3)
            view = np.clip(view * STD + MEAN, 0.0, 1.0)
            view = (view * 255).astype(np.uint8)
            # view = view[:, ::-1]                        # RGB → BGR
            patch_views.append(view)

        # --- Annotated original image ---
        img_annotated = img_hw.copy()
        for p in range(m_patches):
            print(f"[State {p+1}] {states[p]}")
            cx   = states[p, 0] * img_w
            cy   = states[p, 1] * img_h
            size = states[p, 2] * img_size
            half = size / 2
            x1, y1 = int(cx - half), int(cy - half)
            x2, y2 = int(cx + half), int(cy + half)
            r, g, b = colors[p]
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (b, g, r), thickness=1)
            # cv2.putText(img_annotated, str(p), (max(x1+2, 0), max(y1+12, 0)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (b, g, r), 1, cv2.LINE_AA)

        # Resize original to ORIG_DISPLAY_SIZE × ORIG_DISPLAY_SIZE
        orig_resized = cv2.resize(img_annotated,
                                  (ORIG_DISPLAY_SIZE, ORIG_DISPLAY_SIZE),
                                  interpolation=cv2.INTER_LINEAR)

        # Add label bar below original image
        label_bar = np.full((LABEL_BAR, ORIG_DISPLAY_SIZE, 3), 30, dtype=np.uint8)
        cv2.putText(label_bar, label_name,
                    (4, LABEL_BAR - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        orig_block = np.concatenate([orig_resized, label_bar], axis=0)  # [ORIG_WITH_LABEL, ORIG_DISPLAY_SIZE, 3]

        # --- Build patch rows ---
        def build_patch_row(patch_indices):
            thumbs = []
            for p in patch_indices:
                view_img = f.reconstruct(custom_mem=patch_views[p])
                thumb    = cv2.resize(view_img, (PATCH_DISPLAY_SIZE, PATCH_DISPLAY_SIZE),
                                      interpolation=None)
                r, g, b  = colors[p]
                thumb    = cv2.copyMakeBorder(thumb, BORDER, BORDER, BORDER, BORDER,
                                              cv2.BORDER_CONSTANT, value=(b, g, r))
                thumbs.append(thumb)

            # Pad row to patches_per_row tiles if the last row is short
            while len(thumbs) < patches_per_row:
                thumbs.append(np.full((TILE, TILE, 3), 30, dtype=np.uint8))

            return np.concatenate(thumbs, axis=1)   # [TILE, strip_w, 3]

        row0_indices = list(range(0, patches_per_row))
        row1_indices = list(range(patches_per_row, m_patches))

        patch_row0 = build_patch_row(row0_indices)   # [TILE, strip_w, 3]
        patch_row1 = build_patch_row(row1_indices)   # [TILE, strip_w, 3]

        # Stack the two patch rows — total height is 2*TILE
        patch_block = np.concatenate([patch_row0, patch_row1], axis=0)  # [2*TILE, strip_w, 3]

        # Pad patch_block height to match orig_block (ORIG_WITH_LABEL) if needed
        h_diff = ORIG_WITH_LABEL - patch_block.shape[0]
        if h_diff > 0:
            pad = np.full((h_diff, strip_w, 3), 30, dtype=np.uint8)
            patch_block = np.concatenate([patch_block, pad], axis=0)
        elif h_diff < 0:
            # orig_block is shorter — pad it downward (shouldn't normally happen)
            pad = np.full((-h_diff, ORIG_DISPLAY_SIZE, 3), 30, dtype=np.uint8)
            orig_block = np.concatenate([orig_block, pad], axis=0)

        # Separator between original and patches
        sep = np.full((orig_block.shape[0], 3, 3), 60, dtype=np.uint8)
        row = np.concatenate([orig_block, sep, patch_block], axis=1)
        row_images.append(row)

    # Pad all rows to same width and vstack with gap
    max_w = max(r.shape[1] for r in row_images)
    padded_rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1],
                           cv2.BORDER_CONSTANT, value=(30, 30, 30))
        for r in row_images
    ]
    gap   = np.full((3, max_w, 3), 45, dtype=np.uint8)
    final = padded_rows[0]
    for row in padded_rows[1:]:
        final = np.concatenate([final, gap, row], axis=0)

    fig_w = max_w / 96
    fig_h = final.shape[0] / 96
    fig, ax = plt.subplots(figsize=(max(fig_w, 10), max(fig_h, 3)))
    ax.imshow(final)
    ax.axis("off")
    ax.set_title(
        f"Row by Row Curve Patch sampling visualization  —  {n_images} images × {m_patches} patches",
        fontsize=11, pad=8,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    


def log_experiment(
    filepath:   str,
    data:       dict,
    headers:    list = None,
):
    """
    Appends a single experiment result row to a master CSV log.
    Creates the file with a header row if it does not already exist.

    Args:
        filepath:  path to the master CSV log file
        data:      dict mapping header names to values for this run.
                   Missing keys are written as empty strings.
                   Extra keys not in headers are silently ignored.
        headers:   list of column names. If None, DEFAULT_HEADERS is used.
                   Only relevant when creating a new file — existing files
                   retain whatever headers they were created with.

    Example:
        log_experiment(
            filepath = "./runs/master_log.csv",
            data = {
                "run_name":     "moore_n16_d256_run1",
                "model":        "MooreTransformer",
                "patch_method": "moore_curve",
                "num_patches":  16,
                "best_val_acc": 57.3,
                "notes":        "baseline, no mixup",
                **{k: CONFIG[k] for k in [
                    "patch_dim","image_size","epochs","d_model",
                    "num_heads","num_layers","ffn_dim","dropout",
                    "batch_size","lr","weight_decay","warmup_epochs",
                ]},
            },
        )
    """
    DEFAULT_HEADERS = [
        "timestamp",
        "run_name",
        "model",
        "patch_sampling_method", # How view tokens are obtained      
        "num_patches",        # N patches per image (or num grid patches for ViT)
        "patch_representation", # e.g. "moore_curve", "vanilla_vit", "random_crop"
        "patch_dim",          # flattened patch vector length
        "epochs",
        "d_model",
        "num_heads",
        "num_layers",
        "ffn_dim",
        "dropout",
        "batch_size",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "train_split",        # e.g. 10000 or "official_5k"
        "val_split",          # e.g. 3000 or "official_8k"
        "best_val_acc",
        "final_val_acc",
        "best_epoch",
        "total_params",
        "notes",              # free text for anything not captured above
    ]
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine which headers to use
    if headers is None:
        headers = DEFAULT_HEADERS

    # Read existing headers from file if it already exists,
    # so we don't clobber a file created with a different header set
    if path.exists():
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_headers = next(reader, None)
        if existing_headers:
            headers = existing_headers

    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=headers,
            extrasaction="ignore",      # silently drop keys not in headers
        )

        # Write header row only when creating the file for the first time
        if not file_exists:
            writer.writeheader()

        # Fill in timestamp automatically if not provided
        row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        row.update(data)

        # Ensure all fields have a value — missing ones become empty string
        complete_row = {h: row.get(h, "") for h in headers}
        writer.writerow(complete_row)

    print(f"  Logged experiment to {path}  ({data.get('run_name', '?')})")

def plot_class_distribution(dataset, label: str = "Dataset", save_path: str = None, show: bool = True):
    """
    Plots the class distribution of a dataset as a bar chart.

    Args:
        dataset:    any dataset whose __getitem__ returns (image, label)
        label:      title string for the plot
        save_path:  if set, saves the figure here
        show:       call plt.show() if True
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    # Extract all labels — works for torchvision datasets and PatchDataset wrappers
    if hasattr(dataset, "labels"):
        labels = dataset.labels                         # torchvision STL10 / CIFAR
    elif hasattr(dataset, "targets"):
        labels = dataset.targets                        # some torchvision datasets
    else:
        print("Extracting labels by iterating dataset (may be slow)...")
        labels = [dataset[i][1] for i in range(len(dataset))]
        print("done.")

    counts  = Counter(int(l) for l in labels)
    classes = sorted(counts.keys())
    values  = [counts[c] for c in classes]
    print(values)

    # Use class names if the dataset exposes them
    class_names = getattr(dataset, "classes", None)
    tick_labels = class_names if class_names else [str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.8), 4))
    bars = ax.bar(classes, values, color="#378ADD", edgecolor="none", width=0.6)

    # Value annotations above each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            str(val), ha="center", va="bottom", fontsize=9
        )

    ax.set_xticks(classes)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)