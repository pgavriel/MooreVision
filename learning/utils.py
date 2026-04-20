import torch
import torch.nn.functional as F


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
) -> dict:
    """
    Vectorized equivalent of set_state_normed → set_size → move_to → enforce_bounds.
    Returns a dict of validated state tensors, all shape [BN].
    """
    raw_states    = torch.rand(num_states, 3)

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