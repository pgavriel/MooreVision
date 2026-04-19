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