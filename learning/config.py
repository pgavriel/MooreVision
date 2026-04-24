# =============================================================================
# CONFIG — all hyperparameters and settings in one place
# =============================================================================
CONFIG = {
    # --- Paths ---
    "data_dir":             "./data",
    "output_dir":           "./runs",
    "val_cache_path":       "./val_cache/rxr-n36-10k.pt",
    "master_log":           "./runs/master_log.csv",
    "test_name":            "rng_test",
    "test_note":            "Initial run - 10:3 split",

    # --- Dataset ---
    "dataset":            "stl10",      # only stl10 supported here
    "num_classes":        10,
    "image_size":         96,           # STL-10 native resolution

    # --- Patch sampling ---
    # Fill in your patch extraction logic in `extract_patches_and_bboxes` below.
    "num_patches":        36,           # N patches sampled per image
    "patch_dim":          768,          # output dim of your patch encoder
    "curve_iter":           16,          # Curve Iterations
    "curve_mode":           3,          # [0 = Moore[i=4], 1 = Zigzag, 2 = ZIGZAG2[i=8], 3 = RxR[i=16]]

    # Min/max patch size as a fraction of image_size.
    # e.g. 0.2 → smallest patch covers 20% of image width/height
    "min_patch_scale":    0.01,
    "max_patch_scale":    1.0,

    # --- Model ---
    "d_model":            256,
    "num_heads":          8,
    "num_layers":         12,
    "ffn_dim":            1024,
    "dropout":            0.1,

    # --- Training ---
    "epochs":             100,
    "batch_size":         64,
    "num_workers":        4,

    # --- Optimizer (AdamW) ---
    "lr":                 3e-4,
    "weight_decay":       0.05,
    "betas":              (0.9, 0.999),
    "grad_clip":          1.0,          # max gradient norm, None to disable

    # --- LR Schedule (cosine with linear warmup) ---
    "warmup_epochs":      10,
    "min_lr":             1e-6,

    # --- Checkpointing ---
    "save_every_n_epochs": 20,          # also saves best val acc automatically
    "resume_from":        None,         # path to checkpoint to resume, or None

    # --- Reproducibility ---
    "seed":               42,

    # --- Logging ---
    "print_every_n_steps": 20,          # print batch-level stats this often
}
# =============================================================================