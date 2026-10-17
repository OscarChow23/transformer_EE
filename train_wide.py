#!/usr/bin/env python3
# Flexible launch script for Transformer_EE_MV with widened d_model & extras.

import json
import sys
import os
from datetime import datetime
import argparse
from copy import deepcopy
import hashlib

from transformer_ee.noise_config import validate_noise_config

# ---------- Helpers ----------
def kset(d, dotted_key, value):
    """Fill a missing nested config key without replacing configured values."""
    if value is None:
        return
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    if parts[-1] not in cur:
        cur[parts[-1]] = value


def ensure_model_kwargs(cfg):
    """Make sure cfg['model']['kwargs'] exists."""
    if "model" not in cfg:
        cfg["model"] = {}
    if "kwargs" not in cfg["model"]:
        cfg["model"]["kwargs"] = {}
    return cfg["model"]["kwargs"]


def fail_if(cond, msg):
    if cond:
        raise ValueError(msg)


def shorten_with_hash(text, max_len=120):
    """
    Keep strings under service limits while preserving recognizability.
    This is mainly for W&B name/group/id values, which can otherwise exceed
    backend length limits when we pass through long JSON-derived basenames.
    """
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    if len(text) <= max_len:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    keep = max_len - len(digest) - 1
    keep = max(keep, 16)
    return f"{text[:keep]}_{digest}"


def ensure_unique_dir(base_path):
    """
    Return a non-existing run directory under base_path.
    This gives us a local uniqueness guard for repeated stochasticity studies,
    even when the caller intentionally reuses the same base config name.
    """
    base_path = os.path.abspath(base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(f"{base_path}_{timestamp}_{os.getpid()}".encode("utf-8")).hexdigest()[:6]
    candidate = os.path.join(base_path, f"run_{timestamp}_{digest}")
    counter = 0
    while os.path.exists(candidate):
        counter += 1
        candidate = os.path.join(base_path, f"run_{timestamp}_{digest}_{counter:02d}")
    return candidate


def validate_and_autofill(cfg):
    """
    Validate divisibility constraints and fill sensible defaults
    (keeps legacy behavior if you don't opt in).
    """
    mkw = ensure_model_kwargs(cfg)

    nhead = int(mkw.get("nhead", 4))
    # Legacy behavior (no widening) if both d_model and auto_d_model are absent/False.
    explicit_d = mkw.get("d_model", None)
    auto_d = bool(mkw.get("auto_d_model", False))

    if explicit_d is not None:
        explicit_d = int(explicit_d)
        fail_if(explicit_d % nhead != 0,
                f"d_model={explicit_d} must be divisible by nhead={nhead}")
        # If user didn't set dim_feedforward, give a sane default tied to width
        if "dim_feedforward" not in mkw:
            mkw["dim_feedforward"] = max(4 * explicit_d, 128)

    # If auto_d_model is true, the class will compute d_model internally,
    # but we still warn if the user set incompatible min_head_dim & nhead.
    if auto_d:
        # Nothing to validate here beyond types; real selection happens in model.
        # Still: if user supplied d_model together with auto_d_model, prefer explicit.
        if explicit_d is not None:
            # Be explicit about precedence to avoid silent surprises.
            print("[WARN] Both d_model and auto_d_model were set; using explicit d_model.")

    # If neither explicit nor auto, legacy path: d_model == len(vector),
    # which *must* be divisible by nhead. We can’t check that here without opening the
    # base JSON config, but the model will raise with a clear error if it’s not.

    # Suggest a dim_feedforward if user widened but forgot to scale it.
    if explicit_d is not None and "dim_feedforward" in mkw:
        dff = int(mkw["dim_feedforward"])
        if dff < 2 * explicit_d:
            print(f"[WARN] dim_feedforward={dff} may be small for d_model={explicit_d}. "
                  f"Consider >= {4 * explicit_d}.")


def apply_profile(cfg, profile):
    """
    Fast presets so you can A/B ideas without typing a wall of flags.
    Profiles only *set/override* a few keys; everything else remains unchanged.
    """
    mkw = ensure_model_kwargs(cfg)

    if profile == "legacy":
        # exact legacy behavior; nothing to override
        pass

    elif profile == "wide":
        # widen with explicit d_model; keep legacy head and pooling
        mkw["d_model"] = max(128, mkw.get("nhead", 4) * 32)  # safeish starting point
        mkw["auto_d_model"] = False
        mkw["scalar_as_token"] = False
        mkw["use_new_head"] = False
        mkw.setdefault("dim_feedforward", max(4 * mkw["d_model"], 128))

    elif profile == "auto":
        # let the model compute d_model; keep legacy head
        mkw["auto_d_model"] = True
        mkw["min_head_dim"] = 16
        mkw["base_multiple"] = 4
        mkw["scalar_as_token"] = False
        mkw["use_new_head"] = False
        # dim_feedforward will be set in the model layer if not provided

    elif profile == "token":
        # scalar token + auto width; still use legacy head for apples-to-apples
        mkw["auto_d_model"] = True
        mkw["min_head_dim"] = 16
        mkw["base_multiple"] = 4
        mkw["scalar_as_token"] = True
        mkw["use_new_head"] = False

    elif profile == "newhead":
        # scalar token + auto width + modern head
        mkw["auto_d_model"] = True
        mkw["min_head_dim"] = 16
        mkw["base_multiple"] = 4
        mkw["scalar_as_token"] = True
        mkw["use_new_head"] = True

    else:
        fail_if(True, f"Unknown profile '{profile}'. Valid: legacy|wide|auto|token|newhead")


def configure_noise(cfg, args):
    """Merge explicit noise flags over JSON, then validate the effective block.

    Noise arguments default to None, not their physical defaults. Otherwise a
    bare CLI launch would silently replace the JSON distribution, width, or
    correlation mode. This helper runs before opening logs, loading data, or
    contacting W&B, so configuration errors cannot launch a partial run.

    A noise block in JSON already enables augmentation. For a new block require
    --enable-noise; tuning flags alone are rejected rather than silently ignored.
    --disable-noise removes a JSON block and conflicts with tuning flags.
    """
    keys = ("name", "mode", "scalar_mode", "mean", "std", "low", "high", "vector", "scalar")
    overrides = {
        key: getattr(args, "noise_" + key)
        for key in keys if getattr(args, "noise_" + key) is not None
    }
    if args.disable_noise:
        if overrides:
            raise ValueError("--disable-noise cannot be combined with noise tuning options")
        cfg.pop("noise", None)
        return
    if "noise" not in cfg:
        if not args.enable_noise:
            if overrides:
                raise ValueError("Noise options require --enable-noise or a noise block in the base JSON")
            return
        # Keep the historical launcher feature selections. They are GENIE field
        # names, not a safe detector-independent guess. For other datasets pass
        # explicit --noise-vector/--noise-scalar lists (an empty list is allowed).
        cfg["noise"] = {
            "vector": [
                "Final_State_Particles_Energy",
                "Final_State_Particles_Momentum_X",
                "Final_State_Particles_Momentum_Y",
                "Final_State_Particles_Momentum_Z",
                "Final_State_Particles_CosTheta",
                "Final_State_Particles_Theta",
            ],
            "scalar": ["tot_fKE", "tot_hadronic_energy", "p_tot", "P_miss"],
        }
    if not isinstance(cfg["noise"], dict):
        raise ValueError("noise must be a configuration object")
    merged = deepcopy(cfg["noise"])
    merged.update(overrides)
    candidate = dict(cfg, noise=merged)
    cfg["noise"] = validate_noise_config(candidate)


# ---------- CLI ----------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Launch Transformer_EE_MV with flexible d_model / scalar token / head."
    )

    # Core I/O
    p.add_argument("--base-config",
                   default="/home/cborden/git/josh_transformer_EE/transformer_EE/transformer_ee/config/wEID/input_GENIEv3-0-6-Honda-Truth-hA-LFG_Numu_CC_Thresh_p1to1_eventnum_All_NpNpi_MSE_E_Px_Py_Pz_EID.json")
    p.add_argument("--data-path",
                   default="/exp/dune/app/users/rrichi/FinalCodes/Numu_CC_Thresh_p1to1_VectorLeptWithoutNC_eventnum_All_NpNpi.csv")
    p.add_argument("--save-path",
                   default="/exp/dune/data/users/cborden/save_genie-dmodel_TEST/model/test/GENIEv3-0-6-Honda-Truth-hA-LFG_Numu_CC_Thresh_p1to1_eventnum_All_NpNpi_MAE_E_Px_Py_Pz_EID")
    # Optional uniqueness guard for repeated stochasticity studies. When enabled,
    # the final cfg['save_path'] becomes <save-path>/run_YYYYmmdd_HHMMSS_hash.
    p.add_argument("--auto-unique-save-path", action="store_true",
                   help="Append a unique run_* subdirectory beneath --save-path to avoid overwrites.")
    p.add_argument("--dataframe-type", default="polars")
    p.add_argument("--num-workers", type=int, default=10)

    # Correlation and distribution are separate switches. Every optional value
    # stays None until configuration merging so untouched JSON retains priority.
    noise_enable = p.add_mutually_exclusive_group()
    noise_enable.add_argument(
        "--enable-noise",
        action="store_true",
        help="Create a noise block if JSON has none (Gaussian coherent defaults)."
    )
    noise_enable.add_argument(
        "--disable-noise", action="store_true",
        help="Disable augmentation even when the base JSON contains a noise block."
    )
    p.add_argument(
        "--noise-name",
        choices=["gaussian", "uniform"], default=None,
        help="Fractional noise distribution; overrides JSON (new-block default: gaussian)."
    )
    p.add_argument(
        "--noise-mode", choices=["coherent", "particle"], default=None,
        help="One draw per event or per particle, shared over selected features (default: coherent)."
    )
    p.add_argument(
        "--noise-scalar-mode",
        choices=["auto", "none", "coherent", "independent", "particle_mean"], default=None,
        help="auto: share coherent vector draw, leave scalars unchanged in particle mode; "
             "coherent: same draw as coherent vectors (requires coherent mode); "
             "independent: own draw per scalar; particle_mean: shared mean of valid "
             "particle draws (requires particle mode); none: unchanged."
    )
    p.add_argument(
        "--noise-mean",
        type=float,
        default=None,
        help="Gaussian fractional mean; overrides JSON (default: 0.0)."
    )
    p.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="Gaussian fractional standard deviation; overrides JSON (default: 0.2)."
    )
    p.add_argument("--noise-low", type=float, default=None,
                   help="Uniform lower fractional bound; overrides JSON (default: -0.2).")
    p.add_argument("--noise-high", type=float, default=None,
                   help="Uniform upper fractional bound; overrides JSON (default: 0.2).")
    # nargs='*' deliberately distinguishes no flag (None: keep JSON) from a
    # flag with no names ([]: clear selection), useful for vector-only studies.
    p.add_argument(
        "--noise-vector",
        nargs="*",
        default=None,
        help="Selected vector names; overrides JSON. Pass no names to clear selection."
    )
    p.add_argument(
        "--noise-scalar",
        nargs="*",
        default=None,
        help="Selected scalar names; overrides JSON. Pass no names to clear selection."
    )

    # Training / Optim
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--early-stop-window",
        type=int,
        default=None,
        help="Enable early stopping with this trailing epoch window size (must be >= 2).",
    )
    p.add_argument(
        "--early-stop-min-delta-pct",
        type=float,
        default=None,
        help="Minimum percent validation-loss improvement required across the early-stop window.",
    )
    p.add_argument(
        "--keep-last-n-epoch-checkpoints",
        type=int,
        default=None,
        help="Retain rolling last_model_epoch_<i>.zip checkpoints (0 disables).",
    )
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)  # only for SGD
    p.add_argument("--weight-decay", type=float, default=None)

    # Transformer knobs (override as needed)
    p.add_argument("--nhead", type=int, default=2)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--dim-ff", type=int, default=None)

    # Width control
    p.add_argument("--d-model", type=int, default=None,
                   help="Explicit transformer width (must be divisible by nhead).")
    p.add_argument("--auto-d-model", action="store_true",
                   help="If set, pick d_model automatically inside the model.")
    p.add_argument("--min-head-dim", type=int, default=None,
                   help="Used when --auto-d-model is set. Head dim >= this.")
    p.add_argument("--base-multiple", type=int, default=None,
                   help="Used when --auto-d-model is set. d_model >= base_multiple * nvec.")

    # Sequence & head style
    p.add_argument("--scalar-as-token", action="store_true",
                   help="Prepend a scalar-derived token to the sequence.")
    p.add_argument("--use-new-head", action="store_true",
                   help="LayerNorm+GELU MLP head on pooled token. If off, legacy head is used.")
    p.add_argument("--norm-first", action="store_true",
                   help="Use norm_first=True in TransformerEncoderLayer.")

    # fast presets
    p.add_argument("--profile", default=None,
                   help="Quick preset: legacy|wide|auto|token|newhead")

    # WandB
    # Note: keep --wandb-id optional. For these stochasticity sweeps, the raw JSON-derived
    # basename can exceed backend limits, so --wandb-name/--wandb-group are the safer places
    # for human-readable labels while the backend id can be omitted or shortened.
    p.add_argument("--wandb-project", default="GENIE_Atmo")
    p.add_argument("--wandb-entity", default="neutrinoenenergyestimators")
    p.add_argument("--wandb-dir", default="/exp/dune/data/users/cborden/save_genie-dmodel_TEST/wandb")
    # Critical detail: W&B "id" is a persistent backend run identifier, not a display label.
    # Reusing a constant default id across multiple launches will collide with older runs and,
    # when paired with resume="never", immediately fail on rerun. So the safe default here is None.
    p.add_argument("--wandb-id",
                   default=None,
                   help="Optional explicit W&B run id. Leave unset to let W&B auto-generate a unique id.")
    p.add_argument("--wandb-name",
                   default=None,
                   help="Optional short human-readable W&B run name.")
    p.add_argument("--wandb-group",
                   default=None,
                   help="Optional W&B group name for bundling replicas of the same config.")
    p.add_argument("--wandb-tags",
                   nargs="*",
                   default=None,
                   help="Optional W&B tags.")

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.early_stop_window is not None and args.early_stop_window < 2:
        parser.error("--early-stop-window must be >= 2 when early stopping is enabled.")

    # Load base config
    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    try:
        configure_noise(cfg, args)
    except ValueError as exc:
        parser.error(str(exc))

    # Keep --help and configuration validation usable without importing torch,
    # dataframe libraries, or the optional W&B client. Actual training retains
    # the same dependencies; importing them here also avoids opening a log file
    # before discovering a missing runtime dependency.
    from transformer_ee.train import MVtrainer
    from transformer_ee.logger.wandb_train_logger import WandBLogger
    import torch

    # Optional save-path uniquification. This is intentionally opt-in so legacy behavior
    # remains unchanged unless a caller explicitly wants local run folders to never collide.
    kset(cfg, "save_path", args.save_path)
    if args.auto_unique_save_path:
        cfg["save_path"] = ensure_unique_dir(cfg["save_path"])
        print(f"[INFO] Auto-unique save path enabled. Using: {cfg['save_path']}")

    # --- Apply common overrides (I/O, dataloader, training) ---
    kset(cfg, "data_path", args.data_path)
    kset(cfg, "dataframe_type", args.dataframe_type)
    kset(cfg, "num_workers", args.num_workers)

    kset(cfg, "model.epochs", args.epochs)

    # Save stdout as log file to save_path
    orig_stdout = sys.stdout
    if os.path.isdir(cfg["save_path"])==False:
        os.makedirs(cfg["save_path"])
    f = open(cfg["save_path"]+"/train.log", 'a')
    sys.stdout = f

    # Optional early-stopping/checkpoint policy (CLI-driven).
    if args.early_stop_window is not None or args.early_stop_min_delta_pct is not None:
        early_cfg = cfg.get("early_stopping", {})
        early_cfg["enabled"] = True
        if args.early_stop_window is not None:
            early_cfg["window"] = int(args.early_stop_window)
        if args.early_stop_min_delta_pct is not None:
            early_cfg["min_delta_pct"] = float(args.early_stop_min_delta_pct)
        cfg["early_stopping"] = early_cfg

    if args.keep_last_n_epoch_checkpoints is not None:
        ckpt_cfg = cfg.get("checkpointing", {})
        ckpt_cfg["keep_last_n_epoch_checkpoints"] = int(args.keep_last_n_epoch_checkpoints)
        cfg["checkpointing"] = ckpt_cfg

    # Log the effective, validated noise block, including explicit defaults, so
    # checkpoints/W&B record the correlation mode used for this particular run.
    if "noise" in cfg:
        print("[INFO] Noise configuration enabled:", cfg["noise"])


    # Optimizer
    kset(cfg, "optimizer.name", args.optimizer)
    kset(cfg, "optimizer.kwargs.lr", args.lr)
    if cfg["optimizer"]["name"] == "sgd":
        kset(cfg, "optimizer.kwargs.momentum", args.momentum)
    if args.weight_decay is not None:
        kset(cfg, "optimizer.kwargs.weight_decay", args.weight_decay)

    # Transformer basics
    mkw = ensure_model_kwargs(cfg)
    kset(mkw, "nhead", args.nhead)
    kset(mkw, "num_layers", args.num_layers)
    kset(mkw, "dropout", args.dropout)
    kset(mkw, "dim_feedforward", args.dim_ff)

    # Profiles (apply late, they override the above if needed)
    if args.profile:
        apply_profile(cfg, args.profile)

    # Width / style toggles (explicit CLI wins over profile)
    if args.d_model is not None:
        mkw["d_model"] = int(args.d_model)
        mkw["auto_d_model"] = False
    if args.auto_d_model:
        mkw["auto_d_model"] = True
    if args.min_head_dim is not None:
        mkw["min_head_dim"] = int(args.min_head_dim)
    if args.base_multiple is not None:
        mkw["base_multiple"] = int(args.base_multiple)
    if args.scalar_as_token:
        mkw["scalar_as_token"] = True
    if args.use_new_head:
        mkw["use_new_head"] = True
    if args.norm_first:
        mkw["norm_first"] = True

    # Validate & add implied defaults
    validate_and_autofill(cfg)

    # --- WandB logger ---
    # Separate the backend id from the user-facing name/group so that long configuration
    # names can still be tracked cleanly without violating W&B length limits.
    wandb_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "config": cfg,
        "dir": args.wandb_dir,
    }

    wandb_id = shorten_with_hash(args.wandb_id, max_len=120)
    wandb_name = shorten_with_hash(args.wandb_name, max_len=120)
    wandb_group = shorten_with_hash(args.wandb_group, max_len=120)

    if wandb_id is not None:
        # Only pass an explicit backend run id when the caller really asked for one.
        # For stochasticity studies, the usual intent is "new run every time", so omitting
        # id entirely is safer than silently reusing a previous persistent run identity.
        wandb_kwargs["id"] = wandb_id
        wandb_kwargs["resume"] = "never"
    else:
        # Let W&B allocate a fresh backend id for this launch.
        print("[INFO] No explicit W&B run id provided; allowing W&B to create a fresh unique id.")

    if wandb_name is not None:
        wandb_kwargs["name"] = wandb_name
    if wandb_group is not None:
        wandb_kwargs["group"] = wandb_group
    if args.wandb_tags:
        wandb_kwargs["tags"] = args.wandb_tags

    print("[INFO] W&B launch parameters:", {
        k: v for k, v in wandb_kwargs.items() if k != "config"
    })

    my_logger = WandBLogger(**wandb_kwargs)

    starttime = datetime.now()
    print("Start Time: " + str(starttime))

    my_trainer = MVtrainer(cfg, logger=my_logger)
    my_trainer.train_LCL()

    torch.cuda.empty_cache() # free cached blocks from training
    torch.cuda.ipc_collect() # collect any stray IPC handles

    my_trainer.eval()

    endtime = datetime.now()
    print("End Time: " + str(endtime))
    print("Time Elapsed: " + str(endtime - starttime))

    sys.stdout = orig_stdout
    f.close()


if __name__ == "__main__":
    main()
