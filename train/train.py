from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm
from utils import sst, sse
import matplotlib.pyplot as plt
import os
import glob
from model.model import Ouroboros
from model.kernels import fullPolyModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

from typing import Tuple

# filters removed


def save_model(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    location: str,
    n_layers: int = 2,
    d_state: int = 1,
    d_conv: int = 4,
    expand_factor: int = 4,
    max_saved: int = 5,
):
    """
    save a model. requires information about the structure of
    the model as well, so that it can be reproduced upon loading
    also, automatically removes older saves. It would probably be best to just keep
    the 'best' epoch, but whatevah

    inputs
    -----
        - model: Ouroboros model to be saved
        - opt: optimizer for said model
        - location: filename in which to save the model
        - n_layers: number of mamba layers in encoder
        - d_state: internal state dim of ssm
        - d_conv: width of mamba convolutional kernel
        - expand_factor: factor by which we expand input prior to feeding into ssm
        - max_saved: max number of checkpoint files. we only keep the 5 most recent, assuming
            they're saved with epoch number in the tag
    """

    # Match only `checkpoint_<int>.tar` so side-stream saves (e.g. `inflight_latest.tar`
    # from --save-minutes) don't get globbed in and crash the int(epoch) parse below.
    current_saves = glob.glob(os.path.join("/".join(location.split("/")[:-1]), "checkpoint_*.tar"))
    if len(current_saves) >= max_saved:
        save_epochs = [
            int(s.split("/")[-1].split(".tar")[0].split("_")[-1]) for s in current_saves
        ]
        save_order = np.argsort(save_epochs)
        ordered_saves = [current_saves[o] for o in save_order]
        for ii in range(len(current_saves) - max_saved + 1):
            os.remove(ordered_saves[ii])
    sd = {
        "ouroboros": model.state_dict(),
        "opt": opt.state_dict(),
        "tau": model.tau,
        "smooth_len": model.smooth_len,
        "n_layers": n_layers,
        "d_state": d_state,
        "d_conv": d_conv,
        "expand_factor": expand_factor,
        # parameterization tag for forward-compat (always "poly" on this branch)
        "parameterization": getattr(model, "parameterization", "poly"),
        "drive_lowpass_ms": getattr(model, "drive_lowpass_ms", 0.0),
        "alpha_lowpass_ms": getattr(model, "alpha_lowpass_ms", 0.0),
        "keep_const": getattr(model, "keep_const", False),
        "use_tract": getattr(model, "use_tract", False),
        "tract_n_sec": getattr(model, "tract_n_sec", 3),
        "use_envelope": getattr(model, "use_envelope", False),
        "env_lowpass_ms": getattr(model, "env_lowpass_ms", 20.0),
        "enable_noise_forcing": getattr(model, "enable_noise_forcing", False),
        "noise_tau_ms": getattr(model, "noise_tau_ms", 5.0),
        "use_noise_branch": getattr(model, "use_noise_branch", False),
        "noise_tract_n_sec": getattr(model, "noise_tract_n_sec", 3),
        "sigma_lowpass_ms": getattr(model, "sigma_lowpass_ms", 0.0),
    }
    try:
        sd["n_kernel"] = model.kernel.nTerms
    except (KeyError, AttributeError):
        pass

    torch.save(sd, location)


def load_model(
    location: str,
    device: str = "cuda",
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    """
    load a model. requires that save files contained information about the structure of
    the model as well, so that it can be reproduced upon loading


    inputs
    -----
        - location: save file directory you wish to load from. Assumes files include checkpoint number
            and loads the most recent checkpoint

    returns
    -----
        - model: an Ouroboros with weights and structure specified by savefile
        - opt: optimizer for that Ouroboros
        - scheduler: learning rate scheduler for that optimizer
        - epoch: training epoch corresponding to this checkpoint
    """

    # Match only `checkpoint_<int>.tar` so side-stream saves like `inflight_latest.tar`
    # (intra-epoch save from --save-minutes) don't get picked up and crash the int() parse.
    model_files = glob.glob(os.path.join(location, "checkpoint_*.tar"))
    epochs = [int(m.split("/checkpoint_")[-1].split(".tar")[0]) for m in model_files]
    most_recent = np.argsort(epochs)[-1]
    location = model_files[most_recent]
    print(f"loading from {location}")

    sd = torch.load(location, weights_only=False, map_location=device)
    try:
        n_layers = sd["n_layers"]
        d_state = sd["d_state"]
        d_conv = sd["d_conv"]
        expand_factor = sd["expand_factor"]
    except KeyError:
        n_layers = 2
        d_state = 1
        d_conv = 4
        expand_factor = 4
    try:
        # since this is a trained model and we only use lambda during training, i set it to 1 here...
        # but probably should have saved it. oh well! we set to 1 for compatibility with all my saves.
        kernel = fullPolyModule(
            nTerms=sd["n_kernel"],
            device=device,
            x_dim=1,
            z_dim=2,
            activation=lambda x: x,
            lam=1,
        )

        model = Ouroboros(
            d_data=1,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            tau=sd["tau"],
            smooth_len=sd["smooth_len"],
            kernel=kernel,
            device=device,
            drive_lowpass_ms=sd.get("drive_lowpass_ms", 0.0),
            alpha_lowpass_ms=sd.get("alpha_lowpass_ms", 0.0),
            keep_const=sd.get("keep_const", False),
            use_tract=sd.get("use_tract", False),
            tract_n_sec=sd.get("tract_n_sec", 3),
            use_envelope=sd.get("use_envelope", False),
            env_lowpass_ms=sd.get("env_lowpass_ms", 20.0),
            enable_noise_forcing=sd.get("enable_noise_forcing", False),
            noise_tau_ms=sd.get("noise_tau_ms", 5.0),
            use_noise_branch=sd.get("use_noise_branch", False),
            noise_tract_n_sec=sd.get("noise_tract_n_sec", 3),
            sigma_lowpass_ms=sd.get("sigma_lowpass_ms", 0.0),
        )
    except:
        print("no kernel in savefile!")
        raise

    print(f"model tau: {model.tau}")
    opt = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(opt, factor=0.75, patience=5, min_lr=1e-10)
    # strict=False only to tolerate the K_anchor_target buffer being absent in checkpoints
    # saved before the tract-gain anchor existed; the buffer keeps its constructor/data-init
    # value in that case. Any OTHER missing/unexpected key is a real mismatch -> raise.
    _incompat = model.load_state_dict(sd["ouroboros"], strict=False)
    _missing = [k for k in _incompat.missing_keys if not k.endswith("K_anchor_target")]
    if _missing or _incompat.unexpected_keys:
        raise RuntimeError(f"state_dict mismatch: missing={_missing} "
                           f"unexpected={list(_incompat.unexpected_keys)}")
    opt.load_state_dict(sd["opt"])

    # Free any allocator fragments left over from the ckpt-loading sequence.
    # Without this, the resume's startup pool can sit ~hundreds of MiB above the
    # actual working set, which combined with the first-step graph capture
    # transient can OOM at batch sizes that a fresh-start of the same config
    # would fit comfortably in.
    del sd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, opt, scheduler, epochs[most_recent]


def train(
    model,
    optimizer,
    loss_fn,
    loaders,
    scheduler=None,
    nEpochs=100,
    val_freq=25,
    runDir=".",
    dt=1 / 44100,
    vis_freq=100,
    smoothing=False,
    reg_weights=False,
    start_epoch=0,
    model_info={},
    save_freq=0,
    max_saved: int = 5,
    loss_mode: str = "mse_accel",
    # spectral-rollout knobs (only consulted when loss_mode == "spectral_rollout")
    H_min: int = 512,
    H_max: int = 2048,
    H_schedule: str = "geom",
    lam_spec: float = 1.0,
    lam_tf: float = 1.0,
    lam_env: float = 0.0,
    lam_env_log: float = 0.0,       # weight on the log-ratio envelope loss (env_loss_log)
    env_log_eps: float = 1e-4,      # noise floor inside the log() in env_loss_log
    env_ms: float = 2.0,
    lam_reg: float = 0.0,           # scale on the degree-graded L2 penalty on kernel weights
    lam_env_anchor: float = 0.0,    # scale on mean((e - 1)^2) envelope gauge anchor (pulls e toward 1)
    lam_tract_k_anchor: float = 0.0,  # scale on (K/K0 - 1)^2 tract-gain gauge anchor (pins K near data-init)
    lam_env_max_anchor: float = 0.0,  # scale on mean((max_t e - 1)^2) envelope PEAK anchor (pins scale, not shape)
    spec_warmup_epochs: int = 5,    # linearly ramp lam_spec 0 -> lam_spec over these epochs
    env_warmup_epochs: int = 0,     # linearly ramp lam_env AND lam_env_log over these epochs
    # Step-based overrides (None = derived from _epochs * batches_per_epoch at startup).
    # Set these to decouple the curriculum from dataset size: same number of gradient
    # updates regardless of how many batches an epoch contains.
    spec_warmup_steps: int = None,
    env_warmup_steps: int = None,
    H_total_steps: int = None,
    spec_configs=None,
    ic_noise_rms: float = 1e-3,
    grad_clip: float = 5.0,
    rollout_backend: str = "eager",  # RK4 backend: 'eager' | 'cudagraph' | 'compile'
    # Linear LR ramp: lr at epoch 0 == optimizer's current lr (the --lr value);
    # at epoch >= lr_ramp_epochs lr == lr_end; linearly interpolated in between.
    # lr_end=None disables (constant LR), preserving the legacy behaviour.
    lr_end: float = None,
    lr_ramp_epochs: int = 5,
    # Intra-epoch save cadence (minutes wall-clock). 0 disables; otherwise an
    # in-flight checkpoint is overwritten at `inflight_latest.tar` once this
    # many minutes have elapsed since the last save (epoch-boundary saves count).
    # The `inflight_*` prefix is excluded from the `checkpoint_*.tar` glob the
    # loader uses, so resume still picks the latest per-epoch save and the
    # sidecar's epoch-keyed bookkeeping is unaffected.
    save_minutes: float = 0.0,
    # Freeze the drive encoders (omega/gamma/kernel Mambas + linear heads) for the
    # first `freeze_drives_epochs` epochs so envelope+tract can settle the audio
    # amplitude/spectral shape before the polynomial dynamics start tracking. 0
    # disables (legacy: all params trainable from epoch 0).
    freeze_drives_epochs: int = 0,
    # Freeze the vocal-tract filter (pole/zero sections + K_raw + comb) for the
    # first `freeze_tract_epochs` epochs so drive+envelope gradients don't pull on
    # the filter while the dynamics are still random. Tract stays at its init
    # (zeros=poles=identity shape, K_raw at the data-matched gain or whatever the
    # entry script sets) during the freeze. 0 disables.
    freeze_tract_epochs: int = 0,
    # Freeze the envelope head (env_mamba + env_net) for the first
    # `freeze_envelope_epochs` epochs. With env_net zero-init, e(t) stays at 1.0
    # (identity) during the freeze -- amplitude lives entirely in K_raw + drives
    # while the envelope can't roam. 0 disables.
    freeze_envelope_epochs: int = 0,
    # Flow-gated colored-noise forcing schedule (only consulted when the model was built
    # with enable_noise_forcing). The forcing gain ramps 0 -> 1 linearly: it is held at 0
    # until global step `noise_start_step`, then ramps over `noise_warmup_steps` steps.
    # The sigma head (sigma_mamba + sigma_net) is also frozen for the first
    # `freeze_noise_epochs` epochs. Recommended: resume a trained deterministic checkpoint
    # and set noise_start_step so the deterministic model is settled before noise turns on.
    noise_start_step: int = 0,
    noise_warmup_steps: int = 0,
    freeze_noise_epochs: int = 0,
) -> Tuple[
    list[float], list[Tuple[int, float, float]], nn.Module, torch.optim.Optimizer
]:
    """
    main train loop for an Ouroboros model. takes a model, an optimizer, loss function, and dataloaders;
    trains a model, and returns that model and train statistics

    inputs
    -----
        - model: an Ouroboros
        - optimizer: an optimizer for that Ouroboros
        - loss_fn: a training objective. In all cases, we used MSE
        - loaders: Dataloaders, one for the train set and one for the test set
        - scheduler: a learning rate scheduler for the optimizer. Optional, but we use this
        - nEpochs: number of passes through the entire training dataset
        - val_freq: frequency of looking at the validation set to test performance
        - runDir: directory to save tensorboard logs in
        - dt: sampling timestep of the data
        - vis_freq: frequency with which to visualize model reconstructions
        - smoothing: whether or not to smooth latents during training. we do not,but you can
        - reg_weights: whether or not to regularize weights, using procedure laid out in the paper. we always do
        - start_epoch: starting epoch for training. if training a model, this should be 0; if loading a trained model
            this might be higher
        - model_info: dictionary of model structure specification. used for saving models
        - save_freq: how often (in epochs) to save your model

    returns
    ----
        - train_losses: train loss per epoch.
        - val losses: list of tuples, containing time point (in gradient updates) of val, val loss, and val regularization cost
        - trained Ouroboros
        - optimizer for that trained Ouroboros
    """

    writer = SummaryWriter(log_dir=runDir)

    # Hard dependency: the OU noise term can only be supervised in distribution by the
    # phase-discarding MRSTFT magnitude loss. A pointwise/time-domain loss would penalize
    # every noise realization for not matching the specific training draw, which is incoherent.
    if (getattr(model, "enable_noise_forcing", False) or getattr(model, "use_noise_branch", False)) \
            and loss_mode != "spectral_rollout":
        raise ValueError(
            "enable_noise_forcing / use_noise_branch require loss_mode='spectral_rollout' "
            f"(MRSTFT magnitude loss); got loss_mode={loss_mode!r}. The noise realization is "
            "random-phase and cannot be supervised by a pointwise objective."
        )

    train_losses, val_losses = [], []

    if loss_mode == "spectral_rollout":
        from train.spectral_rollout import (
            spectral_rollout_step,
            horizon_for_step,
            DEFAULT_CONFIGS,
            ONSET,
        )
        if spec_configs is None:
            spec_configs = DEFAULT_CONFIGS
        # Precompute Var(d2x) over the whole training set once, in the same
        # rescaled-time units used in the inner loop. Matches rollout_refine.py:110
        # and prevents the per-batch variance from blowing up the TF anchor on
        # batches dominated by silence (ONSET segments).
        tf_var_running = 0.0
        n_seen = 0
        with torch.no_grad():
            for batch in loaders["train"]:
                d2 = batch[2]   # (B, L, 1)
                d2 = d2.to("cuda", non_blocking=True).to(torch.float32) / (dt ** 2) * model.tau ** 2
                tf_var_running += float(d2.var().item()) * d2.shape[0]
                n_seen += d2.shape[0]
        tf_var = max(tf_var_running / max(1, n_seen), 1e-6)
        # Decouple the curricula from dataset size: each ramp is measured in global
        # gradient steps (batches). If the user didn't override, derive from epoch
        # values × current epoch length so existing CLI invocations are unchanged.
        batches_per_epoch = len(loaders["train"])
        if spec_warmup_steps is None:
            spec_warmup_steps_eff = spec_warmup_epochs * batches_per_epoch
        else:
            spec_warmup_steps_eff = int(spec_warmup_steps)
        if env_warmup_steps is None:
            env_warmup_steps_eff = env_warmup_epochs * batches_per_epoch
        else:
            env_warmup_steps_eff = int(env_warmup_steps)
        if H_total_steps is None:
            H_total_steps_eff = nEpochs * batches_per_epoch
        else:
            H_total_steps_eff = int(H_total_steps)
        _k0 = (float(model.tract.K_anchor_target)
               if getattr(model, "tract", None) is not None else float("nan"))
        print(
            f"spectral_rollout mode: lam_spec={lam_spec} lam_tf={lam_tf} lam_env={lam_env} "
            f"lam_env_log={lam_env_log} env_log_eps={env_log_eps} lam_reg={lam_reg} "
            f"lam_env_anchor={lam_env_anchor} lam_tract_k_anchor={lam_tract_k_anchor} "
            f"lam_env_max_anchor={lam_env_max_anchor} "
            f"K_anchor_target={_k0:.4g} kernel.lam={float(model.kernel.lam):.4g} "
            f"H={H_min}->{H_max} ({H_schedule}) ic_noise_rms={ic_noise_rms} tf_var={tf_var:.4g} "
            f"rollout_backend={rollout_backend} "
            f"spec_warmup_steps={spec_warmup_steps_eff} env_warmup_steps={env_warmup_steps_eff} "
            f"H_total_steps={H_total_steps_eff} (batches_per_epoch={batches_per_epoch})",
            flush=True,
        )
        if rollout_backend not in ("eager", "graphstep") and H_schedule not in ("pow2", "const"):
            print(
                f"  WARNING: rollout_backend={rollout_backend!r} captures one CUDA graph "
                f"per distinct H; schedule {H_schedule!r} yields many. Use H_schedule='pow2' "
                f"to bucket horizons in factor-of-2 steps.",
                flush=True,
            )

    # Capture starting LR (set by model_cv.py from --lr) so the ramp interpolates from
    # that value rather than from whatever Adam was constructed with.
    lr_start_per_group = [g['lr'] for g in optimizer.param_groups]
    import time as _time
    _last_save_t = _time.time()

    # Drive params (omega/gamma/kernel Mambas + their linear heads). Captured once
    # so the per-epoch toggle is a quick walk over the list, not a name match each
    # time. kernel.weights is the polynomial-coefficient linear head; tract.* and
    # env_mamba/env_net are NOT in this list (they keep training during the freeze).
    drive_modules = []
    for attr in ("omega_mamba", "gamma_mamba", "kernel_mamba", "omega_net", "gamma_net"):
        m = getattr(model, attr, None)
        if m is not None:
            drive_modules.append(m)
    if hasattr(model, "kernel") and hasattr(model.kernel, "weights"):
        drive_modules.append(model.kernel.weights)
    drive_params = [p for m in drive_modules for p in m.parameters()]
    drives_frozen_now = False
    if freeze_drives_epochs > 0:
        print(f"freeze_drives_epochs={freeze_drives_epochs}: drives "
              f"(omega/gamma/kernel Mambas + heads + kernel.weights, {len(drive_params)} tensors) "
              f"frozen for the first {freeze_drives_epochs} epochs; envelope+tract still train.",
              flush=True)
    # Freeze the SHAPE filter (poles + zeros + trachea comb). Comb adds spectral
    # notches at multiples of 1/tau, which is the same kind of filter-shape
    # mechanism the pole/zero sections are. K_raw stays trainable -- it's just a
    # scalar gain, not a shape, so amplitude has a descent direction.
    tract_params = []
    if hasattr(model, "tract"):
        for attr in ("f0_raw", "zeta_p_raw", "fz_raw", "zeta_z_raw", "r_raw", "tau_raw"):
            if hasattr(model.tract, attr):
                tract_params.append(getattr(model.tract, attr))
    tract_frozen_now = False
    if freeze_tract_epochs > 0:
        print(f"freeze_tract_epochs={freeze_tract_epochs}: tract shape "
              f"(poles + zeros + comb -- {len(tract_params)} tensors) "
              f"frozen for the first {freeze_tract_epochs} epochs; "
              f"K_raw + drives + envelope still train.",
              flush=True)
    envelope_params = []
    for attr in ("env_mamba", "env_net"):
        m = getattr(model, attr, None)
        if m is not None:
            envelope_params.extend(list(m.parameters()))
    envelope_frozen_now = False
    if freeze_envelope_epochs > 0:
        print(f"freeze_envelope_epochs={freeze_envelope_epochs}: envelope "
              f"(env_mamba + env_net, {len(envelope_params)} tensors) frozen for the "
              f"first {freeze_envelope_epochs} epochs; e(t)=1.0 (identity) during freeze.",
              flush=True)
    # Noise gate head (sigma_mamba + sigma_net). Frozen for the first freeze_noise_epochs
    # epochs; also the noise_gain ramp keeps the forcing off until noise_start_step.
    noise_params = []
    for attr in ("sigma_mamba", "sigma_net", "noise_tract"):
        m = getattr(model, attr, None)
        if m is not None:
            noise_params.extend(list(m.parameters()))
    noise_frozen_now = False
    if getattr(model, "enable_noise_forcing", False):
        print(f"enable_noise_forcing: sigma head ({len(noise_params)} tensors), "
              f"noise_tau_ms={getattr(model, 'noise_tau_ms', None)}, "
              f"noise_start_step={noise_start_step}, noise_warmup_steps={noise_warmup_steps}, "
              f"freeze_noise_epochs={freeze_noise_epochs}.", flush=True)

    for epoch in tqdm(range(start_epoch, nEpochs), desc="training model"):
        model.train()
        # Drive freeze schedule: zero requires_grad on the drive params for the first
        # `freeze_drives_epochs` epochs, then unfreeze. Setting requires_grad=False
        # leaves .grad as None so Adam.step() skips those params (no momentum drift).
        # Idempotent — toggling on already-False params is cheap.
        want_frozen = epoch < freeze_drives_epochs
        if want_frozen != drives_frozen_now:
            for p in drive_params:
                p.requires_grad_(not want_frozen)
            drives_frozen_now = want_frozen
            print(f"  epoch {epoch}: drives {'FROZEN' if want_frozen else 'UNFROZEN'}",
                  flush=True)
        # Same gating for the vocal-tract filter -- mirrored so drive and tract
        # freeze windows can be set independently.
        want_tract_frozen = epoch < freeze_tract_epochs
        if want_tract_frozen != tract_frozen_now:
            for p in tract_params:
                p.requires_grad_(not want_tract_frozen)
            tract_frozen_now = want_tract_frozen
            print(f"  epoch {epoch}: tract {'FROZEN' if want_tract_frozen else 'UNFROZEN'}",
                  flush=True)
        # Envelope freeze schedule (env_mamba + env_net): with env_net zero-init
        # the envelope head outputs e(t)=1 identically, so freezing keeps it there.
        want_env_frozen = epoch < freeze_envelope_epochs
        if want_env_frozen != envelope_frozen_now:
            for p in envelope_params:
                p.requires_grad_(not want_env_frozen)
            envelope_frozen_now = want_env_frozen
            print(f"  epoch {epoch}: envelope {'FROZEN' if want_env_frozen else 'UNFROZEN'}",
                  flush=True)
        # Noise gate head freeze schedule (sigma_mamba + sigma_net).
        want_noise_frozen = epoch < freeze_noise_epochs
        if noise_params and want_noise_frozen != noise_frozen_now:
            for p in noise_params:
                p.requires_grad_(not want_noise_frozen)
            noise_frozen_now = want_noise_frozen
            print(f"  epoch {epoch}: noise gate {'FROZEN' if want_noise_frozen else 'UNFROZEN'}",
                  flush=True)
        # Per-epoch LR ramp: linear from lr_start (epoch 0) to lr_end (epoch lr_ramp_epochs),
        # then hold. No-op if lr_end is None (constant LR).
        if lr_end is not None and lr_ramp_epochs > 0:
            frac = min(1.0, epoch / float(lr_ramp_epochs))
            for g, lr0 in zip(optimizer.param_groups, lr_start_per_group):
                g['lr'] = lr0 + (lr_end - lr0) * frac
            writer.add_scalar("Train/lr", float(optimizer.param_groups[0]['lr']),
                              epoch * len(loaders["train"]))

        for idx, batch in enumerate(
            loaders["train"], start=epoch * len(loaders["train"])
        ):
            optimizer.zero_grad()
            if len(batch) == 4:
                x, dxdt, dx2dt2, cats = batch  # categories from edge-biased sampler
            else:
                x, dxdt, dx2dt2 = batch
                cats = None
            bsz, _, n = x.shape

            x = x.to("cuda", non_blocking=True).to(torch.float32)
            dxdt = dxdt.to("cuda", non_blocking=True).to(torch.float32)
            dx2 = (
                dx2dt2.to("cuda", non_blocking=True).to(torch.float32) / (dt**2) * model.tau**2
            )  # rescale dx2, rather than model output

            if loss_mode == "spectral_rollout":
                # Skip model.forward entirely; spectral_rollout_step calls get_funcs
                # internally with a cloned dxdt (forward and get_funcs mutate dxdt in place).
                ic_mask = None
                if cats is not None:
                    ic_mask = (cats == ONSET).to("cuda")
                # All curricula indexed by GLOBAL STEP (= `idx` thanks to
                # enumerate(..., start=epoch * len(loader))) -- decouples them from
                # dataset size. H is still updated per-batch but pow2 only yields a few
                # distinct values so the graphed/compiled backends stay cache-friendly.
                H = horizon_for_step(idx, H_total_steps_eff, H_min, H_max, H_schedule)
                # Linearly ramp the spectral term so the random-init Mamba can first move
                # into the TF basin (where drives become meaningful) before the spectral
                # loss -- which is enormous when the rollout is saturated against quiet
                # targets -- starts pulling on params.
                if spec_warmup_steps_eff > 0 and idx < spec_warmup_steps_eff:
                    lam_spec_t = lam_spec * (idx / float(spec_warmup_steps_eff))
                else:
                    lam_spec_t = lam_spec
                if env_warmup_steps_eff > 0 and idx < env_warmup_steps_eff:
                    ramp = idx / float(env_warmup_steps_eff)
                    lam_env_t = lam_env * ramp
                    lam_env_log_t = lam_env_log * ramp
                else:
                    lam_env_t = lam_env
                    lam_env_log_t = lam_env_log
                # Noise gain ramp: held at 0 until noise_start_step (so the deterministic model
                # settles first), then linearly 0 -> 1 over noise_warmup_steps. Gates BOTH the
                # in-ODE OU forcing and the additive filtered-noise branch. 0 with no noise head.
                if not (getattr(model, "enable_noise_forcing", False)
                        or getattr(model, "use_noise_branch", False)):
                    noise_gain_t = 0.0
                elif idx < noise_start_step:
                    noise_gain_t = 0.0
                elif noise_warmup_steps > 0:
                    noise_gain_t = min(1.0, (idx - noise_start_step) / float(noise_warmup_steps))
                else:
                    noise_gain_t = 1.0
                out = spectral_rollout_step(
                    model, x, dxdt, dx2, dt,
                    H=H, configs=spec_configs,
                    lam_spec=lam_spec_t, lam_tf=lam_tf,
                    lam_env=lam_env_t,
                    lam_env_log=lam_env_log_t, env_log_eps=env_log_eps,
                    env_ms=env_ms,
                    lam_reg=lam_reg,
                    lam_env_anchor=lam_env_anchor,
                    lam_tract_k_anchor=lam_tract_k_anchor,
                    lam_env_max_anchor=lam_env_max_anchor,
                    tf_var=tf_var,
                    ic_mask=ic_mask, ic_noise_rms=ic_noise_rms,
                    rollout_backend=rollout_backend,
                    noise_gain=noise_gain_t,
                )
                total_loss = out["total"]
                if not torch.isfinite(total_loss):
                    writer.add_scalar("Loss/nan_skip", 1.0, idx)
                    # Explicitly release the forward graph: 'continue' alone leaves the
                    # autograd graph live until the next iteration's locals are rebound,
                    # which on CUDA can hold ~hundreds of MiB of saved-for-backward
                    # tensors (env_mamba pscan saves, rollout step ctxs). Without this,
                    # a nan_skip is immediately followed by an OOM on the next forward.
                    del out, total_loss
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                total_loss.backward()
                # Zero out NaN/Inf entries in every param.grad so a single bad backward
                # path (spec 1/(A+eps), env_mamba pscan, TF anchor) contributes zero rather
                # than poisoning every param via clip_grad_norm_'s divide-by-NaN. Healthy
                # params still drive the step. Unconditional -> no per-param syncs.
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                # Intra-epoch save: write/overwrite `inflight_latest.tar` once
                # `save_minutes` of wall-clock have elapsed since the last save.
                # The .item()-free clock check is per-batch, cheap; the actual
                # torch.save runs only when the gate trips.
                if save_minutes > 0 and (_time.time() - _last_save_t) > save_minutes * 60:
                    save_model(
                        model, optimizer,
                        location=os.path.join(runDir, "inflight_latest.tar"),
                        n_layers=model_info["n layers"],
                        d_state=model_info["d state"],
                        d_conv=model_info["d conv"],
                        expand_factor=model_info["expand factor"],
                        max_saved=max_saved,
                    )
                    _last_save_t = _time.time()
                # Stack the raw component tensors for a single host sync, then derive the
                # weighted views in Python so the TB plots show each term's actual contribution
                # to total (= raw value times its lam_*). lam_spec_t / lam_tf / lam_reg are
                # plain floats already on host.
                spec_v, sc_v, logm_v, tf_v, env_v, env_log_v, reg_v, env_anchor_v, k_anchor_v, env_max_v, total_v = torch.stack(
                    [out["spec"], out["sc"], out["logm"], out["tf"],
                     out["env"], out["env_log"], out["reg"], out["env_anchor"],
                     out["k_anchor"], out["env_max"], total_loss]
                ).tolist()
                train_losses.append(spec_v)
                # Raw values
                writer.add_scalar("Loss/spec", spec_v, idx)
                writer.add_scalar("Loss/sc", sc_v, idx)
                writer.add_scalar("Loss/logm", logm_v, idx)
                writer.add_scalar("Loss/tf", tf_v, idx)
                if lam_env > 0:
                    writer.add_scalar("Loss/env", env_v, idx)
                if lam_env_log > 0:
                    writer.add_scalar("Loss/env_log", env_log_v, idx)
                if lam_reg > 0:
                    writer.add_scalar("Loss/reg", reg_v, idx)
                if lam_env_anchor > 0:
                    writer.add_scalar("Loss/env_anchor", env_anchor_v, idx)
                if lam_tract_k_anchor > 0:
                    writer.add_scalar("Loss/k_anchor", k_anchor_v, idx)
                if lam_env_max_anchor > 0:
                    writer.add_scalar("Loss/env_max", env_max_v, idx)
                writer.add_scalar("Loss/total", total_v, idx)
                if getattr(model, "enable_noise_forcing", False) or getattr(model, "use_noise_branch", False):
                    writer.add_scalar("Train/noise_gain", float(noise_gain_t), idx)
                # Weighted (contribution to total) -- directly comparable across components
                writer.add_scalar("LossW/spec",  float(lam_spec_t) * spec_v,  idx)
                writer.add_scalar("LossW/sc",    float(lam_spec_t) * sc_v,    idx)
                writer.add_scalar("LossW/logm",  float(lam_spec_t) * logm_v,  idx)
                writer.add_scalar("LossW/tf",    float(lam_tf)     * tf_v,    idx)
                if lam_env > 0:
                    writer.add_scalar("LossW/env",     float(lam_env_t)     * env_v,     idx)
                if lam_env_log > 0:
                    writer.add_scalar("LossW/env_log", float(lam_env_log_t) * env_log_v, idx)
                if lam_reg > 0:
                    writer.add_scalar("LossW/reg",     float(lam_reg)       * reg_v,     idx)
                if lam_env_anchor > 0:
                    writer.add_scalar("LossW/env_anchor", float(lam_env_anchor) * env_anchor_v, idx)
                if lam_tract_k_anchor > 0:
                    writer.add_scalar("LossW/k_anchor", float(lam_tract_k_anchor) * k_anchor_v, idx)
                if lam_env_max_anchor > 0:
                    writer.add_scalar("LossW/env_max", float(lam_env_max_anchor) * env_max_v, idx)
                writer.add_scalar("Train/H", float(H), idx)
                writer.add_scalar("Train/lam_spec_t", float(lam_spec_t), idx)
                writer.add_scalar("Train/lam_env_t", float(lam_env_t), idx)
                if lam_env_log > 0:
                    writer.add_scalar("Train/lam_env_log_t", float(lam_env_log_t), idx)
                # Memory diagnostics: log allocated (live tensors) and reserved (PyTorch's
                # caching allocator pool, including fragments). If reserved grows while
                # allocated stays flat across batches, that's fragmentation. Logged every
                # 50 batches (cheap query, but per-batch sync isn't free).
                if idx % 50 == 0:
                    writer.add_scalar("Mem/allocated_GiB",
                                      torch.cuda.memory_allocated() / 1024 ** 3, idx)
                    writer.add_scalar("Mem/reserved_GiB",
                                      torch.cuda.memory_reserved() / 1024 ** 3, idx)
                    writer.add_scalar("Mem/free_GiB",
                                      torch.cuda.mem_get_info()[0] / 1024 ** 3, idx)
                # Periodic empty_cache to reclaim fragmented memory between batches. Cost
                # is a brief sync + losing the fast path for tensor allocation for one
                # batch; benefit is that fragments stop accumulating across many batches.
                # Frequency tuned for "every 100 batches" -- about every 14 min at our pace,
                # negligible overhead.
                if idx > 0 and idx % 100 == 0:
                    torch.cuda.empty_cache()
                continue

            dx2hat, weights = model(x, dxdt, dt, smoothing)  # state: B x L x SD

            yhat = dx2hat

            y = dx2
            L = x.shape[1]

            if vis_freq > 0:
                if (idx % vis_freq) == 0:
                    sse_sample = sse(yhat[:1, :, :1], y[:1, :, :1])
                    sst_sample = sst(y[:1, :, :1])
                    r2_sample = (1 - sse_sample / sst_sample).item()

                    on = np.random.choice(L - 600)
                    resids = (
                        (y[0, :, 0] - yhat[0, :, 0]).detach().cpu().numpy()
                    )  # * dt**2
                    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
                        nrows=1, ncols=6, sharey=False, figsize=(20, 5)
                    )

                    ax1.plot(yhat[0, :, 0].detach().cpu().numpy(), label="model")
                    ax1.set_title("model")
                    ax1.set_ylabel("a.u.")
                    ax2.plot(
                        y[0, :, 0].detach().cpu().numpy(),
                        label="data",
                        color="tab:orange",
                    )
                    ax2.set_title("data")
                    ylims = ax2.get_ylim()
                    (l1,) = ax3.plot(
                        y[0, on + 300 : on + 350, 0].detach().cpu().numpy(),
                        label="data",
                        color="tab:orange",
                    )
                    (l2,) = ax3.plot(
                        yhat[0, on + 300 : on + 350, 0].detach().cpu().numpy(),
                        label="model",
                        color="tab:blue",
                    )
                    ax4.spines[["left", "right", "top", "bottom"]].set_visible(False)
                    ax4.set_xticks([])
                    ax4.set_yticks([])
                    ax4.legend([l1, l2], ["Data", "Model"])

                    ax5.plot(resids, label="res", color="tab:red")
                    ax5.set_title("residuals")
                    ax6.hist(resids, bins=100, density=True)
                    xlims = ax6.get_xlim()
                    sd = np.nanstd(resids)
                    def px(x):
                        return (1 / np.sqrt(2 * np.pi * sd**2)) * np.exp(-(x**2) / (2 * sd**2))
                    
                    xax = np.linspace(xlims[0], xlims[1], 1000)
                    yax = px(xax)
                    ax6.plot(xax, yax, color="tab:red")

                    ax1.set_ylim(ylims)
                    ax2.set_ylim(ylims)
                    ax3.set_ylim(ylims)
                    ax5.set_ylim(ylims)
                    ax6.set_xlim(xlims)

                    fig.suptitle(f"sample r2: {r2_sample: 0.4f}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(runDir, f"y_vs_yhat_batch_{idx}.svg"))
                    plt.close()

            ##################################

            train_loss = loss_fn(y, yhat[:, :L, :])

            # Default objective is the data loss; the polynomial parameterization adds a
            # weight-complexity penalty when reg_weights=True. Previously total_loss was
            # only defined inside the if-reg_weights block, which crashed unregularized runs.
            total_loss = train_loss
            if reg_weights:
                B, L, P, P = weights.shape
                lam_mat = torch.arange(
                    P, dtype=torch.float32, device=model.kernel.device
                )[None, None, :, None].expand(B, L, -1, P)

                w = model.kernel.lam ** (lam_mat + lam_mat.transpose(-1, -2))  # new

                penalty = (
                    (w * weights**2).sum(dim=(-1, -2, -3)).mean()
                )  # new (sum over weights, time), average over samples. equivalent to squared L2 norm
                # we take mean over samples to match the loss fn we use (MSE, with mean over samples)
                total_loss = train_loss + penalty

            total_loss.backward()
            optimizer.step()
            
            train_losses.append(train_loss.item())
            # we should probably be adding val loss here too...ugh
            writer.add_scalar("Loss/train", train_loss.item(), idx)
            if reg_weights:
                writer.add_scalar("Penalty/train", penalty.item(), idx)

        # The MSE-accel val loop runs only for the legacy mode (and only if a val loader
        # was provided). In spectral_rollout mode the autonomy-based selection in
        # train.model_cv handles validation, so we skip the body here -- but we MUST
        # fall through to the save_model block below, so do NOT use `continue` (it
        # would skip the rest of this epoch iteration including save_model).
        if (
            epoch % val_freq == 0
            and loss_mode != "spectral_rollout"
            and "val" in loaders
        ):
            model.eval()
            vl = 0.0
            vp = 0.0

            for idx, batch in enumerate(
                loaders["val"], start=epoch * len(loaders["train"])
            ):
                with torch.no_grad():
                    if len(batch) == 4:
                        x, dxdt, dx2dt2, _ = batch
                    else:
                        x, dxdt, dx2dt2 = batch
                    bsz, _, n = x.shape

                    x = x.to("cuda", non_blocking=True).to(torch.float32)
                    dxdt = dxdt.to("cuda", non_blocking=True).to(torch.float32)
                    dx2 = dx2dt2.to("cuda", non_blocking=True).to(torch.float32) / (dt**2) * model.tau**2

                    dx2hat, weights = model(x, dxdt, dt, smoothing)

                    yhat = dx2hat

                    y = dx2
                    L = y.shape[1]
                    if vis_freq > 0:
                        if idx == epoch * len(loaders["train"]):
                            sse_sample = sse(yhat[:1, :, :1], y[:1, :, :1])
                            sst_sample = sst(y[:1, :, :1])
                            r2_sample = (1 - sse_sample / sst_sample).item()

                            on = np.random.choice(L - 600)

                            resids = (
                                (y[0, on : on + 600, 0] - yhat[0, on : on + 600, 0])
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
                                nrows=1, ncols=6, sharey=False, figsize=(20, 5)
                            )

                            ax1.plot(
                                yhat[0, on : on + 600, 0].detach().cpu().numpy(),
                                label="model",
                            )
                            ax1.set_title("model")
                            ax1.set_ylabel("a.u.")
                            ax2.plot(
                                y[0, on : on + 600, 0].detach().cpu().numpy(),
                                label="data",
                                color="tab:orange",
                            )
                            ax2.set_title("data")
                            ylims = ax2.get_ylim()
                            (l1,) = ax3.plot(
                                y[0, on + 300 : on + 350, 0].detach().cpu().numpy(),
                                label="data",
                                color="tab:orange",
                            )
                            (l2,) = ax3.plot(
                                yhat[0, on + 300 : on + 350, 0].detach().cpu().numpy(),
                                label="model",
                                color="tab:blue",
                            )
                            ax4.spines[["left", "right", "top", "bottom"]].set_visible(
                                False
                            )
                            ax4.set_xticks([])
                            ax4.set_yticks([])
                            ax4.legend([l1, l2], ["Data", "Model"])

                            ax5.plot(resids, label="res", color="tab:red")
                            ax5.set_title("residuals")
                            ax6.hist(resids, bins=100, density=True)
                            xlims = ax6.get_xlim()
                            sd = np.nanstd(resids)
                            def px(x):
                                return (1 / np.sqrt(2 * np.pi * sd**2))* np.exp(-(x**2) / (2 * sd**2))
                            
                            xax = np.linspace(xlims[0], xlims[1], 1000)
                            yax = px(xax)
                            ax6.plot(xax, yax, color="tab:red")

                            ax1.set_ylim(ylims)
                            ax2.set_ylim(ylims)
                            ax3.set_ylim(ylims)
                            ax5.set_ylim(ylims)
                            ax6.set_xlim(xlims)
                            fig.suptitle(f"sample r2: {r2_sample: 0.4f}")
                            plt.tight_layout()
                            plt.savefig(
                                os.path.join(runDir, f"y_vs_yhat_batch_{idx}_test.svg")
                            )
                            plt.close()

                    val_loss = loss_fn(y, yhat[:, :L, :])
                    
                    if reg_weights:
                        B, L, P, P = weights.shape
                        lam_mat = torch.arange(
                            P, dtype=torch.float32, device=model.kernel.device
                        )[None, None, :, None].expand(B, L, -1, P)

                        exps = lam_mat + lam_mat.transpose(-1, -2)
                        w = model.kernel.lam**exps
                        penalty = (w * weights**2).sum(dim=(-1, -2)).mean()

                    vl += val_loss.item()

                    if reg_weights:
                        vp += penalty.item()

            if scheduler:
                scheduler.step(vl / len(loaders["val"]))
            val_losses.append(
                (
                    epoch * len(loaders["train"]),
                    vl / len(loaders["val"]),
                    vp / len(loaders["val"]),
                )
            )
            writer.add_scalar("Loss/validation", vl / len(loaders["val"]), idx)
            writer.add_scalar("Penalty/validation", vp / len(loaders["val"]), idx)

        # Periodic checkpoint -- OUTSIDE the val block so spectral_rollout mode (which
        # skips the val loop) still saves. Triggered by save_freq>0 only.
        if save_freq > 0 and (epoch % save_freq) == 0:
            save_model(
                model,
                optimizer,
                location=os.path.join(runDir, f"checkpoint_{epoch}.tar"),
                n_layers=model_info["n layers"],
                d_state=model_info["d state"],
                d_conv=model_info["d conv"],
                expand_factor=model_info["expand factor"],
                max_saved=max_saved,
            )
            _last_save_t = _time.time()  # reset so inflight save doesn't fire right after
    writer.close()
    return train_losses, val_losses, model, optimizer
