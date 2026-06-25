from train.train import train, save_model, load_model
from model.kernels import fullPolyModule
from model.model import Ouroboros
from utils import sse
from visualization.model_vis import loss_plot
from train.eval import eval_model_error

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import glob
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import seaborn as sns


def model_cv_lambdas(
    dls: dict,
    dt: float,
    n_epochs: int = 100,
    lr: float = 1e-3,
    n_kernels: int = 15,
    expand_factor: int = 10,
    n_layers: int = 4,
    d_state: int = 1,
    d_conv: int = 4,
    tau: float = 1 / 1000,
    smooth_len: float = 0.001,
    model_path: str = "",
    save_freq: int = 5,
) -> torch.nn.Module:
    """
    This function trains models and cross-validates across regularization strengths.
    We pick the minimum regularization strength to be 1.01 (to more heavily penalize
    more complex nonlinearities) and the largest to be 10**(4/(2*n_kernels)) (so that the regularization weight
    on the most complex term is 10**4). Saves these in a larger folder,
    alongside train stats, training plots, etc.

    inputs
    -----
        - dls: dictionary of dataloaders, train and test
        - dt: spacing between audio samples, in seconds
        - nEpochs: number of full passes through the dataset to train the model for
        - lr: learning rate
        - n_kernels: maximum polynomial degree for the nonlinearity
        - expand_factor: expansion from audio to mamba input
        - n_layers: number of mamba layers in encoder
        - d_state: internal state size of mamba model
        - d_conv: length of internal convolution of mamba model
        - tau: timescale for model decoder, to de-dimensionalize the data
        - smooth_len: smoothing length for model functions. not used in training
        - model_path: place to save all the models
        - save_freq: how often to save out our models

    returns
    -----
        - the best model (on the test set)
    """

    model_info = {
        "n layers": n_layers,
        "d state": d_state,
        "d conv": d_conv,
        "expand factor": expand_factor,
    }

    min_lambda = 1.01
    max_lambda = 10 ** (4 / (2 * n_kernels))

    lambdas = np.linspace(min_lambda, max_lambda, 7)
    #
    lam_train_cv_err = []
    lam_test_cv_err = []

    lam_train_cv_sd = []
    lam_test_cv_sd = []

    lam_train_cv_r2 = []
    lam_test_cv_r2 = []

    for ii, lam in enumerate(lambdas):
        print(f"Regularizing with lambda={lam}")

        kernel = fullPolyModule(
            nTerms=n_kernels,
            device="cuda",
            x_dim=1,
            z_dim=2,
            activation=lambda x: x,
            lam=lam,
        )
        reg_weights = True
        full_model_poly = Ouroboros(
            d_data=1,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            tau=tau,
            smooth_len=smooth_len,
            kernel=kernel,
        )

        full_opt_poly = Adam(full_model_poly.parameters(), lr=lr)
        full_scheduler_poly = ReduceLROnPlateau(
            full_opt_poly, factor=0.5, patience=max(n_epochs // 25, 2), min_lr=1e-10
        )
        model_path_full_poly = (
            model_path + f"/kernelborous_poly_end_to_end_lambda_{lam}"
        )
        save_loc_poly = model_path_full_poly + f"/checkpoint_{n_epochs}.tar"
        save_files = glob.glob(os.path.join(model_path_full_poly, "*.tar"))

        start_epoch = 0
        if len(save_files) > 0:
            full_model_poly, full_opt_poly, full_scheduler_poly, start_epoch = (
                load_model(model_path_full_poly)
            )

        if start_epoch < n_epochs:
            tl, vl, full_model_poly, full_opt_poly = train(
                full_model_poly,
                full_opt_poly,
                loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
                loaders=dls,
                scheduler=full_scheduler_poly,
                nEpochs=n_epochs,
                val_freq=1,
                runDir=model_path_full_poly,
                dt=dt,
                vis_freq=max(n_epochs // 10, 1),
                smoothing=False,
                reg_weights=reg_weights,
                start_epoch=start_epoch,
                save_freq=save_freq,
                model_info=model_info,
            )

            loss_plot(tl, vl, save_loc=model_path_full_poly, show=False)

            save_model(
                full_model_poly,
                full_opt_poly,
                save_loc_poly,
                n_layers=n_layers,
                d_state=d_state,
                expand_factor=expand_factor,
                d_conv=d_conv,
            )

        full_model_poly.eval()
        with torch.no_grad():
            (train_mu, test_mu), (train_sd, test_sd), (train_r2, test_r2) = (
                eval_model_error(dls, full_model_poly, dt=dt)
            )
        lam_train_cv_err.append(train_mu)
        lam_test_cv_err.append(test_mu)

        lam_train_cv_sd.append(train_sd)
        lam_test_cv_sd.append(test_sd)

        lam_train_cv_r2.append(train_r2)
        lam_test_cv_r2.append(test_r2)

    splits = ["train"] * len(lam_train_cv_err) + ["val"] * len(lam_test_cv_err)
    lambdas_stacked = np.round(np.hstack([lambdas, lambdas]), 3)
    errs = np.hstack([lam_train_cv_err, lam_test_cv_err])
    df = pd.DataFrame({"lam": lambdas_stacked, "split": splits, "R2": errs})

    min_err_ind = np.argmax(lam_test_cv_err)  # argmax, since 'err' is actually r2
    print(f"best R2 alpha for {n_kernels} kernels: {lambdas[min_err_ind]}")
    ax = plt.gca()

    sns.boxplot(
        data=df,
        x="lam",
        y="R2",
        hue="split",
        hue_order=["train", "test"],
        ax=ax,
        gap=0.1,
    )

    ax.set_xlabel("Polynomial degree penalty")
    ax.set_ylabel(r"$R^2$")
    ylim = ax.get_ylim()
    ylim = (min(ylim[0], 0), max(ylim[-1], 1.01))
    ax.set_ylim(ylim)
    ax.legend()
    plt.savefig(
        os.path.join(model_path, "train_test_error_kernel_poly_nkernels_30.svg")
    )
    plt.close()

    model_path_best = (
        model_path + f"/kernelborous_poly_end_to_end_lambda_{lambdas[min_err_ind]}"
    )

    full_model_poly, full_opt_poly, full_scheduler_poly, _ = load_model(model_path_best)
    full_model_poly.eval()
    with torch.no_grad():
        (train_mu, test_mu), (train_sd, test_sd), (train_r2, test_r2) = (
            eval_model_error(dls, full_model_poly, dt=dt, comparison="test")
        )

    data_df = pd.DataFrame(
        {"lambdas": lambdas, "train MSE": lam_train_cv_err, "test MSE": lam_test_cv_err}
    )
    data_df.to_csv(os.path.join(model_path, "cv_errs.csv"))

    return full_model_poly


# --- Seed loop + culling + cold-start autonomy selection ---------------------------------

def model_seed_cv_spectral(
    dls: dict,
    dt: float,
    val_vocs: list,
    test_vocs: list,
    *,
    # model capacity
    n_kernels: int = 15,
    n_layers: int = 4,
    d_state: int = 4,
    d_conv: int = 4,
    expand_factor: int = 4,
    tau: float = None,             # default 1/sr if None
    smooth_len: float = 0.001,
    drive_lowpass_ms: float = 1.0,
    keep_const: bool = False,
    osc_init: bool = False,        # Strategy 1: van der Pol limit-cycle init (see Ouroboros.__init__)
    checkpoint_encoder: bool = False,  # gradient-checkpoint the Mamba drive encoders (memory for larger B)
    use_tract: bool = False,       # opt-in: linear vocal-tract filter (FFT-domain H, identity at init)
    tract_n_sec: int = 3,          # number of second-order pole/zero sections in the tract
    use_envelope: bool = False,    # opt-in: learnable amplitude envelope head e(t)=exp(lowpass(.))
    env_lowpass_ms: float = 20.0,
    lam: float = 1.068,            # fixed kernel-weight lambda (no CV in this PR)
    # training
    n_epochs: int = 50,
    lr: float = 1e-3,
    n_seeds: int = 4,
    cull_frac: float = 0.0,
    cull_keep: int = 2,
    save_freq: int = 5,
    max_saved: int = 5,
    model_path: str = "",
    # spectral loss
    H_min: int = 512,
    H_max: int = 2048,
    H_schedule: str = "geom",
    lam_spec: float = 1.0,
    lam_tf: float = 1.0,
    lam_env: float = 0.0,
    lam_env_log: float = 0.0,
    env_log_eps: float = 1e-4,
    env_ms: float = 2.0,
    lam_reg: float = 0.0,
    lam_env_anchor: float = 0.0,
    spec_warmup_epochs: int = 5,
    env_warmup_epochs: int = 0,
    spec_warmup_steps: int = None,
    env_warmup_steps: int = None,
    H_total_steps: int = None,
    spec_configs=None,
    ic_noise_rms: float = 1e-3,
    grad_clip: float = 5.0,
    rollout_backend: str = "eager",
    # linear LR ramp (constant LR when lr_end is None)
    lr_end: float = None,
    lr_ramp_epochs: int = 5,
    # intra-epoch save cadence in minutes; 0 disables
    save_minutes: float = 0.0,
    # tract.K_raw initial value (None = leave at 0 → K = softplus(0) = log(2)).
    # When set, the entry script picks this from a quick RMS scan of the training
    # audio so K = softplus(K_raw) matches target audio scale from epoch 0.
    K_raw_init: float = None,
    # selection
    rescale_autonomy: bool = False,
    cold_start_autonomy: bool = True,
) -> torch.nn.Module:
    """Seed loop for the spectral-rollout polynomial Ouroboros, with optional
    culling and cold-start-raw autonomy selection.

    Trains `n_seeds` independent random inits at a fixed kernel-weight `lam`, all
    with loss_mode='spectral_rollout' (see train/spectral_rollout.py). Each seed
    integrates over the edge-biased training set; validation uses cold-start
    autonomy (held-out vocs `val_vocs` already include the silence lead-in -- the
    integration IC then equals near-silence and exercises the ignition path).

    If `cull_frac` in (0, 1) and `cull_keep < n_seeds`, train all seeds to
    `cull_frac * n_epochs`, rank by val cold-start autonomy, and finish only the
    top `cull_keep`. Roughly halves the seed-search cost (see
    docs/autonomous_amplitude.md on main branch).

    Returns the best seed by val autonomy. Saves a `seed_cv.csv` summary and
    sets `best._selected_seed`, `best._selected_test_autonomy`,
    `best._selected_breakdown` for the caller's manifest.
    """
    from train.eval import autonomy_score

    model_info = {
        "n layers": n_layers, "d state": d_state, "d conv": d_conv,
        "expand factor": expand_factor,
    }
    assert tau is not None, "model_seed_cv_spectral requires explicit tau (e.g. 1/sr)"

    def _build(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        kernel = fullPolyModule(nTerms=n_kernels, device="cuda", x_dim=1, z_dim=2,
                                activation=lambda x: x, lam=float(lam))
        model = Ouroboros(d_data=1, n_layers=n_layers, d_state=d_state, d_conv=d_conv,
                          expand_factor=expand_factor, tau=tau, smooth_len=smooth_len,
                          kernel=kernel, drive_lowpass_ms=drive_lowpass_ms,
                          keep_const=keep_const, osc_init=osc_init,
                          checkpoint_encoder=checkpoint_encoder,
                          use_tract=use_tract, tract_n_sec=tract_n_sec,
                          use_envelope=use_envelope, env_lowpass_ms=env_lowpass_ms)
        # K_raw_init: set the tract gain so audio amplitude starts near target RMS
        # at epoch 0, instead of relying on it to descend from K_raw=0 during training.
        # Only applies on fresh start; resume restores the trained value.
        if use_tract and K_raw_init is not None:
            with torch.no_grad():
                model.tract.K_raw.data.fill_(float(K_raw_init))
        opt = Adam(model.parameters(), lr=lr)
        sched = ReduceLROnPlateau(opt, factor=0.5, patience=max(n_epochs // 25, 2),
                                  min_lr=1e-10)
        return model, opt, sched

    def _train_to(seed, target):
        run_dir = os.path.join(model_path, f"seed{seed}")
        os.makedirs(run_dir, exist_ok=True)
        if glob.glob(os.path.join(run_dir, "*.tar")):
            model, opt, sched, start_epoch = load_model(run_dir)
            model.kernel.lam = float(lam)
            # load_model restores the opt's saved LR from the ckpt; override here so
            # an --lr passed at resume time actually takes effect.
            for g in opt.param_groups:
                g['lr'] = lr
        else:
            model, opt, sched = _build(seed)
            start_epoch = 0
        if start_epoch < target:
            train(
                model, opt,
                loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),  # unused in spectral mode
                loaders=dls, scheduler=sched,
                nEpochs=target, val_freq=max(1, target // 10), runDir=run_dir,
                dt=dt, vis_freq=0, smoothing=False, reg_weights=False,
                start_epoch=start_epoch, save_freq=save_freq, max_saved=max_saved,
                model_info=model_info,
                loss_mode="spectral_rollout",
                H_min=H_min, H_max=H_max, H_schedule=H_schedule,
                lam_spec=lam_spec, lam_tf=lam_tf, lam_env=lam_env,
                lam_env_log=lam_env_log, env_log_eps=env_log_eps,
                env_ms=env_ms,
                lam_reg=lam_reg,
                lam_env_anchor=lam_env_anchor,
                spec_warmup_epochs=spec_warmup_epochs,
                env_warmup_epochs=env_warmup_epochs,
                spec_warmup_steps=spec_warmup_steps,
                env_warmup_steps=env_warmup_steps,
                H_total_steps=H_total_steps,
                spec_configs=spec_configs, ic_noise_rms=ic_noise_rms,
                grad_clip=grad_clip, rollout_backend=rollout_backend,
                lr_end=lr_end, lr_ramp_epochs=lr_ramp_epochs,
                save_minutes=save_minutes,
            )
            save_model(model, opt, os.path.join(run_dir, f"checkpoint_{target}.tar"),
                       n_layers=n_layers, d_state=d_state, expand_factor=expand_factor,
                       d_conv=d_conv, max_saved=max_saved)
        return model, run_dir

    def _autonomy(model, vocs):
        model.eval()
        with torch.no_grad():
            score, _, bd = autonomy_score(model, vocs, dt,
                                          rescale=rescale_autonomy,
                                          cold_start=cold_start_autonomy)
        return score, bd

    records = []
    do_cull = (0.0 < cull_frac < 1.0) and cull_keep < n_seeds
    if do_cull:
        cull_epoch = max(1, int(round(cull_frac * n_epochs)))
        metric_tag = ("cold-start-raw" if cold_start_autonomy
                      else ("rescaled" if rescale_autonomy else "raw"))
        print(f"\n=== seed culling: train all {n_seeds} seeds to epoch {cull_epoch} "
              f"({cull_frac:.0%} of {n_epochs}), then finish top {cull_keep} by "
              f"{metric_tag} val autonomy ===", flush=True)
        ranked = []
        for seed in range(n_seeds):
            model, run_dir = _train_to(seed, cull_epoch)
            va, bd = _autonomy(model, val_vocs)
            ranked.append((va, seed, run_dir))
            print(f"  [cull@{cull_epoch}] seed={seed}: val autonomy={va:+.4f} "
                  f"(spec={bd['spec_corr']:.2f} amp_pen={bd['amp_pen']:.2f} "
                  f"pitch_pen={bd['pitch_pen']:.2f} bounded={bd['bounded_frac']:.2f})",
                  flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
        ranked.sort(key=lambda r: (r[0] if np.isfinite(r[0]) else -np.inf), reverse=True)
        keep = ranked[:cull_keep]
        culled = ranked[cull_keep:]
        print("  keep: " + ", ".join(f"seed{s}({a:+.3f})" for a, s, _ in keep), flush=True)
        if culled:
            print("  cull: " + ", ".join(f"seed{s}({a:+.3f})" for a, s, _ in culled),
                  flush=True)
        for _, seed, _ in keep:
            model, run_dir = _train_to(seed, n_epochs)
            va, bd = _autonomy(model, val_vocs)
            print(f"  [final] seed={seed}: val autonomy={va:+.4f} "
                  f"(spec={bd['spec_corr']:.2f} amp_pen={bd['amp_pen']:.2f} "
                  f"pitch_pen={bd['pitch_pen']:.2f} bounded={bd['bounded_frac']:.2f})",
                  flush=True)
            records.append({"seed": seed, "val_autonomy": va, "ckpt": run_dir,
                            **{f"val_{k}": v for k, v in bd.items()}})
            del model; gc.collect(); torch.cuda.empty_cache()
    else:
        for seed in range(n_seeds):
            model, run_dir = _train_to(seed, n_epochs)
            va, bd = _autonomy(model, val_vocs)
            print(f"seed={seed}: val autonomy={va:+.4f} "
                  f"(spec={bd['spec_corr']:.2f} amp_pen={bd['amp_pen']:.2f} "
                  f"pitch_pen={bd['pitch_pen']:.2f} bounded={bd['bounded_frac']:.2f})",
                  flush=True)
            records.append({"seed": seed, "val_autonomy": va, "ckpt": run_dir,
                            **{f"val_{k}": v for k, v in bd.items()}})
            del model; gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(model_path, "seed_cv.csv"), index=False)
    # tie-breaker: max bounded_frac among the top val_autonomy (so collapsed-but-lucky
    # rollouts don't beat truly bounded ones if they happen to land at the same score)
    best_row = df.sort_values(["val_autonomy", "val_bounded_frac"]).iloc[-1]
    best_ckpt = best_row["ckpt"]
    print(f"\n=== seed selection: best seed={int(best_row['seed'])} "
          f"val_autonomy={best_row['val_autonomy']:+.4f} ===", flush=True)

    best_model, _, _, _ = load_model(best_ckpt)
    best_model.eval()
    test_score, test_bd = _autonomy(best_model, test_vocs) if test_vocs else (float("nan"), {})
    print(f"BEST seed={int(best_row['seed'])}: test autonomy={test_score:+.4f}",
          flush=True)
    best_model._selected_seed = int(best_row["seed"])
    best_model._selected_lambda = float(lam)
    best_model._selected_val_autonomy = float(best_row["val_autonomy"])
    best_model._selected_test_autonomy = float(test_score)
    best_model._selected_test_breakdown = dict(test_bd)
    return best_model
