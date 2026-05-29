from train.train import train, save_model, load_model
from model.kernels import fullPolyModule
from model.model import Ouroboros, ArneodoOuroboros
from utils import sse
from visualization.model_vis import loss_plot
from train.eval import eval_model_error, autonomy_score

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
    drive_lowpass_ms: float = 0.0,
    n_seeds: int = 1,
    selection: str = "r2",
    val_vocs: list = None,
    test_vocs: list = None,
    keep_const: bool = False,
    rescale_autonomy: bool = False,
    lambdas: list = None,
    cull_frac: float = 0.0,
    cull_keep: int = 2,
) -> torch.nn.Module:
    """
    This function trains models and cross-validates across regularization strengths.
    We pick the minimum regularization strength to be 1.01 (to more heavily penalize
    more complex nonlinearities) and the largest to be 10**(4/(2*n_kernels)) (so that the regularization weight
    on the most complex term is 10**4). Saves these in a larger folder,
    alongside train stats, training plots, etc.

    Seed culling (cull_frac>0, selection='autonomy'): the autonomous-reconstruction quality is set by
    the random seed and its ranking settles well before R^2 plateaus, so instead of training every
    run fully, train all runs to `cull_frac` of n_epochs, rank by rescaled validation autonomy, and
    finish only the top `cull_keep`. The best finished run is selected. Roughly halves the seed search
    (see docs/autonomous_amplitude.md). With cull_frac=0 (default) every run is trained fully.

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

    # default: the standard 7-point lambda grid. Pass `lambdas=[x]` for a single fixed lambda
    # (e.g. the seed-selection production recipe, where lambda is known to be irrelevant for
    # autonomy and only the seed matters).
    if lambdas is None:
        min_lambda = 1.01
        max_lambda = 10 ** (4 / (2 * n_kernels))
        lambdas = np.linspace(min_lambda, max_lambda, 7)
    else:
        lambdas = np.asarray(lambdas, dtype=float)

    if selection == "autonomy" and not val_vocs:
        raise ValueError(
            "selection='autonomy' requires val_vocs (held-out vocalization segments)"
        )

    # --- per-(lambda, seed) training, resumable (a run with a checkpoint >= target is not retrained) ---
    def _train_to(lam, seed, target):
        """build/resume (lam, seed) and train to `target` epochs; return (model, run_dir)."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        kernel = fullPolyModule(nTerms=n_kernels, device="cuda", x_dim=1, z_dim=2,
                                activation=lambda x: x, lam=lam)
        model = Ouroboros(d_data=1, n_layers=n_layers, d_state=d_state, d_conv=d_conv,
                          expand_factor=expand_factor, tau=tau, smooth_len=smooth_len,
                          kernel=kernel, drive_lowpass_ms=drive_lowpass_ms, keep_const=keep_const)
        opt = Adam(model.parameters(), lr=lr)
        sched = ReduceLROnPlateau(opt, factor=0.5, patience=max(n_epochs // 25, 2), min_lr=1e-10)
        run_dir = os.path.join(model_path, f"poly_lam{lam:.3f}_seed{seed}")
        start_epoch = 0
        if glob.glob(os.path.join(run_dir, "*.tar")):
            model, opt, sched, start_epoch = load_model(run_dir)  # resume
            model.kernel.lam = float(lam)  # load_model hardcodes lam=1; restore it
        if start_epoch < target:
            train(model, opt, loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
                  loaders=dls, scheduler=sched, nEpochs=target, val_freq=1, runDir=run_dir,
                  dt=dt, vis_freq=0, smoothing=False, reg_weights=True, start_epoch=start_epoch,
                  save_freq=save_freq, model_info=model_info)
            save_model(model, opt, os.path.join(run_dir, f"checkpoint_{target}.tar"),
                       n_layers=n_layers, d_state=d_state, expand_factor=expand_factor, d_conv=d_conv)
        return model, run_dir

    def _score(model):
        model.eval()
        with torch.no_grad():
            (_, vr2), _, _ = eval_model_error(dls, model, dt=dt, comparison="val")
        if selection == "autonomy":
            va, _, bd = autonomy_score(model, val_vocs, dt, rescale=rescale_autonomy)
        else:
            va, bd = np.nan, None
        return vr2, va, bd

    n_jobs = n_seeds * len(lambdas)
    do_cull = (0.0 < cull_frac < 1.0) and selection == "autonomy" and cull_keep < n_jobs
    records = []
    if do_cull:
        # SEED CULLING: train every run to a cull epoch, rank by rescaled val autonomy, then finish
        # only the top `cull_keep`. The autonomy ranking is ~settled well before R^2 plateaus
        # (see docs/autonomous_amplitude.md), so this finds the best seed at a fraction of the cost.
        cull_epoch = max(1, int(round(cull_frac * n_epochs)))
        print(f"\n=== seed culling: train all {n_jobs} run(s) to epoch {cull_epoch} "
              f"({cull_frac:.0%} of {n_epochs}), then finish top {cull_keep} by rescaled val autonomy ===",
              flush=True)
        ranked = []
        for lam in lambdas:
            for seed in range(n_seeds):
                model, run_dir = _train_to(lam, seed, cull_epoch)
                _, va, _ = _score(model)
                ranked.append((va, lam, seed))
                print(f"  [cull@{cull_epoch}] lam={lam:.3f} seed={seed}: val autonomy={va:+.4f}", flush=True)
                del model; gc.collect(); torch.cuda.empty_cache()
        ranked.sort(key=lambda r: (r[0] if np.isfinite(r[0]) else -np.inf), reverse=True)
        keep, culled = ranked[:cull_keep], ranked[cull_keep:]
        print("  keep: " + ", ".join(f"lam{l:.3f}/seed{s}({a:+.3f})" for a, l, s in keep), flush=True)
        print("  cull: " + ", ".join(f"lam{l:.3f}/seed{s}({a:+.3f})" for a, l, s in culled), flush=True)
        for _, lam, seed in keep:
            model, run_dir = _train_to(lam, seed, n_epochs)
            vr2, va, _ = _score(model)
            print(f"  [final] lam={lam:.3f} seed={seed}: val R2={vr2:.4f}  val autonomy={va:+.4f}", flush=True)
            records.append({"lambda": lam, "seed": seed, "val_r2": vr2, "val_autonomy": va, "ckpt": run_dir})
            del model; gc.collect(); torch.cuda.empty_cache()
    else:
        for lam in lambdas:
            for seed in range(n_seeds):
                model, run_dir = _train_to(lam, seed, n_epochs)
                vr2, va, bd = _score(model)
                if selection == "autonomy":
                    print(f"lam={lam:.3f} seed={seed}: val R2={vr2:.4f}  val autonomy={va:+.4f} "
                          f"(spec={bd['spec_corr']:.2f} amp_pen={bd['amp_pen']:.2f} "
                          f"pitch_pen={bd['pitch_pen']:.2f} bounded={bd['bounded_frac']:.2f})", flush=True)
                else:
                    print(f"lam={lam:.3f} seed={seed}: val R2={vr2:.4f}", flush=True)
                records.append({"lambda": lam, "seed": seed, "val_r2": vr2, "val_autonomy": va, "ckpt": run_dir})
                del model; gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    metric = "val_autonomy" if selection == "autonomy" else "val_r2"
    df.to_csv(os.path.join(model_path, "lambda_seed_cv.csv"), index=False)

    if do_cull:
        # only the kept runs are finished, so select the single best finished run (a per-lambda
        # mean is not meaningful when most seeds were culled).
        best_row = df.sort_values(metric).iloc[-1]
        best_lambda = float(best_row["lambda"])
        best_ckpt = best_row["ckpt"]
        print(f"\n=== culled selection: best finished run lam={best_lambda:.3f} "
              f"seed={int(best_row['seed'])} {metric}={best_row[metric]:+.4f} ===", flush=True)
    else:
        per_lam = df.groupby("lambda")[metric].agg(["mean", "std"])
        best_lambda = float(per_lam["mean"].idxmax())
        print(f"\n=== lambda selection by {selection} (mean over {n_seeds} seed(s)) ===", flush=True)
        for lam, row in per_lam.iterrows():
            mark = "  <-- selected" if abs(lam - best_lambda) < 1e-9 else ""
            sd = 0.0 if np.isnan(row["std"]) else row["std"]
            print(f"  lambda={lam:.3f}: {metric}={row['mean']:+.4f} +- {sd:.4f}{mark}", flush=True)
        plt.figure()
        plt.errorbar(per_lam.index, per_lam["mean"], per_lam["std"].fillna(0.0), marker="o")
        plt.axvline(best_lambda, ls="--", color="0.6")
        plt.xlabel(r"kernel-weight $\lambda$")
        plt.ylabel(f"validation {metric}")
        plt.title(f"lambda selection by {selection} ({n_seeds} seeds)")
        plt.savefig(os.path.join(model_path, f"lambda_selection_{selection}.svg"))
        plt.close()
        # best model = best seed at the selected lambda (by the same validation metric)
        best_ckpt = df[np.isclose(df["lambda"], best_lambda)].sort_values(metric).iloc[-1]["ckpt"]

    best_model, _, _, _ = load_model(best_ckpt)
    best_model.eval()
    with torch.no_grad():
        (_, test_r2), _, _ = eval_model_error(dls, best_model, dt=dt, comparison="test")
    test_auto = np.nan
    if selection == "autonomy" and test_vocs:
        test_auto, _, _ = autonomy_score(best_model, test_vocs, dt, rescale=rescale_autonomy)
    print(f"BEST lambda={best_lambda:.3f}: test R2={test_r2:.4f}  test autonomy={test_auto:+.4f}",
          flush=True)
    best_model._selected_lambda = best_lambda  # authoritative selected lambda for callers/manifests
    return best_model


def train_arneodo(
    dls: dict,
    dt: float,
    n_epochs: int = 100,
    lr: float = 1e-3,
    expand_factor: int = 10,
    n_layers: int = 4,
    d_state: int = 1,
    d_conv: int = 4,
    tau: float = 1 / 1000,
    smooth_len: float = 0.001,
    model_path: str = "",
    save_freq: int = 5,
    drive_lowpass_ms: float = 1.0,
) -> torch.nn.Module:
    """
    trains a single `ArneodoOuroboros` model (the biomechanical syrinx parameterization).

    Unlike `model_cv_lambdas`, there is no regularization-strength cross-validation: the
    Arneodo RHS has no polynomial kernel weights to penalize, so we fit one model with
    `reg_weights=False`. The trained model is saved and returned in memory.

    inputs
    -----
        - dls: dictionary of dataloaders (train / val / test)
        - dt: spacing between audio samples, in seconds
        - n_epochs: number of passes through the training data
        - lr: learning rate
        - expand_factor: expansion from audio to mamba input
        - n_layers: number of mamba layers in each encoder
        - d_state: internal state size of mamba model
        - d_conv: length of internal convolution of mamba model
        - tau: timescale for model decoder, to de-dimensionalize the data
        - smooth_len: smoothing length for model functions (not used in training)
        - model_path: place to save the model and training artifacts
        - save_freq: how often (in epochs) to checkpoint the model
        - drive_lowpass_ms: hard low-pass timescale (ms) on the alpha/beta/delta drives.
            Defaults to 1 ms (slow, physiological drives + cold-start-stable autonomous
            dynamics, at a teacher-forced R^2 cost). Set 0.0 for the unregularized model.

    returns
    -----
        - the trained ArneodoOuroboros
    """

    model_info = {
        "n layers": n_layers,
        "d state": d_state,
        "d conv": d_conv,
        "expand factor": expand_factor,
    }

    model = ArneodoOuroboros(
        d_data=1,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand_factor=expand_factor,
        tau=tau,
        smooth_len=smooth_len,
        drive_lowpass_ms=drive_lowpass_ms,
    )

    opt = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        opt, factor=0.5, patience=max(n_epochs // 25, 2), min_lr=1e-10
    )

    run_dir = os.path.join(model_path, "arneodo")
    os.makedirs(run_dir, exist_ok=True)
    save_loc = os.path.join(run_dir, f"checkpoint_{n_epochs}.tar")

    tl, vl, model, opt = train(
        model,
        opt,
        loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
        loaders=dls,
        scheduler=scheduler,
        nEpochs=n_epochs,
        val_freq=1,
        runDir=run_dir,
        dt=dt,
        vis_freq=max(n_epochs // 10, 1),
        smoothing=False,
        reg_weights=False,  # no kernel weights to regularize in the Arneodo RHS
        start_epoch=0,
        save_freq=save_freq,
        model_info=model_info,
    )

    loss_plot(tl, vl, save_loc=run_dir, show=False)

    save_model(
        model,
        opt,
        save_loc,
        n_layers=n_layers,
        d_state=d_state,
        expand_factor=expand_factor,
        d_conv=d_conv,
    )

    model.eval()
    with torch.no_grad():
        (train_mu, test_mu), (train_sd, test_sd), _ = eval_model_error(
            dls, model, dt=dt, comparison="test"
        )
    print(
        f"Arneodo model R2 -- train: {train_mu:.4f} +- {train_sd:.4f}, "
        f"test: {test_mu:.4f} +- {test_sd:.4f}"
    )

    return model
